"""
Per-Building Peak Hour Analysis
================================
Scans every building under a decarb case directory for ts.csv files,
and/or ResStock baseline parquets, finds each building's summer peak
hour-of-day, and produces histograms showing when individual buildings peak.

This diagnoses whether the morning-peak issue is in the raw ts.csv
data or introduced during aggregation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from concurrent.futures import ProcessPoolExecutor, as_completed


# ── Configuration ──────────────────────────────────────────
CASE_NAMES = [
    # "elec_0_flat_results",
    # "elec_100_flat_results",
    # "elec_0_tou_results",
    # "elec_100_tou_results",
    # "elec_0_flat_cap_results",
    # "elec_100_flat_cap_results",
    "elec_0_tou_cap_results",
    # "elec_100_tou_cap_results",
]

ERCOT_ROOT = r"D:\shared\ercot_project"
UPGRADE = "update_0"
YEAR = 2018
N_WORKERS = 20

RERUN_ANALYSIS = False  # If False, load existing CSVs instead of recomputing

RUN_DECARB = True
RUN_RESSTOCK = False

RESSTOCK_DIR = os.path.join(ERCOT_ROOT, "out", "consumption_files")

PEAK_PERIODS = [
    ("05:00", "09:00"),   # morning: 6am-10am (4 hrs)
    ("18:00", "22:00"),   # evening: 6pm-10pm (4 hrs)
]
# ───────────────────────────────────────────────────────────

SUMMER_START = 3624
SUMMER_END = 5856

METRIC_LABELS = {
    "heat": "Heating",
    "cool": "Cooling",
    "elec": "Total Electricity",
}
METRIC_COLORS = {
    "heat": "#c0504d",   # red
    "cool": "#1f4e79",   # blue
    "elec": "#4a90d9",   # light blue
}


# =============================================================================
# Shared helpers
# =============================================================================

TIME_PERIODS = {
    "Night (12am-6am)":       (0, 5),
    "Morning (6am-10am)":     (6, 9),
    "Midday (10am-2pm)":      (10, 13),
    "Afternoon (2pm-6pm)":    (14, 17),
    "Evening (6pm-10pm)":     (18, 21),
    "Late Night (10pm-12am)": (22, 23),
}


def _is_peak_hour(hour):
    for start_str, end_str in PEAK_PERIODS:
        h0 = int(start_str.split(":")[0])
        h1 = int(end_str.split(":")[0])
        if h0 <= hour < h1:
            return True
    return False


def _peak_period_spans():
    spans = []
    for start_str, end_str in PEAK_PERIODS:
        h0 = int(start_str.split(":")[0])
        h1 = int(end_str.split(":")[0])
        spans.append((h0, h1))
    return spans


def _is_tou_case(case_name):
    """True if the case uses a TOU tariff (regardless of capacity charge)."""
    return "tou" in case_name.lower()


def _title_from_case_name(case_name):
    """Build the figure title from the case name.

    Tariff and capacity charge are independent dimensions:
      - 'tou'  → 'TOU Tariff'   else  'Flat Tariff'
      - 'cap'  → append ' w/ CC'
    Electrification:
      - '100'  → '100% Heating Electrification'
      - else   → '60% Heating Electrification'
    """
    name = case_name.lower()

    tariff = "TOU Tariff" if "tou" in name else "Flat Tariff"
    if "cap" in name:
        tariff += " w/ CC"

    if "100" in name:
        electrification = "100% Heating Electrification"
    else:
        electrification = "60% Heating Electrification"

    return f"DECARB Peak Analysis - {tariff} - {electrification}"


def _load_weights(output_dir, case_dir):
    """Load building_mapping.csv from output_dir (fallback: case_dir) and
    return {bldg_id (int): weight (int)}. Weight comes from the 'count'
    column, summed if the same new_bldg_id appears in multiple rows.
    Returns an empty dict if the file isn't found."""
    candidates = [
        os.path.join(output_dir, "building_mapping.csv"),
        os.path.join(case_dir, "building_mapping.csv"),
    ]
    mapping_path = next((p for p in candidates if os.path.exists(p)), None)
    if mapping_path is None:
        print("  WARNING: building_mapping.csv not found — histograms will "
              "use unweighted building counts.")
        return {}

    m = pd.read_csv(mapping_path)
    if "new_bldg_id" not in m.columns or "count" not in m.columns:
        print(f"  WARNING: {mapping_path} missing 'new_bldg_id' or 'count' "
              "column — histograms will be unweighted.")
        return {}

    m = m[["new_bldg_id", "count"]].dropna()
    m["new_bldg_id"] = m["new_bldg_id"].astype(int)
    m["count"] = m["count"].astype(float)
    grouped = m.groupby("new_bldg_id")["count"].sum()
    print(f"  Loaded weights from {mapping_path} — "
          f"{len(grouped):,} buildings, total dwellings = "
          f"{int(grouped.sum()):,}")
    return grouped.astype(int).to_dict()


def _hour_counts_weighted(hour_series, bldg_id_series, weight_map):
    """Sum weights per hour-of-day (0..23). If weight_map is empty, falls
    back to raw counts."""
    counts = np.zeros(24, dtype=np.int64)
    if not weight_map:
        vc = hour_series.value_counts()
        for h, c in vc.items():
            counts[int(h)] += int(c)
        return counts
    # Vectorized lookup; missing IDs default to weight 1.
    weights = bldg_id_series.astype(int).map(weight_map).fillna(1).astype(int)
    for h, w in zip(hour_series.astype(int).values, weights.values):
        counts[h] += w
    return counts


def compute_peaks_from_hourly(buy, n_hours):
    full_peak_idx = int(np.argmax(buy))
    full_peak_val = buy[full_peak_idx]
    full_peak_hour = full_peak_idx % 24

    summer_end = min(SUMMER_END, n_hours)
    summer_buy = buy[SUMMER_START:summer_end]
    if len(summer_buy) == 0:
        return None

    summer_peak_local_idx = int(np.argmax(summer_buy))
    summer_peak_global_idx = SUMMER_START + summer_peak_local_idx
    summer_peak_val = summer_buy[summer_peak_local_idx]
    summer_peak_hour = summer_peak_global_idx % 24

    winter_buy = np.concatenate([buy[:2160], buy[6552:min(8760, n_hours)]])
    winter_peak_local_idx = int(np.argmax(winter_buy))
    winter_peak_val = winter_buy[winter_peak_local_idx]
    if winter_peak_local_idx < 2160:
        winter_peak_global_idx = winter_peak_local_idx
    else:
        winter_peak_global_idx = 6552 + (winter_peak_local_idx - 2160)
    winter_peak_hour = winter_peak_global_idx % 24

    return {
        "n_hours": n_hours,
        "full_peak_idx": full_peak_idx,
        "full_peak_val": full_peak_val,
        "full_peak_hour": full_peak_hour,
        "summer_peak_idx": summer_peak_global_idx,
        "summer_peak_val": summer_peak_val,
        "summer_peak_hour": summer_peak_hour,
        "winter_peak_idx": winter_peak_global_idx,
        "winter_peak_val": winter_peak_val,
        "winter_peak_hour": winter_peak_hour,
    }


def _classify_peak_season(peak_idx, n_hours):
    """Return 'summer' or 'winter' for the annual peak index.

    Summer = SUMMER_START..SUMMER_END (Jun-Aug, indices 3624-5856).
    Winter = Jan-Mar (0..2160) + Oct-Dec (6552..8760).
    Shoulder months (Apr-May = 2160..3624, Sep = 5856..6552) are snapped
    to the nearer season by distance from the shoulder midpoint.
    """
    if SUMMER_START <= peak_idx < min(SUMMER_END, n_hours):
        return "summer"
    if peak_idx < 2160 or peak_idx >= 6552:
        return "winter"
    # Shoulder: snap to nearer season.
    if peak_idx < SUMMER_START:           # Apr-May (2160..3624)
        midpoint = (2160 + SUMMER_START) / 2   # ~2892
        return "winter" if peak_idx < midpoint else "summer"
    else:                                  # Sep (5856..6552)
        midpoint = (SUMMER_END + 6552) / 2     # ~6204
        return "summer" if peak_idx < midpoint else "winter"


def _load_avg_peak_day_profile(output_dir, case_name):
    """Load aggregated timeseries and return the average 24-hour day profile."""
    agg_path = os.path.join(output_dir, f"{case_name}_aggregated_ts.csv")
    if not os.path.exists(agg_path):
        return None
    agg_df = pd.read_csv(agg_path)
    series = agg_df.iloc[:, 0].values  # Grid Purchases column
    n_days = len(series) // 24
    daily = series[:n_days * 24].reshape(n_days, 24)
    return daily.mean(axis=0)


def print_peak_summary(df, label):
    print(f"\n{'=' * 60}")
    print(f"  {label} — SUMMER PEAK HOUR DISTRIBUTION")
    print(f"{'=' * 60}")
    summer_counts = df["summer_peak_hour"].value_counts().sort_index()
    for hour in range(24):
        c = summer_counts.get(hour, 0)
        pct = 100 * c / len(df)
        bar = "█" * int(pct)
        peak_marker = " ◄ PEAK" if _is_peak_hour(hour) else ""
        print(f"  {hour:02d}:00  {c:>5d}  ({pct:5.1f}%)  {bar}{peak_marker}")

    print(f"\n{'=' * 60}")
    print(f"  {label} — WINTER PEAK HOUR DISTRIBUTION")
    print(f"{'=' * 60}")
    winter_counts = df["winter_peak_hour"].value_counts().sort_index()
    for hour in range(24):
        c = winter_counts.get(hour, 0)
        pct = 100 * c / len(df)
        bar = "█" * int(pct)
        peak_marker = " ◄ PEAK" if _is_peak_hour(hour) else ""
        print(f"  {hour:02d}:00  {c:>5d}  ({pct:5.1f}%)  {bar}{peak_marker}")

    def period_summary(hour_series, sublabel):
        print(f"\n  {sublabel} by Time Period:")
        total = len(hour_series)
        for name, (h0, h1) in TIME_PERIODS.items():
            c = ((hour_series >= h0) & (hour_series <= h1)).sum()
            pct = 100 * c / total
            bar = "█" * int(pct / 2)
            print(f"    {name:<28s} {c:>5d}  ({pct:5.1f}%)  {bar}")

    period_summary(df["summer_peak_hour"], "SUMMER PEAKS")
    period_summary(df["winter_peak_hour"], "WINTER PEAKS")
    period_summary(df["full_peak_hour"], "FULL YEAR PEAKS")


def plot_peak_histograms(df, label, output_path, output_dir, case_name,
                         weight_map=None):
    """Render the 4-panel peak analysis figure.

    The top-row histograms (Summer / Winter peak hour) are weighted by
    `weight_map` (bldg_id -> dwellings represented) so the y-axis reflects
    the dwelling-weighted population rather than raw simulated buildings.
    """
    weight_map = weight_map or {}
    summer_counts = _hour_counts_weighted(
        df["summer_peak_hour"], df["bldg_id"], weight_map
    )
    winter_counts = _hour_counts_weighted(
        df["winter_peak_hour"], df["bldg_id"], weight_map
    )
    spans = _peak_period_spans()
    is_tou = _is_tou_case(case_name)
    y_label = "Number of Dwellings" if weight_map else "Number of Buildings"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(_title_from_case_name(case_name), fontsize=14, fontweight="bold")

    # 1) Summer peak hour histogram (upper-left)
    ax = axes[0, 0]
    if is_tou:
        colors = ["#e74c3c" if _is_peak_hour(h) else "#4a90d9" for h in range(24)]
    else:
        colors = ["#4a90d9"] * 24
    ax.bar(range(24), summer_counts, color=colors, edgecolor="white", lw=0.5)
    ax.set_title("Summer Peak Hour (Jun-Aug)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    if is_tou:
        for h0, h1 in spans:
            ax.axvspan(h0 - 0.5, h1 - 0.5, alpha=0.1, color="red")
        ax.legend(handles=[
            Patch(facecolor="#e74c3c", label="Peak period"),
            Patch(facecolor="#4a90d9", label="Off-peak"),
        ], fontsize=8)

    # 2) Winter peak hour histogram (upper-right)
    ax = axes[0, 1]
    if is_tou:
        colors_w = ["#e74c3c" if _is_peak_hour(h) else "#4a90d9" for h in range(24)]
    else:
        colors_w = ["#4a90d9"] * 24
    ax.bar(range(24), winter_counts, color=colors_w, edgecolor="white", lw=0.5)
    ax.set_title("Winter Peak Hour (Jan-Mar, Oct-Dec)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    if is_tou:
        for h0, h1 in spans:
            ax.axvspan(h0 - 0.5, h1 - 0.5, alpha=0.1, color="red")
        ax.legend(handles=[
            Patch(facecolor="#e74c3c", label="Peak period"),
            Patch(facecolor="#4a90d9", label="Off-peak"),
        ], fontsize=8)

    # 3) Average daily profile (lower-left)
    ax = axes[1, 0]
    avg_profile = _load_avg_peak_day_profile(output_dir, case_name)
    if avg_profile is not None:
        ax.plot(range(24), avg_profile, "o-", color="#333333", lw=2,
                markersize=4, zorder=3, label="Avg load")
        if is_tou:
            for h0, h1 in spans:
                ax.axvspan(h0, h1, alpha=0.15, color="red", zorder=1)
        avg_peak_hour = int(np.argmax(avg_profile))
        ax.axvline(avg_peak_hour, color="#e74c3c", ls="--", alpha=0.7,
                   label=f"Avg peak at {avg_peak_hour}:00")
        ax.scatter(
            [avg_peak_hour], [avg_profile[avg_peak_hour]],
            color="#e74c3c", s=150, zorder=5, marker="*",
            edgecolors="black", linewidths=0.5,
            label="Peak hour",
        )
        ax.set_title("Average Daily Profile (all days)")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("GW")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Aggregated timeseries CSV not found",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")

    # 4) System peak day profile (lower-right)
    ax = axes[1, 1]
    agg_path = os.path.join(output_dir, f"{case_name}_aggregated_ts.csv")
    if os.path.exists(agg_path):
        agg_df = pd.read_csv(agg_path)
        col = agg_df.columns[0]
        series = agg_df[col].values

        peak_hour_idx = int(np.argmax(series))
        day_start = (peak_hour_idx // 24) * 24
        day_profile = series[day_start:day_start + 24]
        start_date = pd.Timestamp(f"{YEAR}-01-01 01:00:00")
        peak_timestamp = start_date + pd.Timedelta(hours=peak_hour_idx)
        peak_date_str = peak_timestamp.strftime("%B %d, %Y")
        peak_hour = peak_hour_idx % 24

        ax.plot(range(24), day_profile, "o-", color="#333333",
                lw=2.5, markersize=6, zorder=3, label="System peak day load")
        if is_tou:
            for h0, h1 in spans:
                ax.axvspan(h0, h1, alpha=0.15, color="red", zorder=1)
        ax.axvline(peak_hour, color="#e74c3c", ls="--", lw=2, alpha=0.7,
                   label=f"System peak at {peak_hour}:00")
        ax.scatter(
            [peak_hour], [day_profile[peak_hour]],
            color="#e74c3c", s=150, zorder=5, marker="*",
            edgecolors="black", linewidths=0.5,
            label="Peak hour",
        )
        ax.set_title(f"System Peak Day Profile\n{peak_date_str}")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel(col)
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Aggregated timeseries CSV not found",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to: {output_path}")


# =============================================================================
# Decarb analysis
# =============================================================================

def analyze_single_decarb(bldg_dir):
    bldg_id = os.path.basename(bldg_dir)
    ts_path = os.path.join(bldg_dir, UPGRADE, "out", "ts.csv")
    if not os.path.exists(ts_path) or os.path.getsize(ts_path) == 0:
        return None
    try:
        wanted = ["buy", "HVACht", "CHPht", "HVACac"]
        header = pd.read_csv(ts_path, nrows=0).columns.tolist()
        usecols = [c for c in wanted if c in header]
        df = pd.read_csv(ts_path, usecols=usecols)
        for c in wanted:
            if c not in df.columns:
                df[c] = 0.0

        buy = df["buy"].fillna(0).values.astype(np.float64)
        result = compute_peaks_from_hourly(buy, len(buy))
        if result is None:
            return None
        result["bldg_id"] = bldg_id

        # Per-metric hourly series for the load-type peak-vs-avg plot.
        series_by_metric = {
            "elec": buy,
            "heat": (df["HVACht"].fillna(0).values.astype(np.float64) +
                     df["CHPht"].fillna(0).values.astype(np.float64)),
            "cool": df["HVACac"].fillna(0).values.astype(np.float64),
        }

        n_hours = len(buy)
        for metric, series in series_by_metric.items():
            if len(series) == 0 or series.sum() == 0:
                result[f"{metric}_peak_idx"] = np.nan
                result[f"{metric}_peak_hour"] = np.nan
                result[f"{metric}_peak_val"] = 0.0
                result[f"{metric}_annual_mean"] = 0.0
                result[f"{metric}_peak_season"] = ""
                continue
            peak_idx = int(np.argmax(series))
            result[f"{metric}_peak_idx"] = peak_idx
            result[f"{metric}_peak_hour"] = peak_idx % 24
            result[f"{metric}_peak_val"] = float(series[peak_idx])
            result[f"{metric}_annual_mean"] = float(series.mean())
            result[f"{metric}_peak_season"] = _classify_peak_season(
                peak_idx, n_hours)

        return result
    except Exception as e:
        print(f"  ERROR building {bldg_id}: {e}")
        return None


def plot_decarb_load_type_peaks(df, case_name, output_dir, weight_map=None):
    """3x2 figure of decarb peak-hour histograms by load type.

    Rows : heating / cooling / total electricity
    Cols : summer-peaking buildings / winter-peaking buildings
           (membership defined per row, by that load's annual peak season,
            with shoulder-month peaks snapped to the nearer season)

    Each cell shows side-by-side grouped bars per hour-of-day:
      - Solid bar  : sum of (weight x annual peak consumption)
      - Hatched bar: sum of (weight x annual mean consumption)
    """
    weight_map = weight_map or {}
    is_tou = _is_tou_case(case_name)
    spans = _peak_period_spans()

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle(
        f"{_title_from_case_name(case_name)}\n"
        f"Peak Hour vs Average Consumption by Load Type",
        fontsize=13, fontweight="bold",
    )

    metrics = ["heat", "cool", "elec"]
    seasons = ["summer", "winter"]
    width = 0.4

    for row_idx, metric in enumerate(metrics):
        peak_hour_col = f"{metric}_peak_hour"
        peak_val_col = f"{metric}_peak_val"
        mean_col = f"{metric}_annual_mean"
        season_col = f"{metric}_peak_season"

        for col_idx, season in enumerate(seasons):
            ax = axes[row_idx, col_idx]

            if peak_hour_col not in df.columns:
                ax.text(0.5, 0.5, f"{peak_hour_col} missing",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            sub = df[[
                "bldg_id", peak_hour_col, peak_val_col,
                mean_col, season_col,
            ]].dropna(subset=[peak_hour_col, season_col])
            sub = sub[sub[season_col] == season]
            sub = sub[sub[peak_hour_col].between(0, 23)]

            if weight_map:
                weights = (sub["bldg_id"].astype(int).map(weight_map)
                           .fillna(1).astype(float))
            else:
                weights = pd.Series(1.0, index=sub.index)

            peak_kwh = sub[peak_val_col].astype(float).values * weights.values
            mean_kwh = sub[mean_col].astype(float).values * weights.values
            hours = sub[peak_hour_col].astype(int).values

            peak_by_hour = np.zeros(24)
            mean_by_hour = np.zeros(24)
            for h, p, m in zip(hours, peak_kwh, mean_kwh):
                peak_by_hour[h] += p
                mean_by_hour[h] += m

            x = np.arange(24)
            base_color = METRIC_COLORS[metric]
            ax.bar(x - width / 2, peak_by_hour, width,
                   color=base_color, edgecolor="white", lw=0.5,
                   label="Peak consumption")
            bars_avg = ax.bar(x + width / 2, mean_by_hour, width,
                              color=base_color, edgecolor="black", lw=0.4,
                              label="Avg consumption", alpha=0.85)
            for b in bars_avg:
                b.set_hatch("//")

            if is_tou:
                for h0, h1 in spans:
                    ax.axvspan(h0 - 0.5, h1 - 0.5, alpha=0.1, color="red")

            season_title = ("Summer-Peaking (Jun-Aug)" if season == "summer"
                            else "Winter-Peaking (Jan-Mar, Oct-Dec)")
            ax.set_title(f"{METRIC_LABELS[metric]} — {season_title}",
                         fontsize=10)
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("kWh (weighted sum)")
            ax.set_xticks(range(24))
            ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
            ax.legend(fontsize=8, loc="upper left")

            n_b = len(sub)
            n_dw = int(weights.sum())
            ax.text(0.98, 0.97,
                    f"n={n_b:,} bldgs\n{n_dw:,} dwellings",
                    transform=ax.transAxes, fontsize=8,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.85,
                              edgecolor="lightgray"))

    plt.tight_layout()
    out_path = os.path.join(output_dir, "decarb_load_type_peak_vs_avg.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run_decarb_analysis(case_name, case_dir, output_dir):
    csv_path = os.path.join(output_dir, "per_building_peak_hours_decarb.csv")

    print(f"\n{'#' * 70}")
    print(f"  DECARB CASE: {case_name}")
    print(f"{'#' * 70}")

    if RERUN_ANALYSIS:
        all_entries = os.listdir(case_dir)
        bldg_dirs = []
        for entry in all_entries:
            full_path = os.path.join(case_dir, entry)
            if os.path.isdir(full_path):
                try:
                    int(entry)
                    bldg_dirs.append(full_path)
                except ValueError:
                    continue

        print(f"Found {len(bldg_dirs)} building directories in {case_dir}")

        results = []
        print(f"Analyzing with {N_WORKERS} workers...")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(analyze_single_decarb, d): d for d in bldg_dirs}
            done = 0
            for future in as_completed(futures):
                done += 1
                res = future.result()
                if res is not None:
                    results.append(res)
                if done % 2000 == 0:
                    print(f"  Processed {done}/{len(bldg_dirs)}...")

        print(f"\nBuildings with ts.csv analyzed: {len(results)}")
        if not results:
            print("No buildings found with ts.csv. Check paths.")
            return

        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"Raw results saved to: {csv_path}")

    else:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Decarb peak CSV not found: {csv_path}\n"
                "Set RERUN_ANALYSIS=True to generate it."
            )
        print(f"Loading existing results from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} buildings.")

    print_peak_summary(df, f"DECARB ({case_name})")

    weight_map = _load_weights(output_dir, case_dir)
    plot_path = os.path.join(output_dir, "per_building_peak_analysis_decarb.png")
    plot_peak_histograms(df, f"Decarb ({case_name})", plot_path, output_dir,
                         case_name, weight_map=weight_map)

    # 3x2 peak-vs-avg by load type (heating / cooling / electricity)
    needed = [f"{m}_peak_hour" for m in ["heat", "cool", "elec"]]
    if all(c in df.columns for c in needed):
        plot_decarb_load_type_peaks(df, case_name, output_dir,
                                    weight_map=weight_map)
    else:
        print("  Skipping decarb_load_type_peak_vs_avg.png — per-metric "
              "columns missing. Re-run with RERUN_ANALYSIS=True.")


# =============================================================================
# ResStock analysis
# =============================================================================

def analyze_single_resstock(args):
    bldg_id, resstock_dir = args
    pq_file = os.path.join(resstock_dir, f"{int(bldg_id)}-0.parquet")
    if not os.path.exists(pq_file):
        return None
    try:
        df = pd.read_parquet(pq_file, engine="fastparquet",
                             columns=["out.electricity.total.energy_consumption..kwh"])
        elec = df["out.electricity.total.energy_consumption..kwh"].fillna(0).values.astype(np.float64)
        n_hours = len(elec) // 4
        elec_h = elec[:n_hours * 4].reshape(-1, 4).sum(axis=1)
        result = compute_peaks_from_hourly(elec_h, n_hours)
        if result is None:
            return None
        result["bldg_id"] = int(bldg_id)
        return result
    except Exception as e:
        print(f"  ERROR ResStock building {bldg_id}: {e}")
        return None


def run_resstock_analysis(case_name, case_dir, output_dir):
    csv_path = os.path.join(output_dir, "per_building_peak_hours_resstock.csv")

    print(f"\n{'#' * 70}")
    print(f"  RESSTOCK BASELINE — {case_name}")
    print(f"{'#' * 70}")

    if RERUN_ANALYSIS:
        mapping_path = os.path.join(output_dir, "building_mapping.csv")
        if not os.path.exists(mapping_path):
            mapping_path = os.path.join(case_dir, "building_mapping.csv")
        if not os.path.exists(mapping_path):
            print(f"ERROR: building_mapping.csv not found.")
            return

        mapping_df = pd.read_csv(mapping_path)
        if "new_bldg_id" not in mapping_df.columns:
            print(f"ERROR: 'new_bldg_id' column not found in {mapping_path}")
            return

        bldg_ids = mapping_df["new_bldg_id"].dropna().astype(int).unique().tolist()
        print(f"Loaded {len(bldg_ids)} building IDs from {mapping_path}")

        args_list = [(bldg_id, RESSTOCK_DIR) for bldg_id in bldg_ids]
        results = []
        print(f"Analyzing with {N_WORKERS} workers...")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(analyze_single_resstock, a): a for a in args_list}
            done = 0
            for future in as_completed(futures):
                done += 1
                res = future.result()
                if res is not None:
                    results.append(res)
                if done % 2000 == 0:
                    print(f"  Processed {done}/{len(bldg_ids)}...")

        print(f"\nResStock buildings analyzed: {len(results)}")
        if not results:
            print("No ResStock buildings found. Check paths.")
            return

        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"Raw results saved to: {csv_path}")

    else:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"ResStock peak CSV not found: {csv_path}\n"
                "Set RERUN_ANALYSIS=True to generate it."
            )
        print(f"Loading existing results from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} buildings.")

    print_peak_summary(df, f"RESSTOCK BASELINE ({case_name})")

    weight_map = _load_weights(output_dir, case_dir)
    plot_path = os.path.join(output_dir, "per_building_peak_analysis_resstock.png")
    plot_peak_histograms(df, f"ResStock Baseline ({case_name})", plot_path,
                         output_dir, case_name, weight_map=weight_map)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    for case_name in CASE_NAMES:
        case_dir = os.path.join(ERCOT_ROOT, "in", "decarb_inputs_alt", case_name)
        output_dir = os.path.join(ERCOT_ROOT, "out", "decarb_results", case_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'*' * 70}")
        print(f"  RUNNING CASE: {case_name}")
        print(f"{'*' * 70}")

        if RUN_DECARB:
            run_decarb_analysis(case_name, case_dir, output_dir)

        if RUN_RESSTOCK:
            run_resstock_analysis(case_name, case_dir, output_dir)

    if not RUN_DECARB and not RUN_RESSTOCK:
        print("Nothing to run. Set RUN_DECARB and/or RUN_RESSTOCK to True.")

    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}\n")