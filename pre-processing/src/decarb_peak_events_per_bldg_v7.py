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
        df = pd.read_csv(ts_path, usecols=["buy"])
        buy = df["buy"].values
        result = compute_peaks_from_hourly(buy, len(buy))
        if result is None:
            return None
        result["bldg_id"] = bldg_id
        return result
    except Exception as e:
        print(f"  ERROR building {bldg_id}: {e}")
        return None


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

    # Aggregate midnight-peakers vs everyone else (decarb only).
    plot_midnight_peaker_profiles(df, case_name, case_dir, output_dir,
                                  weight_map=weight_map)
    plot_peak_day_histogram(df, case_name, output_dir, weight_map=weight_map)
    plot_top_peak_slots(df, case_name, output_dir, weight_map=weight_map)


def _load_single_ts(args):
    """Worker: read a building's hourly 'buy' series and return (bldg_id, array)."""
    bldg_id, case_dir = args
    ts_path = os.path.join(case_dir, str(bldg_id), UPGRADE, "out", "ts.csv")
    if not os.path.exists(ts_path) or os.path.getsize(ts_path) == 0:
        return bldg_id, None
    try:
        df = pd.read_csv(ts_path, usecols=["buy"])
        return bldg_id, df["buy"].values.astype(np.float64)
    except Exception as e:
        print(f"  ERROR ts.csv for building {bldg_id}: {e}")
        return bldg_id, None


def _aggregate_group(bldg_ids, case_dir, weight_map, n_workers=N_WORKERS,
                     label=""):
    """Sum hourly 'buy' across the given building IDs, scaled by their
    dwelling weight. Returns (annual_array, n_buildings, total_weight).
    All series are clipped/padded to 8760 hours."""
    args_list = [(b, case_dir) for b in bldg_ids]
    total = np.zeros(8760, dtype=np.float64)
    n_used = 0
    weight_used = 0
    print(f"  [{label}] Aggregating {len(bldg_ids):,} buildings with "
          f"{n_workers} workers...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_load_single_ts, a): a[0]
                   for a in args_list}
        done = 0
        for future in as_completed(futures):
            bldg_id, series = future.result()
            done += 1
            if series is None:
                continue
            w = weight_map.get(int(bldg_id), 1) if weight_map else 1
            # Pad/clip to 8760
            if len(series) >= 8760:
                contrib = series[:8760]
            else:
                contrib = np.zeros(8760, dtype=np.float64)
                contrib[:len(series)] = series
            total += contrib * w
            n_used += 1
            weight_used += w
            if done % 2000 == 0:
                print(f"    {label}: {done}/{len(bldg_ids)}")
    print(f"  [{label}] Done. {n_used:,} buildings, "
          f"{int(weight_used):,} dwellings.")
    return total, n_used, weight_used


def plot_peak_day_histogram(peaks_df, case_name, output_dir, weight_map=None):
    """Two-panel histogram of the calendar day-of-year on which each
    building's annual peak occurs, comparing midnight-peakers
    (summer or winter peak at hour 00) against the rest.

    Bars are weighted by dwelling count from building_mapping.csv when
    available. Days run from 1 to 365.
    """
    weight_map = weight_map or {}
    mid_mask = ((peaks_df["summer_peak_hour"] == 0) |
                (peaks_df["winter_peak_hour"] == 0))
    mid = peaks_df.loc[mid_mask].copy()
    rest = peaks_df.loc[~mid_mask].copy()

    # Day of year (1-indexed) from the global full_peak_idx
    mid["peak_doy"] = (mid["full_peak_idx"].astype(int) // 24) + 1
    rest["peak_doy"] = (rest["full_peak_idx"].astype(int) // 24) + 1

    def weighted_doy_hist(sub):
        bins = np.zeros(366, dtype=np.float64)  # index 1..365
        if weight_map:
            w = sub["bldg_id"].astype(int).map(weight_map).fillna(1).astype(float)
        else:
            w = pd.Series(np.ones(len(sub)), index=sub.index)
        for doy, weight in zip(sub["peak_doy"].astype(int).values, w.values):
            if 1 <= doy <= 365:
                bins[doy] += weight
        return bins[1:]  # length-365

    mid_h = weighted_doy_hist(mid)
    rest_h = weighted_doy_hist(rest)

    y_label = "Number of Dwellings" if weight_map else "Number of Buildings"
    n_mid_b = len(mid)
    n_rest_b = len(rest)
    n_mid_dw = int(mid_h.sum())
    n_rest_dw = int(rest_h.sum())

    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=False)
    fig.suptitle(f"{_title_from_case_name(case_name)}\n"
                 f"Annual Peak Day Distribution",
                 fontsize=13, fontweight="bold")

    # Month boundaries (cumulative day-of-year for non-leap year)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    days = np.arange(1, 366)

    def _decorate(ax, title):
        ax.set_title(title)
        ax.set_xlabel("Day of Year")
        ax.set_ylabel(y_label)
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels, fontsize=9)
        for m in month_starts[1:]:
            ax.axvline(m - 0.5, color="gray", lw=0.4, alpha=0.4)
        ax.set_xlim(0.5, 365.5)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].bar(days, mid_h, width=1.0, color="#1f4e79",
                edgecolor="none", align="center")
    _decorate(axes[0],
              f"Midnight-peakers  "
              f"(n={n_mid_b:,} buildings | {n_mid_dw:,} dwellings)")

    axes[1].bar(days, rest_h, width=1.0, color="#c0504d",
                edgecolor="none", align="center")
    _decorate(axes[1],
              f"Rest  "
              f"(n={n_rest_b:,} buildings | {n_rest_dw:,} dwellings)")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "peak_day_of_year_histogram.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_top_peak_slots(peaks_df, case_name, output_dir,
                        weight_map=None, top_n=10):
    """For midnight-peakers vs rest, find the top-N hour-slots of the year
    (specific calendar timestamps) on which buildings most commonly hit
    their annual peak. Renders as side-by-side horizontal bar charts,
    weighted by dwelling count when available.
    """
    weight_map = weight_map or {}
    mid_mask = ((peaks_df["summer_peak_hour"] == 0) |
                (peaks_df["winter_peak_hour"] == 0))
    mid = peaks_df.loc[mid_mask].copy()
    rest = peaks_df.loc[~mid_mask].copy()

    start = pd.Timestamp(f"{YEAR}-01-01 01:00:00")

    def top_slots(sub):
        if weight_map:
            w = sub["bldg_id"].astype(int).map(weight_map).fillna(1).astype(float)
        else:
            w = pd.Series(np.ones(len(sub)), index=sub.index)
        df_local = pd.DataFrame({
            "idx": sub["full_peak_idx"].astype(int).values,
            "w": w.values,
        })
        slot_w = (df_local.groupby("idx")["w"].sum()
                  .sort_values(ascending=False).head(top_n))
        labels, values = [], []
        for idx, weight in slot_w.items():
            ts = start + pd.Timedelta(hours=int(idx))
            labels.append(ts.strftime("%b %d, %H:00"))
            values.append(int(weight))
        return labels, values

    mid_labels, mid_values = top_slots(mid)
    rest_labels, rest_values = top_slots(rest)

    x_label = "Number of Dwellings" if weight_map else "Number of Buildings"

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"{_title_from_case_name(case_name)}\n"
                 f"Top {top_n} Hour-Slots for Annual Peak",
                 fontsize=13, fontweight="bold")

    def _bar(ax, labels, values, color, title):
        if not labels:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            return
        # Reverse so the most-frequent slot ends up at the top of the bar chart
        labels = labels[::-1]
        values = values[::-1]
        y = np.arange(len(labels))
        ax.barh(y, values, color=color, edgecolor="white", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
        # value annotations
        xmax = max(values) if values else 1
        for yi, v in zip(y, values):
            ax.text(v + xmax * 0.01, yi, f"{v:,}",
                    va="center", fontsize=8)
        ax.set_xlim(0, xmax * 1.15)

    n_mid_dw = (int(pd.Series(mid["bldg_id"].astype(int).map(weight_map))
                    .fillna(1).sum()) if weight_map else len(mid))
    n_rest_dw = (int(pd.Series(rest["bldg_id"].astype(int).map(weight_map))
                     .fillna(1).sum()) if weight_map else len(rest))

    _bar(axes[0], mid_labels, mid_values, "#1f4e79",
         f"Midnight-peakers  "
         f"(n={len(mid):,} buildings | {n_mid_dw:,} dwellings)")
    _bar(axes[1], rest_labels, rest_values, "#c0504d",
         f"Rest  "
         f"(n={len(rest):,} buildings | {n_rest_dw:,} dwellings)")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "top_peak_slots.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_midnight_peaker_profiles(peaks_df, case_name, case_dir, output_dir,
                                  weight_map=None):
    """Build two figures comparing buildings that peak at midnight
    (summer_peak_hour == 0 OR winter_peak_hour == 0) against the rest.

    Each figure has 3 panels:
      1. Average daily profile across all 365 days (24h)
      2. System peak day profile (24h, day identified from full aggregate)
      3. Annual hourly total consumption (8760h)

    Plot A — only the midnight-peakers.
    Plot B — overlay of midnight-peakers and everyone else.
    """
    weight_map = weight_map or {}
    spans = _peak_period_spans()
    is_tou = _is_tou_case(case_name)

    mid_mask = ((peaks_df["summer_peak_hour"] == 0) |
                (peaks_df["winter_peak_hour"] == 0))
    mid_ids = peaks_df.loc[mid_mask, "bldg_id"].astype(int).tolist()
    rest_ids = peaks_df.loc[~mid_mask, "bldg_id"].astype(int).tolist()
    print(f"\n  Midnight-peakers (summer or winter): {len(mid_ids):,}")
    print(f"  Rest: {len(rest_ids):,}")
    if not mid_ids:
        print("  No midnight-peakers found — skipping midnight-peaker plots.")
        return

    mid_total, n_mid, w_mid = _aggregate_group(
        mid_ids, case_dir, weight_map, label="midnight")
    rest_total, n_rest, w_rest = _aggregate_group(
        rest_ids, case_dir, weight_map, label="rest")

    # Use GW for plotting (buy is in kW). Convert.
    mid_gw = mid_total / 1e6
    rest_gw = rest_total / 1e6
    full_gw = mid_gw + rest_gw

    def daily_avg(series_8760):
        return series_8760.reshape(365, 24).mean(axis=0)

    mid_avg_day = daily_avg(mid_gw)
    rest_avg_day = daily_avg(rest_gw)
    full_avg_day = daily_avg(full_gw)

    # System peak day is determined by the full (combined) aggregate so the
    # day chosen is the same in both plots.
    sys_peak_idx = int(np.argmax(full_gw))
    day_start = (sys_peak_idx // 24) * 24
    mid_peak_day = mid_gw[day_start:day_start + 24]
    rest_peak_day = rest_gw[day_start:day_start + 24]
    full_peak_day = full_gw[day_start:day_start + 24]
    start_date = pd.Timestamp(f"{YEAR}-01-01 01:00:00")
    peak_date_str = (start_date + pd.Timedelta(hours=sys_peak_idx)
                     ).strftime("%B %d, %Y")

    title_base = _title_from_case_name(case_name)

    # ── Plot A: midnight-peakers only ───────────────────────────────
    figA, axA = plt.subplots(1, 3, figsize=(22, 6))
    figA.suptitle(f"{title_base}\nMidnight-Peakers Only "
                  f"(n={n_mid:,} buildings, {int(w_mid):,} dwellings)",
                  fontsize=13, fontweight="bold")

    # Panel 1: avg daily
    ax = axA[0]
    ax.plot(range(24), mid_avg_day, "o-", color="#1f4e79", lw=2,
            markersize=5, label="Midnight-peakers")
    if is_tou:
        for h0, h1 in spans:
            ax.axvspan(h0, h1, alpha=0.15, color="red", zorder=1)
    pk_h = int(np.argmax(mid_avg_day))
    ax.axvline(pk_h, color="#e74c3c", ls="--", alpha=0.7,
               label=f"Avg peak at {pk_h}:00")
    ax.scatter([pk_h], [mid_avg_day[pk_h]], color="#e74c3c", s=120,
               marker="*", edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_title("Average Daily Profile (all 365 days)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("GW")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: system peak day
    ax = axA[1]
    ax.plot(range(24), mid_peak_day, "o-", color="#1f4e79", lw=2.5,
            markersize=6, label="Midnight-peakers")
    if is_tou:
        for h0, h1 in spans:
            ax.axvspan(h0, h1, alpha=0.15, color="red", zorder=1)
    sys_peak_hour = sys_peak_idx % 24
    ax.axvline(sys_peak_hour, color="#e74c3c", ls="--", alpha=0.7,
               label=f"System peak at {sys_peak_hour}:00")
    ax.set_title(f"System Peak Day Profile\n{peak_date_str}")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("GW")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: annual hourly consumption
    ax = axA[2]
    ax.plot(np.arange(8760) / 24.0, mid_gw, color="#1f4e79", lw=0.4,
            label="Midnight-peakers")
    ax.set_title(f"Annual Hourly Load — total = "
                 f"{mid_gw.sum() / 1000:,.1f} TWh")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("GW")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    pathA = os.path.join(output_dir, "midnight_peakers_profiles.png")
    figA.savefig(pathA, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figA)
    print(f"  Saved: {pathA}")

    # ── Plot B: midnight-peakers vs rest ─────────────────────────────
    figB, axB = plt.subplots(1, 3, figsize=(22, 6))
    figB.suptitle(f"{title_base}\nMidnight-Peakers vs Rest",
                  fontsize=13, fontweight="bold")

    mid_label = f"Midnight-peakers (n={n_mid:,}, {int(w_mid):,} dw)"
    rest_label = f"Rest (n={n_rest:,}, {int(w_rest):,} dw)"

    # Panel 1: avg daily
    ax = axB[0]
    ax.plot(range(24), mid_avg_day, "o-", color="#1f4e79", lw=2,
            markersize=5, label=mid_label)
    ax.plot(range(24), rest_avg_day, "s-", color="#c0504d", lw=2,
            markersize=5, label=rest_label)
    if is_tou:
        for h0, h1 in spans:
            ax.axvspan(h0, h1, alpha=0.15, color="red", zorder=1)
    ax.set_title("Average Daily Profile (all 365 days)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("GW")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: system peak day
    ax = axB[1]
    ax.plot(range(24), mid_peak_day, "o-", color="#1f4e79", lw=2.5,
            markersize=6, label=mid_label)
    ax.plot(range(24), rest_peak_day, "s-", color="#c0504d", lw=2.5,
            markersize=6, label=rest_label)
    if is_tou:
        for h0, h1 in spans:
            ax.axvspan(h0, h1, alpha=0.15, color="red", zorder=1)
    ax.axvline(sys_peak_hour, color="#333333", ls="--", alpha=0.5,
               label=f"System peak at {sys_peak_hour}:00")
    ax.set_title(f"System Peak Day Profile\n{peak_date_str}")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("GW")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: annual hourly consumption — stack midnight on bottom for
    # readability
    ax = axB[2]
    days = np.arange(8760) / 24.0
    ax.plot(days, mid_gw, color="#1f4e79", lw=0.4, alpha=0.9,
            label=f"{mid_label}: {mid_gw.sum() / 1000:,.1f} TWh")
    ax.plot(days, rest_gw, color="#c0504d", lw=0.4, alpha=0.7,
            label=f"{rest_label}: {rest_gw.sum() / 1000:,.1f} TWh")
    ax.set_title("Annual Hourly Load")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("GW")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    pathB = os.path.join(output_dir, "midnight_peakers_vs_rest.png")
    figB.savefig(pathB, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figB)
    print(f"  Saved: {pathB}")


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