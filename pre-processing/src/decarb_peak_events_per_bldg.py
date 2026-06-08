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
from concurrent.futures import ProcessPoolExecutor, as_completed


# ── Configuration ──────────────────────────────────────────
CASE_NAMES = [
    "elec_100_flat_results",
    # "elec_0_tou",
    # "elec_0_tou_2_period",
    # Add more cases here, e.g.:
    # "elec_1_flat",
    # "gas_0_flat",
]

ERCOT_ROOT = r"D:\shared\ercot_project"
UPGRADE = "update_0"
YEAR = 2018
N_WORKERS = 20

RUN_DECARB = True
RUN_RESSTOCK = True

RESSTOCK_DIR = os.path.join(ERCOT_ROOT, "out", "consumption_files")
# ───────────────────────────────────────────────────────────

# Summer = Jun/Jul/Aug = hours 3624..5856 (for 8760-hour year starting hour 0 = Jan 1 00:00)
SUMMER_START = 3624
SUMMER_END = 5856  # exclusive


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


def compute_peaks_from_hourly(buy, n_hours):
    """Given an hourly 'buy' array, return peak info dict."""
    # Full year peak
    full_peak_idx = int(np.argmax(buy))
    full_peak_val = buy[full_peak_idx]
    full_peak_hour = full_peak_idx % 24

    # Summer peak
    summer_end = min(SUMMER_END, n_hours)
    summer_buy = buy[SUMMER_START:summer_end]
    if len(summer_buy) == 0:
        return None

    summer_peak_local_idx = int(np.argmax(summer_buy))
    summer_peak_global_idx = SUMMER_START + summer_peak_local_idx
    summer_peak_val = summer_buy[summer_peak_local_idx]
    summer_peak_hour = summer_peak_global_idx % 24

    # Winter peak (Jan-Mar + Oct-Dec)
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


def print_peak_summary(df, label):
    """Print hourly and time-period summaries for a results dataframe."""
    print(f"\n{'=' * 60}")
    print(f"  {label} — SUMMER PEAK HOUR DISTRIBUTION")
    print(f"{'=' * 60}")
    summer_counts = df["summer_peak_hour"].value_counts().sort_index()
    for hour in range(24):
        c = summer_counts.get(hour, 0)
        pct = 100 * c / len(df)
        bar = "█" * int(pct)
        print(f"  {hour:02d}:00  {c:>5d}  ({pct:5.1f}%)  {bar}")

    print(f"\n{'=' * 60}")
    print(f"  {label} — WINTER PEAK HOUR DISTRIBUTION")
    print(f"{'=' * 60}")
    winter_counts = df["winter_peak_hour"].value_counts().sort_index()
    for hour in range(24):
        c = winter_counts.get(hour, 0)
        pct = 100 * c / len(df)
        bar = "█" * int(pct)
        print(f"  {hour:02d}:00  {c:>5d}  ({pct:5.1f}%)  {bar}")

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


def plot_peak_histograms(df, label, output_path):
    """Generate the 2x2 peak analysis plot."""
    summer_counts = df["summer_peak_hour"].value_counts().sort_index()
    winter_counts = df["winter_peak_hour"].value_counts().sort_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Per-Building Peak Hour Analysis — {label}\n"
        f"({len(df)} buildings)",
        fontsize=14, fontweight="bold",
    )

    # 1) Summer peak hour histogram
    ax = axes[0, 0]
    counts = [summer_counts.get(h, 0) for h in range(24)]
    colors = ["#e74c3c" if 16 <= h <= 19 else "#4a90d9" for h in range(24)]
    ax.bar(range(24), counts, color=colors, edgecolor="white", lw=0.5)
    ax.set_title("Summer Peak Hour (Jun-Aug)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Buildings")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.axvspan(16, 19, alpha=0.1, color="red", label="Expected peak window")
    ax.legend(fontsize=8)

    # 2) Winter peak hour histogram
    ax = axes[0, 1]
    w_counts = [winter_counts.get(h, 0) for h in range(24)]
    colors_w = ["#e74c3c" if 6 <= h <= 9 else "#4a90d9" for h in range(24)]
    ax.bar(range(24), w_counts, color=colors_w, edgecolor="white", lw=0.5)
    ax.set_title("Winter Peak Hour (Jan-Mar, Oct-Dec)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Buildings")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.axvspan(6, 9, alpha=0.1, color="red", label="Expected peak window")
    ax.legend(fontsize=8)

    # 3) Summer peak by time period
    ax = axes[1, 0]
    period_names = list(TIME_PERIODS.keys())
    period_counts_summer = []
    for name, (h0, h1) in TIME_PERIODS.items():
        c = ((df["summer_peak_hour"] >= h0) & (df["summer_peak_hour"] <= h1)).sum()
        period_counts_summer.append(c)
    pcolors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(period_names)))
    ax.barh(range(len(period_names)), period_counts_summer, color=pcolors, edgecolor="white")
    ax.set_yticks(range(len(period_names)))
    ax.set_yticklabels(period_names, fontsize=9)
    ax.set_title("Summer Peaks by Time Period")
    ax.set_xlabel("Number of Buildings")
    for i, c in enumerate(period_counts_summer):
        ax.text(c + 0.5, i, str(c), va="center", fontsize=9)

    # 4) Sample single building daily profile (hottest day)
    ax = axes[1, 1]
    top_bldg = df.loc[df["summer_peak_val"].idxmax()]
    peak_idx = int(top_bldg["summer_peak_idx"])
    bldg_id = top_bldg["bldg_id"]

    day_buy = top_bldg.get("_peak_day_profile", None)
    if day_buy is not None and len(day_buy) == 24:
        ax.plot(range(24), day_buy, "o-", color="#e74c3c", lw=2, markersize=4)
        ax.axvline(peak_idx % 24, color="gray", ls="--", alpha=0.7,
                   label=f"Peak at {peak_idx % 24}:00")
        ax.set_title(f"Peak Day Profile — Building {bldg_id}\n(highest summer peak)")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Electricity (kW)")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Daily profile not available", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to: {output_path}")


# =============================================================================
# Decarb analysis
# =============================================================================

def analyze_single_decarb(bldg_dir):
    """Load one building's ts.csv and find peak hours."""
    bldg_id = os.path.basename(bldg_dir)
    ts_path = os.path.join(bldg_dir, UPGRADE, "out", "ts.csv")

    if not os.path.exists(ts_path):
        return None

    try:
        df = pd.read_csv(ts_path, usecols=["buy"])
        buy = df["buy"].values
        result = compute_peaks_from_hourly(buy, len(buy))
        if result is None:
            return None
        result["bldg_id"] = bldg_id

        # Grab the peak day profile for the top-building plot
        peak_idx = result["summer_peak_idx"]
        day_start = (peak_idx // 24) * 24
        result["_peak_day_profile"] = buy[day_start:day_start + 24].tolist()

        return result
    except Exception as e:
        print(f"  ERROR building {bldg_id}: {e}")
        return None


def run_decarb_analysis(case_name, case_dir, output_dir):
    """Scan all buildings in case_dir for ts.csv and analyze peaks."""
    print(f"\n{'#' * 70}")
    print(f"  DECARB CASE: {case_name}")
    print(f"{'#' * 70}")

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

    csv_path = os.path.join(output_dir, "per_building_peak_hours_decarb.csv")
    df.drop(columns=["_peak_day_profile"], errors="ignore").to_csv(csv_path, index=False)
    print(f"Raw results saved to: {csv_path}")

    print_peak_summary(df, f"DECARB ({case_name})")
    plot_path = os.path.join(output_dir, "per_building_peak_analysis_decarb.png")
    plot_peak_histograms(df, f"Decarb ({case_name})", plot_path)


# =============================================================================
# ResStock analysis
# =============================================================================

def analyze_single_resstock(args):
    """Load one ResStock parquet, downsample 15-min -> hourly, find peak hours."""
    bldg_id, resstock_dir = args
    pq_file = os.path.join(resstock_dir, f"{int(bldg_id)}-0.parquet")

    if not os.path.exists(pq_file):
        return None

    try:
        df = pd.read_parquet(pq_file, engine="fastparquet",
                             columns=["out.electricity.total.energy_consumption..kwh"])
        elec = df["out.electricity.total.energy_consumption..kwh"].fillna(0).values.astype(np.float64)

        # Downsample 15-min -> hourly by summing every 4 rows
        n_quarter = len(elec)
        n_hours = n_quarter // 4
        elec_h = elec[:n_hours * 4].reshape(-1, 4).sum(axis=1)

        result = compute_peaks_from_hourly(elec_h, n_hours)
        if result is None:
            return None
        result["bldg_id"] = int(bldg_id)

        # Grab the peak day profile
        peak_idx = result["summer_peak_idx"]
        day_start = (peak_idx // 24) * 24
        result["_peak_day_profile"] = elec_h[day_start:day_start + 24].tolist()

        return result
    except Exception as e:
        print(f"  ERROR ResStock building {bldg_id}: {e}")
        return None


def run_resstock_analysis(case_name, case_dir, output_dir):
    """Scan ResStock consumption_files for buildings listed in building_mapping.csv."""
    print(f"\n{'#' * 70}")
    print(f"  RESSTOCK BASELINE — {case_name}")
    print(f"{'#' * 70}")

    # Load building IDs from building_mapping.csv
    mapping_path = os.path.join(case_dir, "building_mapping.csv")
    if not os.path.exists(mapping_path):
        print(f"ERROR: building_mapping.csv not found at {mapping_path}")
        return

    mapping_df = pd.read_csv(mapping_path)
    if "new_bldg_id" not in mapping_df.columns:
        print(f"ERROR: 'new_bldg_id' column not found in {mapping_path}")
        print(f"  Available columns: {list(mapping_df.columns)}")
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

    csv_path = os.path.join(output_dir, "per_building_peak_hours_resstock.csv")
    df.drop(columns=["_peak_day_profile"], errors="ignore").to_csv(csv_path, index=False)
    print(f"Raw results saved to: {csv_path}")

    print_peak_summary(df, f"RESSTOCK BASELINE ({case_name})")
    plot_path = os.path.join(output_dir, "per_building_peak_analysis_resstock.png")
    plot_peak_histograms(df, f"ResStock Baseline ({case_name})", plot_path)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    for case_name in CASE_NAMES:
        case_dir = os.path.join(ERCOT_ROOT, "in", "decarb_inputs_alt", case_name)
        output_dir = os.path.join(ERCOT_ROOT, "out", "decarb_results_alt", case_name)
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