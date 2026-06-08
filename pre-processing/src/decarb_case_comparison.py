"""
ERCOT Peak Shifting & Cost Comparison
=======================================
Compares TOU cases against a flat-rate baseline to quantify:
  1. Energy shifted from peak to off-peak periods
  2. Peak demand reduction (system-level)
  3. Total electricity expenditure changes

Baseline case: elec_0_flat_results
Peak definitions per case:
  - elec_0_flat_results:          no peak window (flat rate)
  - elec_0_tou_results:           peak = 4pm-7pm (hours 16-19) + 6am-9am (hours 6-9)
  - elec_0_tou_2_period_results:  peak = 4pm-7pm (hours 16-19) + 6am-9am (hours 6-9)

All cases read from decarb_results_alt / decarb_inputs_alt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Configuration ──────────────────────────────────────────
ERCOT_ROOT = r"D:\shared\ercot_project"
RESULTS_DIR = os.path.join(ERCOT_ROOT, "out", "decarb_results_alt")
INPUTS_DIR  = os.path.join(ERCOT_ROOT, "in",  "decarb_inputs_alt")
YEAR = 2018

BASELINE_CASE = "elec_0_flat_results"

CASE_NAMES = [
    "elec_0_flat_results",
    "elec_100_flat_results",
    "elec_0_tou_results",
]

# Peak hour definitions per case (hours in 0-23 range, inclusive)
# Each entry is a list of (start_hour, end_hour) tuples
PEAK_HOURS = {
    "elec_0_flat_results":   [(18, 22), (6, 10)],  # 6pm-10pm + 6am-10am
    "elec_100_flat_results": [(18, 22), (6, 10)],  # 6pm-10pm + 6am-10am
    "elec_0_tou_results":    [(18, 22), (6, 10)],  # 6pm-10pm + 6am-10am
}

# Reference building ID for loading tm.csv (all buildings have identical rates)
REF_BLDG_ID = 100

OUTPUT_DIR = os.path.join(RESULTS_DIR, "peak_comparison")
# ───────────────────────────────────────────────────────────


def is_peak_hour(hour, case_name):
    """Check if a given hour falls within peak windows for the case."""
    windows = PEAK_HOURS.get(case_name, [])
    for h0, h1 in windows:
        if h0 <= hour <= h1:
            return True
    return False


def load_aggregated_ts(case_name):
    """Load the aggregated time-series CSV for a case."""
    filepath = os.path.join(RESULTS_DIR, case_name, f"{case_name}_aggregated_ts.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Not found: {filepath}")

    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(filepath, sep=sep, engine="python")
            if len(df.columns) > 1:
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not parse {filepath}")

    # Build hourly datetime index
    n = len(df)
    start = pd.Timestamp(f"{YEAR}-01-01 01:00:00")
    df.index = pd.date_range(start, periods=n, freq="h")
    print(f"  Loaded aggregated_ts: {n} rows — {case_name}")
    return df


def load_rate(case_name):
    """Load the hourly electricity rate ($/kWh) from tm.csv."""
    tm_path = os.path.join(INPUTS_DIR, case_name, str(REF_BLDG_ID), "update_0", "in", "tm.csv")
    if not os.path.exists(tm_path):
        raise FileNotFoundError(f"Rate file not found: {tm_path}")

    tm_df = pd.read_csv(tm_path)
    if "pQcostBuy" not in tm_df.columns:
        raise ValueError(f"'pQcostBuy' column not found in {tm_path}. Columns: {list(tm_df.columns)}")

    rate = tm_df["pQcostBuy"].values  # $/kWh
    print(f"  Loaded rate: {len(rate)} hours — {case_name} "
          f"(min=${rate.min():.4f}, max=${rate.max():.4f}, mean=${rate.mean():.4f} $/kWh)")
    return rate


def analyze_case(case_name, grid_gw, rate_per_kwh):
    """
    Compute peak/off-peak energy, peak demand, and total cost for one case.

    Parameters
    ----------
    case_name : str
    grid_gw : np.ndarray — hourly grid purchases in GW (8760,)
    rate_per_kwh : np.ndarray — hourly rate in $/kWh (8760,)

    Returns
    -------
    dict with all metrics
    """
    n = len(grid_gw)
    hours = np.array([pd.Timestamp(f"{YEAR}-01-01 01:00:00") + pd.Timedelta(hours=i)
                       for i in range(n)])
    hour_of_day = np.array([t.hour for t in hours])

    # Peak mask
    peak_mask    = np.array([is_peak_hour(h, case_name) for h in hour_of_day])
    offpeak_mask = ~peak_mask

    # Energy (GWh) — grid_gw is in GW, hourly data, so each value = GWh for that hour
    total_energy_gwh    = grid_gw.sum()
    peak_energy_gwh     = grid_gw[peak_mask].sum()
    offpeak_energy_gwh  = grid_gw[offpeak_mask].sum()
    peak_hours_count    = peak_mask.sum()
    offpeak_hours_count = offpeak_mask.sum()

    # Peak demand (GW)
    system_peak_gw        = grid_gw.max()
    system_peak_idx       = int(np.argmax(grid_gw))
    system_peak_hour      = hour_of_day[system_peak_idx]
    system_peak_timestamp = hours[system_peak_idx]

    # Top 10 peak demand hours
    top10_idx = np.argsort(grid_gw)[-10:][::-1]
    top10 = [(hours[i], grid_gw[i], hour_of_day[i]) for i in top10_idx]

    # Peak-window peak demand (max within defined peak hours)
    if peak_mask.any():
        peak_window_max_gw = grid_gw[peak_mask].max()
    else:
        peak_window_max_gw = np.nan

    # Cost — grid_gw is in GW, rate is $/kWh
    # GW * 1e6 = kW, so hourly cost = GW * 1e6 * $/kWh = $ per hour
    hourly_cost  = grid_gw * 1e6 * rate_per_kwh  # $ per hour
    total_cost   = hourly_cost.sum()
    peak_cost    = hourly_cost[peak_mask].sum()
    offpeak_cost = hourly_cost[offpeak_mask].sum()

    # Average rates realized
    avg_rate_overall = total_cost / (total_energy_gwh * 1e6)    if total_energy_gwh > 0    else 0
    avg_rate_peak    = peak_cost / (peak_energy_gwh * 1e6)      if peak_energy_gwh > 0     else 0
    avg_rate_offpeak = offpeak_cost / (offpeak_energy_gwh * 1e6) if offpeak_energy_gwh > 0 else 0

    return {
        "case_name": case_name,
        # Energy
        "total_energy_gwh":    total_energy_gwh,
        "peak_energy_gwh":     peak_energy_gwh,
        "offpeak_energy_gwh":  offpeak_energy_gwh,
        "peak_hours_count":    peak_hours_count,
        "offpeak_hours_count": offpeak_hours_count,
        # Peak demand
        "system_peak_gw":        system_peak_gw,
        "system_peak_hour":      system_peak_hour,
        "system_peak_timestamp": system_peak_timestamp,
        "peak_window_max_gw":    peak_window_max_gw,
        "top10_peaks":           top10,
        # Cost
        "total_cost_million":   total_cost / 1e6,
        "peak_cost_million":    peak_cost / 1e6,
        "offpeak_cost_million": offpeak_cost / 1e6,
        "avg_rate_overall":     avg_rate_overall,
        "avg_rate_peak":        avg_rate_peak,
        "avg_rate_offpeak":     avg_rate_offpeak,
        # Raw for plotting
        "_grid_gw":     grid_gw,
        "_hourly_cost": hourly_cost,
        "_hour_of_day": hour_of_day,
        "_peak_mask":   peak_mask,
        "_rate":        rate_per_kwh,
    }


def compute_deltas(baseline, case):
    """Compute changes from baseline."""
    return {
        "case_name": case["case_name"],
        # Energy shifts
        "delta_total_energy_gwh":    case["total_energy_gwh"]   - baseline["total_energy_gwh"],
        "delta_peak_energy_gwh":     case["peak_energy_gwh"]    - baseline["peak_energy_gwh"],
        "delta_offpeak_energy_gwh":  case["offpeak_energy_gwh"] - baseline["offpeak_energy_gwh"],
        "pct_peak_energy_change":    100 * (case["peak_energy_gwh"] - baseline["peak_energy_gwh"])
                                     / baseline["peak_energy_gwh"] if baseline["peak_energy_gwh"] > 0 else np.nan,
        "pct_offpeak_energy_change": 100 * (case["offpeak_energy_gwh"] - baseline["offpeak_energy_gwh"])
                                     / baseline["offpeak_energy_gwh"] if baseline["offpeak_energy_gwh"] > 0 else np.nan,
        # Peak demand
        "delta_system_peak_gw":   case["system_peak_gw"] - baseline["system_peak_gw"],
        "pct_system_peak_change": 100 * (case["system_peak_gw"] - baseline["system_peak_gw"])
                                  / baseline["system_peak_gw"],
        "delta_peak_window_max_gw": case["peak_window_max_gw"] - baseline["peak_window_max_gw"]
                                    if not np.isnan(case["peak_window_max_gw"])
                                       and not np.isnan(baseline["peak_window_max_gw"]) else np.nan,
        # Cost
        "delta_total_cost_million": case["total_cost_million"] - baseline["total_cost_million"],
        "pct_total_cost_change":    100 * (case["total_cost_million"] - baseline["total_cost_million"])
                                    / baseline["total_cost_million"] if baseline["total_cost_million"] > 0 else 0,
    }


def print_summary(all_results, all_deltas, baseline):
    """Print formatted comparison summary."""
    bl = baseline

    print(f"\n{'=' * 80}")
    print(f"  PEAK SHIFTING & COST COMPARISON — Baseline: {bl['case_name']}")
    print(f"{'=' * 80}")

    # Baseline summary
    print(f"\n{'─' * 60}")
    print(f"  BASELINE: {bl['case_name']}")
    print(f"{'─' * 60}")
    print(f"  Total energy:        {bl['total_energy_gwh']:>12.2f} GWh")
    print(f"  System peak demand:  {bl['system_peak_gw']:>12.4f} GW")
    print(f"  System peak at:      {bl['system_peak_timestamp']} (hour {bl['system_peak_hour']})")
    print(f"  Total cost:          ${bl['total_cost_million']:>11.2f} M")

    # Per-case comparison
    for res in all_results:
        if res["case_name"] == bl["case_name"]:
            continue

        delta    = next(d for d in all_deltas if d["case_name"] == res["case_name"])
        peak_def = PEAK_HOURS.get(res["case_name"], [])
        peak_str = ", ".join(f"{h0}:00-{h1+1}:00" for h0, h1 in peak_def) if peak_def else "None"

        print(f"\n{'─' * 60}")
        print(f"  CASE: {res['case_name']}")
        print(f"  Peak windows: {peak_str}")
        print(f"{'─' * 60}")

        print(f"\n  Energy:")
        print(f"    Total:             {res['total_energy_gwh']:>12.2f} GWh  "
              f"({delta['delta_total_energy_gwh']:+.2f} GWh)")
        if not np.isnan(delta['pct_peak_energy_change']):
            print(f"    Peak-window:       {res['peak_energy_gwh']:>12.2f} GWh  "
                  f"({delta['delta_peak_energy_gwh']:+.2f} GWh, "
                  f"{delta['pct_peak_energy_change']:+.1f}%)")
        else:
            print(f"    Peak-window:       {res['peak_energy_gwh']:>12.2f} GWh")
        if not np.isnan(delta['pct_offpeak_energy_change']):
            print(f"    Off-peak:          {res['offpeak_energy_gwh']:>12.2f} GWh  "
                  f"({delta['delta_offpeak_energy_gwh']:+.2f} GWh, "
                  f"{delta['pct_offpeak_energy_change']:+.1f}%)")
        else:
            print(f"    Off-peak:          {res['offpeak_energy_gwh']:>12.2f} GWh")

        print(f"\n  Peak Demand:")
        print(f"    System peak:       {res['system_peak_gw']:>12.4f} GW   "
              f"({delta['delta_system_peak_gw']:+.4f} GW, {delta['pct_system_peak_change']:+.1f}%)")
        print(f"    Peak at:           {res['system_peak_timestamp']} (hour {res['system_peak_hour']})")

        print(f"\n  Cost:")
        print(f"    Total expenditure: ${res['total_cost_million']:>11.2f} M   "
              f"({delta['delta_total_cost_million']:+.2f} M, {delta['pct_total_cost_change']:+.1f}%)")
        print(f"    Avg rate overall:  ${res['avg_rate_overall']:.4f}/kWh")
        if res["avg_rate_peak"] > 0:
            print(f"    Avg rate peak:     ${res['avg_rate_peak']:.4f}/kWh")
            print(f"    Avg rate off-peak: ${res['avg_rate_offpeak']:.4f}/kWh")

    # Top 10 peaks comparison
    print(f"\n{'─' * 60}")
    print(f"  TOP 10 SYSTEM PEAK HOURS — by case")
    print(f"{'─' * 60}")
    for res in all_results:
        print(f"\n  {res['case_name']}:")
        for rank, (ts, val, hr) in enumerate(res["top10_peaks"], 1):
            print(f"    {rank:>2d}. {ts}  {val:.4f} GW  (hour {hr})")


def plot_comparison(all_results, baseline, output_dir):
    """Generate comparison plots."""
    n_cases    = len(all_results)
    case_names = [r["case_name"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Peak Shifting & Cost Comparison\nBaseline: {baseline['case_name']}",
                 fontsize=15, fontweight="bold")
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # ── 1) Average hourly load profiles ──
    ax     = axes[0, 0]
    colors = plt.cm.Set1(np.linspace(0, 0.8, n_cases))
    for i, res in enumerate(all_results):
        grid        = res["_grid_gw"]
        hod         = res["_hour_of_day"]
        avg_by_hour = np.array([grid[hod == h].mean() for h in range(24)])
        style = "-"  if res["case_name"] == baseline["case_name"] else "--"
        lw    = 2.5  if res["case_name"] == baseline["case_name"] else 1.8
        ax.plot(range(24), avg_by_hour, style, color=colors[i], lw=lw, label=res["case_name"])

    for res in all_results:
        if res["case_name"] == baseline["case_name"]:
            continue
        for h0, h1 in PEAK_HOURS.get(res["case_name"], []):
            ax.axvspan(h0, h1 + 1, alpha=0.07, color="red")

    ax.set_title("Average Hourly Load Profile (All Year)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Grid Purchases (GW)")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 2) Energy comparison (peak vs off-peak) ──
    ax    = axes[0, 1]
    x     = np.arange(n_cases)
    width = 0.35
    peak_vals    = [r["peak_energy_gwh"]    for r in all_results]
    offpeak_vals = [r["offpeak_energy_gwh"] for r in all_results]

    bars1 = ax.bar(x - width / 2, peak_vals,    width, label="Peak Energy (GWh)",     color="#e74c3c", edgecolor="white")
    bars2 = ax.bar(x + width / 2, offpeak_vals, width, label="Off-Peak Energy (GWh)", color="#4a90d9", edgecolor="white")
    ax.set_title("Energy by Peak / Off-Peak Period")
    ax.set_xticks(x)
    ax.set_xticklabels(case_names, fontsize=9)
    ax.set_ylabel("Energy (GWh)")
    ax.legend(fontsize=8)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    # ── 3) System peak demand comparison ──
    ax           = axes[1, 0]
    peak_demands = [r["system_peak_gw"] for r in all_results]
    bar_colors   = ["#e74c3c" if r["case_name"] == baseline["case_name"] else "#4a90d9"
                    for r in all_results]
    bars = ax.bar(x, peak_demands, 0.5, color=bar_colors, edgecolor="white")
    ax.set_title("System Peak Demand")
    ax.set_xticks(x)
    ax.set_xticklabels(case_names, fontsize=9)
    ax.set_ylabel("Peak Demand (GW)")
    for bar, val in zip(bars, peak_demands):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    for i, res in enumerate(all_results):
        if res["case_name"] != baseline["case_name"]:
            pct = 100 * (res["system_peak_gw"] - baseline["system_peak_gw"]) / baseline["system_peak_gw"]
            ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                    bars[i].get_height() * 0.95,
                    f"({pct:+.1f}%)", ha="center", va="top", fontsize=8, color="white",
                    fontweight="bold")

    # ── 4) Total cost comparison ──
    ax         = axes[1, 1]
    costs      = [r["total_cost_million"] for r in all_results]
    bar_colors = ["#e74c3c" if r["case_name"] == baseline["case_name"] else "#2ecc71"
                  for r in all_results]
    bars = ax.bar(x, costs, 0.5, color=bar_colors, edgecolor="white")
    ax.set_title("Total Electricity Expenditure")
    ax.set_xticks(x)
    ax.set_xticklabels(case_names, fontsize=9)
    ax.set_ylabel("Cost ($M)")
    for bar, val in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${val:.2f}M", ha="center", va="bottom", fontsize=9)
    for i, res in enumerate(all_results):
        if res["case_name"] != baseline["case_name"]:
            pct = 100 * (res["total_cost_million"] - baseline["total_cost_million"]) / baseline["total_cost_million"]
            ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                    bars[i].get_height() * 0.95,
                    f"({pct:+.1f}%)", ha="center", va="top", fontsize=8, color="white",
                    fontweight="bold")

    plot_path = os.path.join(output_dir, "peak_shifting_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")


def plot_hourly_delta(all_results, baseline, output_dir):
    """Plot hour-by-hour energy change from baseline for each case."""
    non_baseline = [r for r in all_results if r["case_name"] != baseline["case_name"]]
    if not non_baseline:
        return

    n    = len(non_baseline)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)
    fig.suptitle(f"Hourly Load Change vs Baseline ({baseline['case_name']})",
                 fontsize=14, fontweight="bold")

    bl_grid = baseline["_grid_gw"]
    bl_hod  = baseline["_hour_of_day"]
    bl_avg  = np.array([bl_grid[bl_hod == h].mean() for h in range(24)])

    for i, res in enumerate(non_baseline):
        ax    = axes[0, i]
        grid  = res["_grid_gw"]
        hod   = res["_hour_of_day"]
        avg   = np.array([grid[hod == h].mean() for h in range(24)])
        delta = avg - bl_avg

        colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in delta]
        ax.bar(range(24), delta, color=colors, edgecolor="white", lw=0.5)
        ax.axhline(0, color="black", lw=0.8)

        for h0, h1 in PEAK_HOURS.get(res["case_name"], []):
            ax.axvspan(h0, h1 + 1, alpha=0.1, color="orange", label="Peak window")

        ax.set_title(f"{res['case_name']}")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Δ Grid Purchases (GW)")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plot_path = os.path.join(output_dir, "hourly_load_delta.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


def save_detailed_csv(all_results, all_deltas, output_dir):
    """Save summary and detailed CSVs."""
    # Summary CSV
    summary_rows = []
    for res in all_results:
        row = {k: v for k, v in res.items() if not k.startswith("_")}
        row["system_peak_timestamp"] = str(row["system_peak_timestamp"])
        row.pop("top10_peaks", None)
        summary_rows.append(row)
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "case_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary CSV saved: {summary_path}")

    # Deltas CSV
    if all_deltas:
        deltas_df   = pd.DataFrame(all_deltas)
        deltas_path = os.path.join(output_dir, "case_deltas_vs_baseline.csv")
        deltas_df.to_csv(deltas_path, index=False)
        print(f"  Deltas CSV saved:  {deltas_path}")

    # Detailed hourly CSV per case
    for res in all_results:
        n     = len(res["_grid_gw"])
        start = pd.Timestamp(f"{YEAR}-01-01 01:00:00")
        timestamps = pd.date_range(start, periods=n, freq="h")

        detail_df = pd.DataFrame({
            "timestamp":           timestamps,
            "hour_of_day":         res["_hour_of_day"],
            "grid_purchases_gw":   res["_grid_gw"],
            "rate_dollar_per_kwh": res["_rate"],
            "hourly_cost_dollar":  res["_hourly_cost"],
            "is_peak":             res["_peak_mask"],
        })
        detail_path = os.path.join(output_dir, f"hourly_detail_{res['case_name']}.csv")
        detail_df.to_csv(detail_path, index=False)
        print(f"  Hourly detail CSV: {detail_path}")


def plot_avg_and_peak_day(all_results, output_dir):
    """
    Two-panel figure:
      Left:  average hourly load profile across all year (same as plot_comparison panel 1)
      Right: the 24-hour profile of each case's system-peak day
    Legend labels include the case name and the date of the peak day.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Average Load Profile vs. System Peak Day Profile",
                 fontsize=14, fontweight="bold")
    plt.subplots_adjust(wspace=0.3)

    n_cases = len(all_results)
    colors  = plt.cm.Set1(np.linspace(0, 0.8, n_cases))

    for i, res in enumerate(all_results):
        grid = res["_grid_gw"]
        hod  = res["_hour_of_day"]
        ts   = res["system_peak_timestamp"]  # pd.Timestamp of the peak hour

        # ── Left: annual average by hour of day ──
        avg_by_hour = np.array([grid[hod == h].mean() for h in range(24)])
        peak_date_str = pd.Timestamp(ts).strftime("%b %d")
        label = f"{res['case_name']} ({peak_date_str})"

        axes[0].plot(range(24), avg_by_hour, color=colors[i], lw=2, label=label)

        # ── Right: 24-hour slice of the peak day ──
        # Build a full timestamp array aligned to the grid array
        start      = pd.Timestamp(f"{YEAR}-01-01 01:00:00")
        timestamps = pd.date_range(start, periods=len(grid), freq="h")
        peak_date  = pd.Timestamp(ts).date()
        day_mask   = np.array([t.date() == peak_date for t in timestamps])
        day_hours  = np.array([t.hour for t in timestamps[day_mask]])
        day_grid   = grid[day_mask]

        # Sort by hour in case the day wraps oddly
        sort_idx  = np.argsort(day_hours)
        day_hours = day_hours[sort_idx]
        day_grid  = day_grid[sort_idx]

        axes[1].plot(day_hours, day_grid, color=colors[i], lw=2, label=label)

    # ── Formatting ──
    for ax, title in zip(axes, ["Annual Average Hourly Load", "System Peak Day Load"]):
        ax.set_title(title)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Grid Purchases (GW)")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "avg_and_peak_day_profiles.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  PEAK SHIFTING & COST COMPARISON")
    print(f"  Baseline: {BASELINE_CASE}")
    print(f"  Cases:    {', '.join(CASE_NAMES)}")
    print(f"{'=' * 70}")

    # Load all cases
    all_results = []
    for case_name in CASE_NAMES:
        print(f"\n  Loading {case_name}...")
        try:
            agg_df   = load_aggregated_ts(case_name)
            grid_col = "Grid Purchases [GW]"
            if grid_col not in agg_df.columns:
                grid_col = agg_df.select_dtypes(include=[np.number]).columns[0]
                print(f"    WARNING: '{grid_col}' used instead of 'Grid Purchases [GW]'")

            grid_gw = agg_df[grid_col].fillna(0).values
            rate    = load_rate(case_name)

            # Ensure same length
            n       = min(len(grid_gw), len(rate))
            grid_gw = grid_gw[:n]
            rate    = rate[:n]

            result = analyze_case(case_name, grid_gw, rate)
            all_results.append(result)
        except Exception as e:
            print(f"    ERROR loading {case_name}: {e}")
            continue

    if not all_results:
        print("No cases loaded. Check paths.")
        exit(1)

    # Find baseline
    baseline = next((r for r in all_results if r["case_name"] == BASELINE_CASE), None)
    if baseline is None:
        print(f"Baseline case '{BASELINE_CASE}' not found in results.")
        exit(1)

    # Compute deltas vs baseline
    # Recompute baseline metrics using each case's own peak windows for fair comparison
    all_deltas = []
    for res in all_results:
        if res["case_name"] == BASELINE_CASE:
            continue

        bl_recomputed = analyze_case(
            res["case_name"],     # use this case's peak definition
            baseline["_grid_gw"],
            baseline["_rate"],
        )
        bl_recomputed["case_name"] = BASELINE_CASE

        delta = compute_deltas(bl_recomputed, res)
        all_deltas.append(delta)

    # Print
    print_summary(all_results, all_deltas, baseline)

    # Plots
    plot_comparison(all_results, baseline, OUTPUT_DIR)
    plot_hourly_delta(all_results, baseline, OUTPUT_DIR)
    plot_avg_and_peak_day(all_results, OUTPUT_DIR)

    # CSVs
    save_detailed_csv(all_results, all_deltas, OUTPUT_DIR)

    print(f"\n{'=' * 70}")
    print(f"  Done! Output: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")