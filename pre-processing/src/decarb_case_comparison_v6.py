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
from matplotlib.lines import Line2D


# ── Configuration ──────────────────────────────────────────
ERCOT_ROOT = r"D:\shared\ercot_project"
RESULTS_DIR = os.path.join(ERCOT_ROOT, "out", "decarb_results")
INPUTS_DIR  = os.path.join(ERCOT_ROOT, "in",  "decarb_inputs_alt")
YEAR = 2018

BASELINE_CASE = "elec_0_flat_results"

CASE_NAMES = [
    "elec_0_flat_results",
    "elec_100_flat_results",
    "elec_0_tou_v2_results",
    "elec_100_tou_v2_results",
    "elec_0_flat_cap_results",
    "elec_100_flat_cap_results",
    "elec_0_tou_cap_results",
    "elec_100_tou_cap_results",
]

# Peak hour definitions per case (hours in 0-23 range, inclusive)
# Each entry is a list of (start_hour, end_hour) tuples
PEAK_HOURS = {
    "elec_0_flat_results":      [],
    "elec_100_flat_results":    [],
    # "elec_0_tou_results":       [(18, 22), (6, 10)],
    # "elec_100_tou_results":     [(18, 22), (6, 10)],
    "elec_0_tou_v2_results":  [(18, 22), (5, 9)],
    "elec_100_tou_v2_results":  [(18, 22), (5, 9)],
    "elec_0_flat_cap_results":  [],
    "elec_100_flat_cap_results": [],
    "elec_0_tou_cap_results":  [(18, 22), (5, 9)],
    "elec_100_tou_cap_results":  [(18, 22), (5, 9)],
}

# ── Plot styling ────────────────────────────────────────────
# Tariff family → color
#   Flat       : black
#   TOU        : default matplotlib blue (#1f77b4)
#   Flat w/ CC : red
# Electrification level → linestyle
#   elec_0   (60%)  : solid
#   elec_100 (100%) : dashed

CASE_STYLE = {
    "elec_0_flat_results":       {"color": "black",   "linestyle": "-",  "label": "Flat 60%"},
    "elec_100_flat_results":     {"color": "black",   "linestyle": "--", "label": "Flat 100%"},
    "elec_0_tou_v2_results":        {"color": "#1f77b4", "linestyle": "-",  "label": "TOU 60%"},
    "elec_100_tou_v2_results":      {"color": "#1f77b4", "linestyle": "--", "label": "TOU 100%"},
    "elec_0_flat_cap_results":   {"color": "red",     "linestyle": "-",  "label": "Flat w/ CC 60%"},
    "elec_100_flat_cap_results": {"color": "red",     "linestyle": "--", "label": "Flat w/ CC 100%"},
    "elec_0_tou_cap_results":   {"color": "#D5C655",     "linestyle": "-",  "label": "TOU w/ CC 60%"},
    "elec_100_tou_cap_results": {"color": "#D5C655",     "linestyle": "--", "label": "TOU w/ CC 100%"},
}

def get_style(case_name):
    """Return (color, linestyle, label) for a case, with a safe fallback."""
    s = CASE_STYLE.get(case_name, {"color": "gray", "linestyle": "-", "label": case_name})
    return s["color"], s["linestyle"], s["label"]


# Pretty display names for use in plot titles/suptitles.
# Falls back to the raw case name if not listed.
DISPLAY_NAMES = {
    "elec_0_flat_results":       "Flat Rate - 60 % Electrification",
    "elec_100_flat_results":     "Flat Rate - 100 % Electrification",
    "elec_0_tou_v2_results":     "TOU - 60 % Electrification",
    "elec_100_tou_v2_results":   "TOU - 100 % Electrification",
    "elec_0_flat_cap_results":   "Flat Rate w/ CC - 60 % Electrification",
    "elec_100_flat_cap_results": "Flat Rate w/ CC - 100 % Electrification",
    "elec_0_tou_cap_results":   "TOU Rate w/ CC - 60 % Electrification",
    "elec_100_tou_cap_results": "TOU Rate w/ CC - 100 % Electrification",
}

def display_name(case_name):
    """Return the pretty title-friendly name for a case."""
    return DISPLAY_NAMES.get(case_name, case_name)


# ── Paths ───────────────────────────────────────────────────
# Reference building ID for loading tm.csv (all buildings have identical rates)
REF_BLDG_ID = 100

OUTPUT_DIR = os.path.join(RESULTS_DIR, "peak_comparison")

# ── Electrification sweep config ────────────────────────────
# Each tariff has a summary.csv covering all electrification levels (60% → 100%)
SWEEP_DIRS = {
    "Flat":       "elec_all_flat",
    "TOU":        "elec_all_tou",
    "Flat w/ CC": "elec_all_flat_cap",
}

TARIFF_COLORS = {
    "Flat":       "black",
    "TOU":        "#1f77b4",
    "Flat w/ CC": "red",
}

# ── GenX system-cost config ─────────────────────────────────
# GenX case folders live in a separate tree and use a different naming
# convention: elec_NNNpct_<tariff>_ed_1bus, with each tariff potentially
# writing to its own results subfolder.
GENX_CASES_DIR = os.path.join(ERCOT_ROOT, "out", "genx_cases")

# Tariff key → (folder match regex, results subfolder, display name)
# Order matters here too: more specific keys (flat_cap) must be checked first.
import re as _re  # local alias to avoid touching other regex use
GENX_TARIFFS = [
    {
        "key":     "flat_cap",
        "label":   "Flat w/ CC",
        "pattern": _re.compile(r"flat_cap", _re.IGNORECASE),
        "results": "results",
    },
    {
        "key":     "tou",
        "label":   "TOU",
        "pattern": _re.compile(r"(?<![A-Za-z])tou(?![A-Za-z])", _re.IGNORECASE),
        "results": "results",
    },
    {
        "key":     "flat",
        "label":   "Flat",
        "pattern": _re.compile(r"(?<![A-Za-z])flat(?!_cap)", _re.IGNORECASE),
        "results": "results_1",
    },
]
GENX_ELEC_PATTERN = _re.compile(r"elec[_\-](\d+)pct", _re.IGNORECASE)
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
    grid_gw : np.ndarray — hourly electrical demand in GW (8760,)
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
    print(f"  Total electrical demand: {bl['total_energy_gwh']:>12.2f} GWh")
    print(f"  System peak demand:      {bl['system_peak_gw']:>12.4f} GW")
    print(f"  System peak at:          {bl['system_peak_timestamp']} (hour {bl['system_peak_hour']})")
    print(f"  Total cost:              ${bl['total_cost_million']:>11.2f} M")

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

        print(f"\n  Electrical Demand:")
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


def _genx_cost_for_case(case_name):
    """Map a demand-side case name (e.g. 'elec_100_tou_v2_results') to the
    corresponding GenX cTotal in $B. Returns NaN if the case folder or the
    cost can't be resolved.

    Mapping rules:
      - elec_0_*   → 60% electrification
      - elec_100_* → 100% electrification
      - tariff suffix:
          *_flat_cap_results  → flat_cap (results subfolder: 'results')
          *_tou_*_results     → tou      (results subfolder: 'results')
          *_flat_results      → flat     (results subfolder: 'results_1')
    """
    name = case_name.lower()

    # Electrification level
    if name.startswith("elec_0_"):
        pct = 60
    elif name.startswith("elec_100_"):
        pct = 100
    else:
        return np.nan

    # Tariff (check most specific first)
    if "flat_cap" in name:
        tariff_keys = ["flat_cap"]
    elif "_tou" in name:
        tariff_keys = ["tou"]
    elif "_flat" in name:
        tariff_keys = ["flat"]
    else:
        return np.nan

    tariff_cfg = next((t for t in GENX_TARIFFS if t["key"] in tariff_keys), None)
    if tariff_cfg is None:
        return np.nan

    if not os.path.isdir(GENX_CASES_DIR):
        return np.nan

    # Find a folder matching elec_<pct>pct + this tariff's pattern.
    # GenX folders pad to 3 digits (e.g. elec_060pct, elec_100pct).
    pct_token = f"elec_{pct:03d}pct"
    for entry in os.listdir(GENX_CASES_DIR):
        if pct_token not in entry.lower():
            continue
        if not tariff_cfg["pattern"].search(entry):
            continue
        case_dir = os.path.join(GENX_CASES_DIR, entry)
        if not os.path.isdir(case_dir):
            continue
        c_total = _read_genx_total_cost(case_dir, tariff_cfg["results"])
        if c_total is None:
            continue
        return c_total / 1e9  # $ → $B

    return np.nan


def plot_comparison(all_results, baseline, output_dir):
    """Generate comparison plots: system peak demand and GenX total system cost."""
    n_cases    = len(all_results)
    case_names = [r["case_name"] for r in all_results]
    x          = np.arange(n_cases)

    fig, axes = plt.subplots(1, 2, figsize=(24, 7))
    fig.suptitle(f"Peak Shifting & Cost Comparison\nBaseline: {display_name(baseline['case_name'])}",
                 fontsize=15, fontweight="bold")
    plt.subplots_adjust(wspace=0.25)

    # ── 1) System peak demand comparison ──
    ax           = axes[0]
    peak_demands = [r["system_peak_gw"] for r in all_results]
    bar_colors_3 = [get_style(r["case_name"])[0] for r in all_results]
    bar_hatch_3  = ["" if get_style(r["case_name"])[1] == "-" else "//" for r in all_results]
    bar_labels_3 = [get_style(r["case_name"])[2] for r in all_results]

    bars = ax.bar(x, peak_demands, 0.5,
                  color=bar_colors_3, hatch=bar_hatch_3, edgecolor="white", alpha=0.85)
    ax.set_title("System Peak Electrical Demand")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels_3, fontsize=9)
    ax.set_ylabel("Peak Electrical Demand (GW)")
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

    # ── 2) GenX total system cost ──
    ax = axes[1]
    genx_costs = [_genx_cost_for_case(r["case_name"]) for r in all_results]
    bar_colors_4 = [get_style(r["case_name"])[0] for r in all_results]
    bar_hatch_4  = ["" if get_style(r["case_name"])[1] == "-" else "//" for r in all_results]
    bar_labels_4 = [get_style(r["case_name"])[2] for r in all_results]

    bars = ax.bar(x, genx_costs, 0.5,
                  color=bar_colors_4, hatch=bar_hatch_4, edgecolor="white", alpha=0.85)
    ax.set_title("Total Capacity Expansion Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels_4, fontsize=9)
    ax.set_ylabel("System Cost ($B / year)")
    for bar, val in zip(bars, genx_costs):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"${val:.2f}B", ha="center", va="bottom", fontsize=9)

    # % delta vs baseline (only when baseline GenX cost resolved)
    baseline_genx = _genx_cost_for_case(baseline["case_name"])
    if not np.isnan(baseline_genx) and baseline_genx > 0:
        for i, (res, val) in enumerate(zip(all_results, genx_costs)):
            if res["case_name"] == baseline["case_name"] or np.isnan(val):
                continue
            pct = 100 * (val - baseline_genx) / baseline_genx
            ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                    bars[i].get_height() * 0.95,
                    f"({pct:+.1f}%)", ha="center", va="top", fontsize=8, color="white",
                    fontweight="bold")

    # If any cases came back NaN, flag them so the plot doesn't silently lie
    missing = [r["case_name"] for r, v in zip(all_results, genx_costs) if np.isnan(v)]
    if missing:
        print(f"  [WARN] GenX cTotal could not be resolved for: {missing}")

    plot_path = os.path.join(output_dir, "peak_shifting_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")


def plot_hourly_delta(all_results, baseline, output_dir):
    """Plot hour-by-hour electrical demand change from baseline for each case.

    Layout: 3x2 grid. Upper-left is empty, upper-right is Flat 100%
    (elec_100_flat_results). Remaining cases fill the lower two rows
    left-to-right, top-to-bottom. All panels share the same y-axis scale.
    """
    non_baseline = [r for r in all_results if r["case_name"] != baseline["case_name"]]
    if not non_baseline:
        return

    bl_grid = baseline["_grid_gw"]
    bl_hod  = baseline["_hour_of_day"]
    bl_avg  = np.array([bl_grid[bl_hod == h].mean() for h in range(24)])

    # Pre-compute deltas so we can determine a shared y-axis range
    case_deltas = {}
    for res in non_baseline:
        grid = res["_grid_gw"]
        hod  = res["_hour_of_day"]
        avg  = np.array([grid[hod == h].mean() for h in range(24)])
        case_deltas[res["case_name"]] = avg - bl_avg

    all_delta_vals = np.concatenate(list(case_deltas.values()))
    ymax = np.max(np.abs(all_delta_vals)) * 1.15  # 15% headroom
    ylim = (-ymax, ymax)

    # ── Assign cases to grid positions ──
    # Slot map: (row, col) → case_name (or None for empty)
    # Upper row: [empty, Flat 100%]
    # Remaining cases fill (1,0), (1,1), (2,0), (2,1) in order.
    flat_100_name = "elec_100_flat_results"
    others = [r for r in non_baseline if r["case_name"] != flat_100_name]

    slot_assignments = {(0, 0): None, (0, 1): None}
    if any(r["case_name"] == flat_100_name for r in non_baseline):
        slot_assignments[(0, 1)] = flat_100_name

    fill_slots = [(1, 0), (1, 1), (2, 0), (2, 1)]
    for slot, res in zip(fill_slots, others):
        slot_assignments[slot] = res["case_name"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle(f"Hourly Electrical Demand Change vs Baseline ({display_name(baseline['case_name'])})",
                 fontsize=14, fontweight="bold")

    # Map case_name → result for quick lookup
    by_name = {r["case_name"]: r for r in non_baseline}

    for (row, col), case_name in slot_assignments.items():
        ax = axes[row, col]

        if case_name is None:
            ax.axis("off")
            continue

        res   = by_name[case_name]
        color, ls, label = get_style(case_name)
        delta = case_deltas[case_name]

        bar_alpha = [0.85 if d > 0 else 0.45 for d in delta]
        for h, (d, a) in enumerate(zip(delta, bar_alpha)):
            ax.bar(h, d, color=color, alpha=a, edgecolor="white", lw=0.5)

        ax.axhline(0, color="black", lw=0.8)

        for h0, h1 in PEAK_HOURS.get(case_name, []):
            ax.axvspan(h0, h1, alpha=0.1, color="orange", label="Peak window")

        ax.set_title(label)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Δ Electrical Demand (GW)")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax.set_ylim(ylim)
        ax.tick_params(axis="y", labelleft=True)  # show y-tick labels on all panels (override sharey)
        ax.grid(True, alpha=0.3)

        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
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
            "timestamp":                timestamps,
            "hour_of_day":              res["_hour_of_day"],
            "electrical_demand_gw":     res["_grid_gw"],
            "rate_dollar_per_kwh":      res["_rate"],
            "hourly_cost_dollar":       res["_hourly_cost"],
            "is_peak":                  res["_peak_mask"],
        })
        detail_path = os.path.join(output_dir, f"hourly_detail_{res['case_name']}.csv")
        detail_df.to_csv(detail_path, index=False)
        print(f"  Hourly detail CSV: {detail_path}")


def plot_avg_and_peak_day(all_results, output_dir):
    """
    Two-panel figure:
      Left:  average hourly electrical demand profile across all year
      Right: the 24-hour profile of each case's system-peak day
    Single shared legend below both panels, two rows:
      Top row    → 60% electrification cases
      Bottom row → 100% electrification cases
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Average Electrical Demand Profile vs. System Peak Day Profile",
                 fontsize=16, fontweight="bold")
    plt.subplots_adjust(wspace=0.3)

    # Tariff ordering used for the legend. Both the 60% row and the 100% row
    # are sorted the same way so the columns line up vertically.
    TARIFF_ORDER = ["Flat", "TOU", "Flat w/ CC", "TOU w/ CC"]

    def _tariff_key(case_name):
        """Return the tariff sort index for a case name (lower = earlier)."""
        _, _, lbl = get_style(case_name)
        # Strip the trailing "60%"/"100%" off the style label to get the tariff
        base = lbl.rsplit(" ", 1)[0]
        try:
            return TARIFF_ORDER.index(base)
        except ValueError:
            return len(TARIFF_ORDER)  # unknowns go to the end

    # Track handles/labels per electrification level along with their tariff
    # sort key, so we can reorder them after collecting.
    entries_60  = []  # list of (sort_key, handle, label)
    entries_100 = []

    for res in all_results:
        case_name = res["case_name"]
        color, ls, label = get_style(case_name)
        grid = res["_grid_gw"]
        hod  = res["_hour_of_day"]
        ts   = res["system_peak_timestamp"]

        # ── Left: annual average by hour of day ──
        avg_by_hour   = np.array([grid[hod == h].mean() for h in range(24)])
        peak_date_str = pd.Timestamp(ts).strftime("%b %d")
        full_label    = f"{label} (peak {peak_date_str})"

        line_left, = axes[0].plot(range(24), avg_by_hour,
                                   color=color, linestyle=ls, lw=2, label=full_label)

        # ── Right: 24-hour slice of the peak day ──
        start      = pd.Timestamp(f"{YEAR}-01-01 01:00:00")
        timestamps = pd.date_range(start, periods=len(grid), freq="h")
        peak_date  = pd.Timestamp(ts).date()
        day_mask   = np.array([t.date() == peak_date for t in timestamps])
        day_hours  = np.array([t.hour for t in timestamps[day_mask]])
        day_grid   = grid[day_mask]

        sort_idx  = np.argsort(day_hours)
        day_hours = day_hours[sort_idx]
        day_grid  = day_grid[sort_idx]

        axes[1].plot(day_hours, day_grid, color=color, linestyle=ls, lw=2)

        # Classify into 60% vs 100% row, tagging with tariff sort key
        entry = (_tariff_key(case_name), line_left, full_label)
        if case_name.startswith("elec_100_"):
            entries_100.append(entry)
        else:
            entries_60.append(entry)

    # Sort each row by tariff order so 60% and 100% columns line up
    entries_60.sort(key=lambda e: e[0])
    entries_100.sort(key=lambda e: e[0])

    handles_60,  labels_60  = [e[1] for e in entries_60],  [e[2] for e in entries_60]
    handles_100, labels_100 = [e[1] for e in entries_100], [e[2] for e in entries_100]

    # ── Formatting ──
    for ax, title in zip(axes, ["Annual Average Hourly Electrical Demand",
                                 "System Peak Day Electrical Demand"]):
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Hour of Day", fontsize=13)
        ax.set_ylabel("Electrical Demand (GW)", fontsize=13)
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=10)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, alpha=0.3)

    # ── Single shared legend, two rows, below the figure ──
    # Pad the shorter row out to equal length so columns align nicely.
    n_cols = max(len(handles_60), len(handles_100))

    def _pad(handles, labels, n):
        while len(handles) < n:
            handles.append(Line2D([0], [0], color="none", label=""))
            labels.append("")
        return handles, labels

    handles_60,  labels_60  = _pad(handles_60,  labels_60,  n_cols)
    handles_100, labels_100 = _pad(handles_100, labels_100, n_cols)

    # Concatenate so matplotlib's row-major fill produces:
    #   row 1 = 60% entries, row 2 = 100% entries (both in TARIFF_ORDER)
    combined_handles = handles_60 + handles_100
    combined_labels  = labels_60  + labels_100

    fig.legend(
        combined_handles, combined_labels,
        loc="lower center",
        ncol=n_cols,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.12, 1, 0.96])

    plot_path = os.path.join(output_dir, "avg_and_peak_day_profiles.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


# ─────────────────────────────────────────────────────────
# Electrification sweep — three tariffs across all elec %
# ─────────────────────────────────────────────────────────
def load_sweep_summary(tariff_name, sweep_dir):
    """Load summary.csv for one tariff sweep.

    Returns DataFrame with columns including:
      - electrification_pct (e.g. 'elec_060pct')
      - peak_buy           (kW → converted to GW)
      - total_cost         ($ → converted to $B)
      - elec_pct_int       (parsed integer, 60..100)
    """
    path = os.path.join(RESULTS_DIR, sweep_dir, "summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sweep summary not found: {path}")

    df = pd.read_csv(path)
    df["elec_pct_int"]   = df["electrification_pct"].str.extract(r"elec_(\d+)pct").astype(int)
    df["peak_buy_gw"]    = df["peak_buy"] / 1e3      # MW → GW
    df["total_cost_b"]   = df["total_cost"] / 1e9    # $  → $B
    df = df.sort_values("elec_pct_int").reset_index(drop=True)
    print(f"  Loaded sweep: {tariff_name} ({len(df)} rows) — {path}")
    return df


def plot_sweep_comparison(output_dir):
    """Plot system peak demand and total expenditure across electrification levels
    for each tariff family. Produces TWO separate PNG files."""
    # Load all three sweeps
    sweeps = {}
    for tariff, sweep_dir in SWEEP_DIRS.items():
        try:
            sweeps[tariff] = load_sweep_summary(tariff, sweep_dir)
        except Exception as e:
            print(f"  WARNING: could not load sweep for {tariff}: {e}")

    if not sweeps:
        print("  No sweep data loaded; skipping sweep plots.")
        return

    # Use the union of electrification levels across tariffs (sorted)
    all_pcts = sorted({pct for df in sweeps.values() for pct in df["elec_pct_int"]})
    n_levels  = len(all_pcts)
    n_tariffs = len(sweeps)

    x = np.arange(n_levels)
    bar_width = 0.8 / n_tariffs

    # ── Plot 1: System Peak Electrical Demand ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (tariff, df) in enumerate(sweeps.items()):
        # Align values to the master x-axis (NaN where missing)
        lookup = dict(zip(df["elec_pct_int"], df["peak_buy_gw"]))
        vals   = [lookup.get(p, np.nan) for p in all_pcts]

        offset = (i - (n_tariffs - 1) / 2) * bar_width
        bars   = ax.bar(x + offset, vals, bar_width,
                        color=TARIFF_COLORS.get(tariff, "gray"),
                        edgecolor="white", alpha=0.85, label=tariff)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_title("System Peak Electrical Demand by Tariff and Electrification Level")
    ax.set_xlabel("Electrification Level")
    ax.set_ylabel("Peak Electrical Demand (GW)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in all_pcts])
    ax.legend(title="Tariff", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    peak_path = os.path.join(output_dir, "sweep_system_peak_demand_bar.png")
    plt.savefig(peak_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {peak_path}")

    # ── Plot 1b: System Peak Electrical Demand (line version) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for tariff, df in sweeps.items():
        lookup = dict(zip(df["elec_pct_int"], df["peak_buy_gw"]))
        vals   = [lookup.get(p, np.nan) for p in all_pcts]
        color  = TARIFF_COLORS.get(tariff, "gray")
        ax.plot(x, vals,
                marker="o", linewidth=2, markersize=8,
                color=color, markerfacecolor="white", markeredgewidth=2,
                label=tariff)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.1f}", (xi, v),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7.5,
                            color=color, fontweight="bold")

    ax.set_title("System Peak Electrical Demand by Tariff and Electrification Level")
    ax.set_xlabel("Electrification Level")
    ax.set_ylabel("Peak Electrical Demand (GW)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in all_pcts])
    ax.legend(title="Tariff", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    peak_line_path = os.path.join(output_dir, "sweep_system_peak_demand_line.png")
    plt.savefig(peak_line_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {peak_line_path}")

    # ── Plot 2: Total Electricity Expenditure ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (tariff, df) in enumerate(sweeps.items()):
        lookup = dict(zip(df["elec_pct_int"], df["total_cost_b"]))
        vals   = [lookup.get(p, np.nan) for p in all_pcts]

        offset = (i - (n_tariffs - 1) / 2) * bar_width
        bars   = ax.bar(x + offset, vals, bar_width,
                        color=TARIFF_COLORS.get(tariff, "gray"),
                        edgecolor="white", alpha=0.85, label=tariff)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"${v:.1f}B", ha="center", va="bottom", fontsize=7)

    ax.set_title("Total Electricity Expenditure by Tariff and Electrification Level")
    ax.set_xlabel("Electrification Level")
    ax.set_ylabel("Cost ($B)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in all_pcts])
    ax.legend(title="Tariff", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    cost_path = os.path.join(output_dir, "sweep_total_expenditure.png")
    plt.savefig(cost_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {cost_path}")


# ─────────────────────────────────────────────────────────
# GenX system costs — read costs.csv from each case folder
# ─────────────────────────────────────────────────────────
def _classify_genx_tariff(folder_name):
    """Return the GenX tariff config dict that matches a folder name, or None."""
    for tariff in GENX_TARIFFS:
        if tariff["pattern"].search(folder_name):
            return tariff
    return None


def _read_genx_total_cost(case_dir, results_subfolder):
    """Return cTotal from costs.csv in the given case dir (in $), or None."""
    path = os.path.join(case_dir, results_subfolder, "costs.csv")
    if not os.path.exists(path):
        print(f"  [WARN] missing: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] could not read {path}: {e}")
        return None

    # costs.csv format: first column is the cost name, second is Total
    if df.shape[1] < 2:
        return None
    name_col, total_col = df.columns[0], df.columns[1]
    row = df[df[name_col].astype(str).str.strip() == "cTotal"]
    if row.empty:
        return None
    try:
        return float(row.iloc[0][total_col])
    except (ValueError, TypeError):
        return None


def load_genx_system_costs():
    """Scan GENX_CASES_DIR for case folders and pull cTotal from each.

    Returns a dict: {tariff_label: {elec_pct_int: cost_in_billions}}
    """
    if not os.path.isdir(GENX_CASES_DIR):
        print(f"  [WARN] GenX cases dir not found: {GENX_CASES_DIR}")
        return {}

    results = {t["label"]: {} for t in GENX_TARIFFS}

    for entry in sorted(os.listdir(GENX_CASES_DIR)):
        case_dir = os.path.join(GENX_CASES_DIR, entry)
        if not os.path.isdir(case_dir):
            continue

        m = GENX_ELEC_PATTERN.search(entry)
        if not m:
            continue
        pct = int(m.group(1))

        tariff = _classify_genx_tariff(entry)
        if tariff is None:
            continue

        c_total = _read_genx_total_cost(case_dir, tariff["results"])
        if c_total is None:
            continue

        results[tariff["label"]][pct] = c_total / 1e9  # $ → $B
        print(f"  [OK]  {tariff['label']:11s}  elec={pct:3d}%  cTotal=${c_total / 1e9:.2f}B")

    # Drop tariffs that produced nothing
    results = {k: v for k, v in results.items() if v}
    return results


def plot_genx_system_costs(output_dir):
    """Make two plots from GenX cTotal values across electrification levels:
       - sweep_genx_system_cost_bar.png  (grouped bars per tariff)
       - sweep_genx_system_cost_line.png (overlaid line per tariff)
    Both use the existing TARIFF_COLORS palette (black/blue/red).
    """
    data = load_genx_system_costs()
    if not data:
        print("  No GenX system-cost data loaded; skipping system-cost plots.")
        return

    # Master x-axis: union of electrification levels across tariffs
    all_pcts = sorted({pct for d in data.values() for pct in d.keys()})
    if not all_pcts:
        return
    n_levels  = len(all_pcts)
    n_tariffs = len(data)
    x         = np.arange(n_levels)
    bar_width = 0.8 / n_tariffs

    # Maintain a consistent tariff order (Flat, TOU, Flat w/ CC) when present
    tariff_order = [t for t in ["Flat", "TOU", "Flat w/ CC"] if t in data]

    # ── Bar version ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, tariff in enumerate(tariff_order):
        lookup = data[tariff]
        vals   = [lookup.get(p, np.nan) for p in all_pcts]

        offset = (i - (n_tariffs - 1) / 2) * bar_width
        bars   = ax.bar(x + offset, vals, bar_width,
                        color=TARIFF_COLORS.get(tariff, "gray"),
                        edgecolor="white", alpha=0.85, label=tariff)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"${v:.1f}B", ha="center", va="bottom", fontsize=7)

    ax.set_title("Capacity Expansion Costs by Tariff and Electrification Level")
    ax.set_xlabel("Electrification Level")
    ax.set_ylabel("Total System Cost ($B / year)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in all_pcts])
    ax.legend(title="Tariff", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "sweep_genx_system_cost_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {bar_path}")

    # ── Line version ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for tariff in tariff_order:
        lookup = data[tariff]
        vals   = [lookup.get(p, np.nan) for p in all_pcts]
        color  = TARIFF_COLORS.get(tariff, "gray")
        ax.plot(x, vals,
                marker="o", linewidth=2, markersize=8,
                color=color, markerfacecolor="white", markeredgewidth=2,
                label=tariff)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.annotate(f"${v:.2f}B", (xi, v),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7.5,
                            color=color, fontweight="bold")

    ax.set_title("Capacity Expansion Costs by Tariff and Electrification Level")
    ax.set_xlabel("Electrification Level")
    ax.set_ylabel("Total System Cost ($B / year)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in all_pcts])
    ax.legend(title="Tariff", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    line_path = os.path.join(output_dir, "sweep_genx_system_cost_line.png")
    plt.savefig(line_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {line_path}")


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
    all_deltas = []
    for res in all_results:
        if res["case_name"] == BASELINE_CASE:
            continue

        bl_recomputed = analyze_case(
            res["case_name"],
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

    # Electrification sweep plots (across all elec %, three tariffs)
    print(f"\n  Loading electrification sweeps...")
    plot_sweep_comparison(OUTPUT_DIR)

    # GenX system-cost plots (bar + line)
    print(f"\n  Loading GenX system costs...")
    plot_genx_system_costs(OUTPUT_DIR)

    # CSVs
    save_detailed_csv(all_results, all_deltas, OUTPUT_DIR)

    print(f"\n{'=' * 70}")
    print(f"  Done! Output: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")