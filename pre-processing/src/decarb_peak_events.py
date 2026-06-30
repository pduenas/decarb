"""
ERCOT Peak Events Analyzer
===========================
Analyzes hourly grid data to identify:
  1. Which time-of-day periods see the majority of peak events
  2. When the absolute peak event of the year occurs
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def build_paths(case_name, base_dir):
    """Build file path and output directory from case_name."""
    case_dir = os.path.join(base_dir, case_name)
    filename = f"{case_name}_aggregated_ts.csv"
    filepath = os.path.join(case_dir, filename)
    return filepath, case_dir


def load_data(filepath):
    """Load the hourly time-series CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(filepath, sep=sep, engine="python")
            if len(df.columns) > 1:
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not parse {filepath}")

    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    print(f"  File: {filepath}")
    print(f"  Columns: {list(df.columns)}")
    return df


def build_datetime_index(df, year):
    """
    Build datetime index. Checks for existing datetime columns first,
    otherwise assigns hourly index starting year-01-01 01:00.
    """
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ["date", "time", "timestamp", "datetime"]):
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col).sort_index()
                print(f"  Using datetime column: '{col}'")
                return df
            except Exception:
                continue

    # Assign hourly index starting at year-01-01 01:00
    n = len(df)
    start = pd.Timestamp(f"{year}-01-01 01:00:00")
    idx = pd.date_range(start, periods=n, freq="h")
    df.index = idx
    print(f"  Assigned hourly index: {idx[0]} to {idx[-1]} ({n} hours)")
    return df


def identify_grid_column(df):
    """Find the column most likely representing grid purchases / demand."""
    keywords = ["grid", "purchase", "demand", "load", "consumption",
                "import", "power", "electric", "elec", "mw", "gw", "kw"]
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in keywords):
            return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        return numeric_cols[0]

    raise ValueError(f"No numeric column found. Columns: {list(df.columns)}")


def analyze_peaks(df, col, top_n, threshold_pct):
    """Identify peak events and their time-of-day distribution."""
    series = df[col].dropna()
    threshold = np.percentile(series, threshold_pct)

    # Annual peak
    peak_idx = series.idxmax()
    peak_val = series.max()
    annual_peak = {
        "timestamp": peak_idx,
        "value": peak_val,
        "hour": peak_idx.hour,
        "day_of_week": peak_idx.strftime("%A"),
    }

    # Top N peak hours
    top_n_events = series.nlargest(top_n)
    top_n_df = pd.DataFrame({
        "value": top_n_events,
        "hour": top_n_events.index.hour,
        "month": top_n_events.index.month,
        "day_of_week": top_n_events.index.dayofweek,
    })

    # Threshold-based peak events
    peak_events = series[series >= threshold]
    peak_df = pd.DataFrame({
        "value": peak_events,
        "hour": peak_events.index.hour,
        "month": peak_events.index.month,
        "day_of_week": peak_events.index.dayofweek,
    })

    # Time-of-day periods
    time_periods = {
        "Night (12am-6am)":       (0, 5),
        "Morning (6am-10am)":     (6, 9),
        "Midday (10am-2pm)":      (10, 13),
        "Afternoon (2pm-6pm)":    (14, 17),
        "Evening (6pm-10pm)":     (18, 21),
        "Late Night (10pm-12am)": (22, 23),
    }

    def count_by_period(events_df):
        counts = {}
        for name, (h0, h1) in time_periods.items():
            counts[name] = ((events_df["hour"] >= h0) & (events_df["hour"] <= h1)).sum()
        return counts

    return {
        "column": col,
        "series": series,
        "annual_peak": annual_peak,
        "top_n": {
            "n": top_n,
            "events": top_n_df,
            "by_period": count_by_period(top_n_df),
            "by_hour": top_n_df["hour"].value_counts().sort_index(),
            "by_month": top_n_df["month"].value_counts().sort_index(),
        },
        "threshold": {
            "pct": threshold_pct,
            "value": threshold,
            "count": len(peak_df),
            "events": peak_df,
            "by_period": count_by_period(peak_df),
            "by_hour": peak_df["hour"].value_counts().sort_index(),
            "by_month": peak_df["month"].value_counts().sort_index(),
        },
    }


def print_report(results, case_name):
    """Print formatted text report."""
    col = results["column"]
    ap = results["annual_peak"]
    topn = results["top_n"]
    th = results["threshold"]

    print(f"\n{'=' * 70}")
    print(f"  PEAK EVENT ANALYSIS — Case: {case_name} | Column: '{col}'")
    print(f"{'=' * 70}")

    # Annual peak
    print(f"\n{'─' * 50}")
    print(f"  PEAK EVENT OF THE YEAR")
    print(f"{'─' * 50}")
    print(f"  Timestamp:  {ap['timestamp']}")
    print(f"  Value:      {ap['value']:.4f}")
    print(f"  Date:       {ap['timestamp'].strftime('%B %d')}")
    print(f"  Hour:       {ap['hour']}:00 ({ap['day_of_week']})")

    # Top-N by time period
    print(f"\n{'─' * 50}")
    print(f"  TOP {topn['n']} PEAK HOURS — by Time Period")
    print(f"{'─' * 50}")
    total = sum(topn["by_period"].values())
    for period, count in topn["by_period"].items():
        pct = 100 * count / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {period:<28s} {count:>4d}  ({pct:5.1f}%)  {bar}")

    # Hourly breakdown
    print(f"\n  Hourly breakdown (Top {topn['n']}):")
    for hour in range(24):
        count = topn["by_hour"].get(hour, 0)
        pct = 100 * count / total if total else 0
        bar = "█" * int(pct)
        print(f"    {hour:02d}:00  {count:>3d}  ({pct:5.1f}%)  {bar}")

    # Monthly breakdown
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print(f"\n  Monthly breakdown (Top {topn['n']}):")
    for m in range(1, 13):
        count = topn["by_month"].get(m, 0)
        pct = 100 * count / total if total else 0
        bar = "█" * int(pct)
        print(f"    {month_names[m-1]}  {count:>3d}  ({pct:5.1f}%)  {bar}")

    # Threshold analysis
    print(f"\n{'─' * 50}")
    print(f"  P{th['pct']} THRESHOLD EVENTS (>= {th['value']:.4f})")
    print(f"  Total events: {th['count']}")
    print(f"{'─' * 50}")
    total_th = sum(th["by_period"].values())
    for period, count in th["by_period"].items():
        pct = 100 * count / total_th if total_th else 0
        bar = "█" * int(pct / 2)
        print(f"  {period:<28s} {count:>4d}  ({pct:5.1f}%)  {bar}")


def plot_results(results, case_name, output_dir):
    """Generate and save analysis plots."""
    col = results["column"]
    series = results["series"]
    topn = results["top_n"]
    ap = results["annual_peak"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Peak Event Analysis — {case_name} — '{col}'",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # 1) Full year with peaks highlighted
    ax = axes[0, 0]
    ax.plot(series.index, series.values, lw=0.3, color="#4a90d9", alpha=0.7, label="Hourly")
    top_vals = series.loc[series.index.isin(topn["events"].index)]
    ax.scatter(top_vals.index, top_vals.values, color="#e74c3c", s=8, zorder=5,
               label=f"Top {topn['n']} peaks", alpha=0.7)
    ax.scatter([ap["timestamp"]], [ap["value"]], color="#e74c3c", s=120, zorder=6,
               marker="*", edgecolors="black", linewidths=0.5, label="Annual peak")
    ax.set_title("Annual Profile with Peak Events")
    ax.set_xlabel("Date")
    ax.set_ylabel(col)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # 2) Peaks by hour of day
    ax = axes[0, 1]
    hours = list(range(24))
    counts = [topn["by_hour"].get(h, 0) for h in hours]
    mean_c, std_c = np.mean(counts), np.std(counts)
    colors = ["#e74c3c" if c > mean_c + std_c else "#4a90d9" for c in counts]
    ax.bar(hours, counts, color=colors, edgecolor="white", lw=0.5)
    ax.set_title(f"Top {topn['n']} Peak Events by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Count")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours], fontsize=7)

    # 3) Peaks by time period
    ax = axes[1, 0]
    periods = list(topn["by_period"].keys())
    pcounts = list(topn["by_period"].values())
    sorted_pairs = sorted(zip(pcounts, periods), reverse=True)
    s_counts, s_periods = zip(*sorted_pairs)
    pcolors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(s_periods)))
    bars = ax.barh(range(len(s_periods)), s_counts, color=pcolors, edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(s_periods)))
    ax.set_yticklabels(s_periods, fontsize=9)
    ax.set_title(f"Top {topn['n']} Peaks by Time Period")
    ax.set_xlabel("Count")
    for bar, c in zip(bars, s_counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(c), va="center", fontsize=9)

    # 4) Peaks by month
    ax = axes[1, 1]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mcounts = [topn["by_month"].get(m, 0) for m in range(1, 13)]
    mean_m, std_m = np.mean(mcounts), np.std(mcounts)
    mcolors = ["#e74c3c" if c > mean_m + std_m
               else "#f39c12" if c > mean_m
               else "#4a90d9" for c in mcounts]
    ax.bar(range(12), mcounts, color=mcolors, edgecolor="white", lw=0.5)
    ax.set_title(f"Top {topn['n']} Peak Events by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=9)

    plot_path = os.path.join(output_dir, f"peak_analysis_{case_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")


def save_csv_report(results, case_name, output_dir):
    """Save peak events detail to CSV."""
    events = results["top_n"]["events"].copy()
    events["timestamp"] = events.index
    events = events.sort_values("value", ascending=False)
    csv_path = os.path.join(output_dir, f"peak_events_{case_name}.csv")
    events.to_csv(csv_path, index=False)
    print(f"  CSV saved:  {csv_path}")


# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Configuration ───────────────────────────────────
    case_names = [
        # "elec_100_flat_results",
        # "elec_0_flat_results",
        # "elec_0_tou_results",
        # "elec_100_tou_results",
        # "elec_0_flat_cap_results"
        # "elec_100_flat_cap_results"
        "elec_0_tou_cap_results"
        "elec_100_tou_cap_results"
]
    base_dir        = r"D:\shared\ercot_project\out\decarb_results"
    year            = 2018
    top_n           = 100
    threshold_pct   = 95
    # ────────────────────────────────────────────────────

    for case_name in case_names:
        print(f"\n{'*' * 70}")
        print(f"  RUNNING CASE: {case_name}")
        print(f"{'*' * 70}")

        filepath, case_dir = build_paths(case_name, base_dir)
        print(f"Case: {case_name}")
        print(f"File: {filepath}")

        try:
            df = load_data(filepath)
            df = build_datetime_index(df, year)

            col = identify_grid_column(df)
            print(f"\nAnalyzing column: '{col}'")
            print(f"  Min: {df[col].min():.4f}  Max: {df[col].max():.4f}  Mean: {df[col].mean():.4f}")

            results = analyze_peaks(df, col, top_n, threshold_pct)
            print_report(results, case_name)
            plot_results(results, case_name, case_dir)
            save_csv_report(results, case_name, case_dir)
        except Exception as e:
            print(f"  ERROR processing case {case_name}: {e}")
            continue

    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}\n")