"""
Thermal Coefficient Comparison: Per-Top-Hour Groups vs Rest
============================================================
For a chosen decarb case, loads the per-building peak-hour CSV produced
by the upstream analysis. For each peak distribution (summer / winter /
full-year), it identifies the 5 most populated hour-of-day bins, then
splits buildings into 6 groups: one group per top hour, plus a 'rest'
group for everything else. The thermal model coefficients (k1, k2, k3)
are compared across those 6 groups so we can see how peak time of day
relates to thermal characteristics.

Outputs (per peak-type: summer, winter, full):
  - summary stats CSV (mean, median, std, count) per group per coefficient
  - box plot (k1, k2, k3 side-by-side, 6 groups each)
  - violin plot (same layout)
  - overlaid histogram (one subplot per coefficient, one curve per group)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Configuration ──────────────────────────────────────────
CASE_NAME = "elec_0_flat_results"
ERCOT_ROOT = r"D:\shared\ercot_project"

PEAK_CSV = os.path.join(
    ERCOT_ROOT, "out", "decarb_results", CASE_NAME,
    "per_building_peak_hours_decarb.csv",
)
COEFF_PATH = os.path.join(
    ERCOT_ROOT, "out", "thermal_model", "baseline",
    "regression_coeff_baseline.parquet",
)
OUTPUT_DIR = os.path.join(
    ERCOT_ROOT, "out", "decarb_results", CASE_NAME, "bldg_char_comp",
)

PEAK_TYPES = ["summer", "winter", "full"]
COEFFS = ["k1", "k2", "k3"]
TOP_N_BINS = 5
HIST_BINS = 50
REST_LABEL = "rest"

# Trim values outside [CLIP_LO_PCT, CLIP_HI_PCT] (within each coefficient,
# across all groups) before plotting, so extreme outliers don't squash the
# interesting part of the distribution.
CLIP_LO_PCT = 5
CLIP_HI_PCT = 95
# ───────────────────────────────────────────────────────────


def _sci_yaxis(ax):
    """Force scientific notation on the y-axis."""
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                        useMathText=True)


def _clip_bounds(df, group_col, ordered_labels, coeff,
                 lo_pct=CLIP_LO_PCT, hi_pct=CLIP_HI_PCT):
    """Compute the [lo, hi] percentile bounds across all groups for a
    given coefficient. Used to filter outliers consistently."""
    all_vals = np.concatenate([
        df.loc[df[group_col] == g, coeff].dropna().values
        for g in ordered_labels
    ]) if len(ordered_labels) else np.array([])
    if len(all_vals) == 0:
        return None, None
    lo, hi = np.percentile(all_vals, [lo_pct, hi_pct])
    return lo, hi


def top_n_hours(series, n=TOP_N_BINS):
    """Return the n most-populated hour-of-day bins as a sorted list (0..23)."""
    counts = series.value_counts()
    return sorted(counts.head(n).index.astype(int).tolist())


def hour_label(h):
    return f"h{int(h):02d}"


def build_groups(merged, peak_col, top_hours):
    """Assign each row a group label: 'hHH' for each top hour, else 'rest'.

    Returns:
        group_series: pd.Series of labels aligned with merged
        ordered_labels: labels in plotting order (top hours sorted, then 'rest')
        colors: hex colors aligned with ordered_labels — hours colored along a
                cyclic colormap so time-of-day reads visually; 'rest' is grey.
    """
    grp = np.where(
        merged[peak_col].isin(top_hours),
        merged[peak_col].apply(lambda h: hour_label(h)),
        REST_LABEL,
    )
    grp = pd.Series(grp, index=merged.index)

    ordered_hours = sorted(top_hours)
    ordered_labels = [hour_label(h) for h in ordered_hours] + [REST_LABEL]

    cmap = cm.get_cmap("twilight_shifted")
    rgba = [cmap(h / 24.0) for h in ordered_hours] + [(0.6, 0.6, 0.6, 1.0)]
    hex_colors = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for (r, g, b, _) in rgba
    ]
    return grp, ordered_labels, hex_colors


def summarize(df, group_col, coeffs, ordered_labels):
    rows = []
    for grp in ordered_labels:
        sub = df[df[group_col] == grp]
        for c in coeffs:
            vals = sub[c].dropna()
            rows.append({
                "group": grp,
                "coefficient": c,
                "n": int(len(vals)),
                "mean": vals.mean() if len(vals) else np.nan,
                "median": vals.median() if len(vals) else np.nan,
                "std": vals.std() if len(vals) else np.nan,
                "min": vals.min() if len(vals) else np.nan,
                "p25": vals.quantile(0.25) if len(vals) else np.nan,
                "p75": vals.quantile(0.75) if len(vals) else np.nan,
                "max": vals.max() if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def _gather(df, group_col, ordered_labels, coeff, bounds=None):
    """List of arrays per group, NaNs dropped. If bounds=(lo, hi) is given,
    clip values to that closed interval."""
    out = []
    for g in ordered_labels:
        vals = df.loc[df[group_col] == g, coeff].dropna().values
        if bounds is not None:
            lo, hi = bounds
            vals = vals[(vals >= lo) & (vals <= hi)]
        out.append(vals)
    return out


def plot_box(df, group_col, coeffs, ordered_labels, colors, title, save_path):
    fig, axes = plt.subplots(1, len(coeffs), figsize=(5 * len(coeffs), 5.5))
    if len(coeffs) == 1:
        axes = [axes]
    for ax, c in zip(axes, coeffs):
        bounds = _clip_bounds(df, group_col, ordered_labels, c)
        data = _gather(df, group_col, ordered_labels, c, bounds=bounds)
        bp = ax.boxplot(data, tick_labels=ordered_labels, showfliers=False,
                        patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(c)
        ax.set_ylabel(c)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=30)
        _sci_yaxis(ax)
    fig.suptitle(f"{title}  (clipped to {CLIP_LO_PCT}–{CLIP_HI_PCT} pct)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_violin(df, group_col, coeffs, ordered_labels, colors, title, save_path):
    fig, axes = plt.subplots(1, len(coeffs), figsize=(5 * len(coeffs), 5.5))
    if len(coeffs) == 1:
        axes = [axes]
    for ax, c in zip(axes, coeffs):
        bounds = _clip_bounds(df, group_col, ordered_labels, c)
        data = _gather(df, group_col, ordered_labels, c, bounds=bounds)
        positions, plot_data, plot_colors = [], [], []
        for i, (d, col) in enumerate(zip(data, colors), start=1):
            if len(d) > 0:
                positions.append(i)
                plot_data.append(d)
                plot_colors.append(col)
        if not plot_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        parts = ax.violinplot(plot_data, positions=positions,
                              showmeans=True, showmedians=True)
        for body, col in zip(parts["bodies"], plot_colors):
            body.set_facecolor(col)
            body.set_alpha(0.65)
            body.set_edgecolor("black")
        ax.set_xticks(range(1, len(ordered_labels) + 1))
        ax.set_xticklabels(ordered_labels)
        ax.set_title(c)
        ax.set_ylabel(c)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=30)
        _sci_yaxis(ax)
    fig.suptitle(f"{title}  (clipped to {CLIP_LO_PCT}–{CLIP_HI_PCT} pct)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hist(df, group_col, coeffs, ordered_labels, colors, title, save_path):
    """Step-histograms (density) so 6 overlapping groups stay legible."""
    fig, axes = plt.subplots(1, len(coeffs), figsize=(5 * len(coeffs), 5.5))
    if len(coeffs) == 1:
        axes = [axes]
    for ax, c in zip(axes, coeffs):
        bounds = _clip_bounds(df, group_col, ordered_labels, c)
        groups_data = _gather(df, group_col, ordered_labels, c, bounds=bounds)
        if not any(len(d) for d in groups_data):
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        lo, hi = bounds
        if lo == hi:
            all_vals = np.concatenate(groups_data)
            lo, hi = all_vals.min(), all_vals.max()
        bins = np.linspace(lo, hi, HIST_BINS)
        for d, lbl, col in zip(groups_data, ordered_labels, colors):
            if len(d) == 0:
                continue
            ax.hist(d, bins=bins, histtype="step", linewidth=2,
                    density=True, color=col,
                    label=f"{lbl} (n={len(d)})")
        ax.set_title(c)
        ax.set_xlabel(c)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),
                            useMathText=True)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                            useMathText=True)
    fig.suptitle(f"{title}  (clipped to {CLIP_LO_PCT}–{CLIP_HI_PCT} pct)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading peak CSV: {PEAK_CSV}")
    peaks = pd.read_csv(PEAK_CSV)
    print(f"  {len(peaks):,} buildings with peak data")

    print(f"Loading coefficients: {COEFF_PATH}")
    coeffs = pd.read_parquet(COEFF_PATH)
    print(f"  {len(coeffs):,} buildings with coefficients")
    missing = [c for c in COEFFS if c not in coeffs.columns]
    if missing:
        raise ValueError(f"Missing coefficient columns: {missing}")

    peaks["bldg_id"] = peaks["bldg_id"].astype(np.int64)
    coeffs["bldg_id"] = coeffs["bldg_id"].astype(np.int64)

    merged = peaks.merge(coeffs[["bldg_id"] + COEFFS], on="bldg_id", how="inner")
    print(f"  {len(merged):,} buildings after inner-join on bldg_id\n")
    if len(merged) == 0:
        raise RuntimeError("Join produced 0 rows. Check bldg_id matching.")

    for peak_type in PEAK_TYPES:
        col = f"{peak_type}_peak_hour"
        if col not in merged.columns:
            print(f"[skip] column {col} not in peak CSV")
            continue

        top_hours = top_n_hours(merged[col], TOP_N_BINS)
        merged["_grp"], ordered_labels, colors = build_groups(
            merged, col, top_hours
        )

        print(f"\n{'=' * 70}")
        print(f"  PEAK TYPE: {peak_type.upper()}  (col = {col})")
        print(f"{'=' * 70}")
        print(f"  Top-{TOP_N_BINS} hours (sorted): {top_hours}")
        for lbl in ordered_labels:
            n = (merged["_grp"] == lbl).sum()
            print(f"    {lbl}: {n:,}")

        stats = summarize(merged, "_grp", COEFFS, ordered_labels)
        stats_path = os.path.join(OUTPUT_DIR, f"stats_{peak_type}.csv")
        stats.to_csv(stats_path, index=False)
        print(f"\n  Stats → {stats_path}")
        for c in COEFFS:
            sub = (stats[stats["coefficient"] == c]
                   .set_index("group")
                   .loc[ordered_labels])
            print(f"\n  {c}:")
            print(sub[["n", "mean", "median", "std", "p25", "p75"]]
                  .to_string(float_format=lambda x: f"{x:.6g}"))

        title_suffix = (f"{CASE_NAME} — {peak_type.title()} peak — "
                        f"per-hour groups (top {TOP_N_BINS}) + rest")

        plot_box(
            merged, "_grp", COEFFS, ordered_labels, colors,
            f"Box plot: {title_suffix}",
            os.path.join(OUTPUT_DIR, f"box_{peak_type}.png"),
        )
        plot_violin(
            merged, "_grp", COEFFS, ordered_labels, colors,
            f"Violin plot: {title_suffix}",
            os.path.join(OUTPUT_DIR, f"violin_{peak_type}.png"),
        )
        plot_hist(
            merged, "_grp", COEFFS, ordered_labels, colors,
            f"Histogram: {title_suffix}",
            os.path.join(OUTPUT_DIR, f"hist_{peak_type}.png"),
        )
        print(f"  Plots → {OUTPUT_DIR} (box_, violin_, hist_{peak_type}.png)")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()