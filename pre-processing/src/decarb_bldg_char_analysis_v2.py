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


# ── Configuration ──────────────────────────────────────────
CASE_NAME = "elec_0_tou_cap_results"
ERCOT_ROOT = r"D:\shared\ercot_project"

PEAK_CSV = os.path.join(
    ERCOT_ROOT, "out", "decarb_results", CASE_NAME,
    "per_building_peak_hours_decarb.csv",
)
COEFF_PATH = os.path.join(
    ERCOT_ROOT, "out", "thermal_model", "baseline",
    "regression_coeff_baseline.parquet",
)
UPGRADE_PATH = os.path.join(ERCOT_ROOT, "in", "TX_upgrade0.parquet")
MAPPING_PATH = os.path.join(
    ERCOT_ROOT, "out", "decarb_results", CASE_NAME, "building_mapping.csv"
)
OUTPUT_DIR = os.path.join(
    ERCOT_ROOT, "out", "decarb_results", CASE_NAME, "bldg_char_comp",
)

# ASHP detection — match any row whose heating-efficiency string starts with
# 'ASHP' (case-insensitive). Adjust if you want a stricter rule.
ASHP_COL = "in.hvac_heating_efficiency"
ASHP_PREFIX = "ASHP"

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

    cmap = plt.get_cmap("twilight_shifted")
    rgba = [cmap(h / 24.0) for h in ordered_hours] + [(0.6, 0.6, 0.6, 1.0)]
    hex_colors = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for (r, g, b, _) in rgba
    ]
    return grp, ordered_labels, hex_colors


def _wstats(values, weights):
    """Weighted descriptive stats: mean, median, std, p25, p75, min, max, n."""
    if len(values) == 0:
        out = {k: np.nan for k in
               ["mean", "median", "std", "p25", "p75", "min", "max"]}
        out["n"] = 0
        return out
    w = weights.astype(float)
    v = values.astype(float)
    total_w = w.sum()
    mean = (v * w).sum() / total_w
    var = ((w * (v - mean) ** 2).sum() / total_w)
    std = float(np.sqrt(var))
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted) - 0.5 * w_sorted
    cw /= total_w
    p25 = float(np.interp(0.25, cw, v_sorted))
    median = float(np.interp(0.50, cw, v_sorted))
    p75 = float(np.interp(0.75, cw, v_sorted))
    return {
        "n": int(total_w),
        "mean": float(mean),
        "median": median,
        "std": std,
        "min": float(v.min()),
        "p25": p25,
        "p75": p75,
        "max": float(v.max()),
    }


def summarize(df, group_col, coeffs, ordered_labels, weight_col=None):
    rows = []
    for grp in ordered_labels:
        sub = df[df[group_col] == grp]
        for c in coeffs:
            vals = sub[c].dropna()
            if weight_col is not None:
                w = sub.loc[vals.index, weight_col].astype(float).values
            else:
                w = np.ones(len(vals), dtype=float)
            stats = _wstats(vals.values, w)
            stats["group"] = grp
            stats["coefficient"] = c
            rows.append(stats)
    cols = ["group", "coefficient", "n", "mean", "median",
            "std", "min", "p25", "p75", "max"]
    return pd.DataFrame(rows)[cols]


def _weighted_percentile(values, weights, percentiles):
    """Compute weighted percentiles. `percentiles` in 0..100."""
    if len(values) == 0:
        return np.full(len(percentiles), np.nan)
    order = np.argsort(values)
    v = values[order]
    w = weights[order].astype(float)
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(np.array(percentiles) / 100.0, cw, v)


def _clip_bounds(df, group_col, ordered_labels, coeff,
                 lo_pct=CLIP_LO_PCT, hi_pct=CLIP_HI_PCT, weight_col=None):
    """Compute the [lo, hi] percentile bounds across all groups for a given
    coefficient. Uses weighted percentiles when weight_col is provided."""
    parts_v, parts_w = [], []
    for g in ordered_labels:
        sub = df[df[group_col] == g]
        vals = sub[coeff].dropna()
        parts_v.append(vals.values)
        if weight_col is not None:
            parts_w.append(sub.loc[vals.index, weight_col].astype(float).values)
        else:
            parts_w.append(np.ones(len(vals), dtype=float))
    if not parts_v or sum(len(v) for v in parts_v) == 0:
        return None, None
    all_v = np.concatenate(parts_v)
    all_w = np.concatenate(parts_w)
    lo, hi = _weighted_percentile(all_v, all_w, [lo_pct, hi_pct])
    return float(lo), float(hi)


def _gather(df, group_col, ordered_labels, coeff,
            bounds=None, weight_col=None):
    """Return list of (values, weights) tuples per group, NaNs dropped and
    optionally clipped to bounds=(lo, hi)."""
    out = []
    for g in ordered_labels:
        sub = df[df[group_col] == g]
        vals = sub[coeff].dropna()
        if weight_col is not None:
            w = sub.loc[vals.index, weight_col].astype(float).values
        else:
            w = np.ones(len(vals), dtype=float)
        v = vals.values
        if bounds is not None:
            lo, hi = bounds
            mask = (v >= lo) & (v <= hi)
            v, w = v[mask], w[mask]
        out.append((v, w))
    return out


def _repeat_for_distplot(v, w):
    """Convert weights to integer repeats for tools that don't accept weights
    (boxplot, violinplot). Caps repetition at a sane maximum to keep memory
    bounded; pads via stochastic rounding for fractional weights."""
    if len(v) == 0:
        return np.empty(0)
    # If all weights are 1, short-circuit.
    if np.allclose(w, 1.0):
        return v
    # Scale so total replicated samples is at most ~200k (plenty for stats).
    total = w.sum()
    max_n = 200_000
    if total > max_n:
        scale = max_n / total
        w = w * scale
    # stochastic rounding
    floor = np.floor(w).astype(int)
    frac = w - floor
    rng = np.random.default_rng(0)
    extra = (rng.random(len(w)) < frac).astype(int)
    reps = floor + extra
    return np.repeat(v, reps)


def plot_box(df, group_col, coeffs, ordered_labels, colors, title, save_path,
             weight_col=None):
    fig, axes = plt.subplots(1, len(coeffs), figsize=(5 * len(coeffs), 5.5))
    if len(coeffs) == 1:
        axes = [axes]
    for ax, c in zip(axes, coeffs):
        bounds = _clip_bounds(df, group_col, ordered_labels, c,
                              weight_col=weight_col)
        gathered = _gather(df, group_col, ordered_labels, c,
                           bounds=bounds, weight_col=weight_col)
        # boxplot can't take weights → expand via stochastic-rounded repeats
        data = [_repeat_for_distplot(v, w) for v, w in gathered]
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
    weight_note = " (weighted)" if weight_col else ""
    fig.suptitle(f"{title}{weight_note}  "
                 f"(clipped to {CLIP_LO_PCT}–{CLIP_HI_PCT} pct)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_violin(df, group_col, coeffs, ordered_labels, colors, title, save_path,
                weight_col=None):
    fig, axes = plt.subplots(1, len(coeffs), figsize=(5 * len(coeffs), 5.5))
    if len(coeffs) == 1:
        axes = [axes]
    for ax, c in zip(axes, coeffs):
        bounds = _clip_bounds(df, group_col, ordered_labels, c,
                              weight_col=weight_col)
        gathered = _gather(df, group_col, ordered_labels, c,
                           bounds=bounds, weight_col=weight_col)
        # violinplot also has no weights param → repeat
        data = [_repeat_for_distplot(v, w) for v, w in gathered]
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
    weight_note = " (weighted)" if weight_col else ""
    fig.suptitle(f"{title}{weight_note}  "
                 f"(clipped to {CLIP_LO_PCT}–{CLIP_HI_PCT} pct)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hist(df, group_col, coeffs, ordered_labels, colors, title, save_path,
              weight_col=None):
    """Step-histograms (density) so 6 overlapping groups stay legible."""
    fig, axes = plt.subplots(1, len(coeffs), figsize=(5 * len(coeffs), 5.5))
    if len(coeffs) == 1:
        axes = [axes]
    for ax, c in zip(axes, coeffs):
        bounds = _clip_bounds(df, group_col, ordered_labels, c,
                              weight_col=weight_col)
        gathered = _gather(df, group_col, ordered_labels, c,
                           bounds=bounds, weight_col=weight_col)
        if not any(len(v) for v, _ in gathered):
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        lo, hi = bounds
        if lo == hi:
            all_vals = np.concatenate([v for v, _ in gathered])
            lo, hi = all_vals.min(), all_vals.max()
        bins = np.linspace(lo, hi, HIST_BINS)
        for (v, w), lbl, col in zip(gathered, ordered_labels, colors):
            if len(v) == 0:
                continue
            ax.hist(v, bins=bins, histtype="step", linewidth=2,
                    density=True, color=col, weights=w,
                    label=f"{lbl} (n={int(w.sum()):,})")
        ax.set_title(c)
        ax.set_xlabel(c)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),
                            useMathText=True)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                            useMathText=True)
    weight_note = " (weighted)" if weight_col else ""
    fig.suptitle(f"{title}{weight_note}  "
                 f"(clipped to {CLIP_LO_PCT}–{CLIP_HI_PCT} pct)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_ashp_flags(upgrade_path, ashp_col=ASHP_COL, prefix=ASHP_PREFIX):
    """Return a DataFrame with [bldg_id, is_ashp] from the upgrade parquet."""
    print(f"Loading upgrade parquet: {upgrade_path}")
    up = pd.read_parquet(upgrade_path, columns=["bldg_id", ashp_col])
    print(f"  {len(up):,} rows")
    if ashp_col not in up.columns:
        raise ValueError(f"Column '{ashp_col}' missing from {upgrade_path}")
    up["bldg_id"] = up["bldg_id"].astype(np.int64)
    up["is_ashp"] = (up[ashp_col].astype(str)
                                  .str.strip()
                                  .str.upper()
                                  .str.startswith(prefix.upper()))
    n_ashp = int(up["is_ashp"].sum())
    print(f"  ASHP buildings: {n_ashp:,}  | non-ASHP: {len(up) - n_ashp:,}")
    return up[["bldg_id", "is_ashp"]]


def load_weights(mapping_path):
    """Return a DataFrame [bldg_id, weight] where weight is taken from the
    'count' column of building_mapping.csv (= dwellings each sampled building
    represents). If a new_bldg_id appears in multiple rows, the counts are
    summed."""
    print(f"Loading building mapping: {mapping_path}")
    m = pd.read_csv(mapping_path)
    if "new_bldg_id" not in m.columns:
        raise ValueError(f"'new_bldg_id' column missing from {mapping_path}")
    if "count" not in m.columns:
        raise ValueError(f"'count' column missing from {mapping_path}")
    m = m[["new_bldg_id", "count"]].dropna()
    m["new_bldg_id"] = m["new_bldg_id"].astype(np.int64)
    m["count"] = m["count"].astype(float)
    counts = (m.groupby("new_bldg_id", as_index=False)["count"].sum()
               .rename(columns={"new_bldg_id": "bldg_id", "count": "weight"}))
    print(f"  {len(counts):,} unique buildings | "
          f"total weight = {int(counts['weight'].sum()):,} | "
          f"mean = {counts['weight'].mean():.1f} | "
          f"min = {int(counts['weight'].min())} | "
          f"max = {int(counts['weight'].max())}")
    return counts


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

    # ── ASHP flag merge ────────────────────────────────────────
    ashp_df = load_ashp_flags(UPGRADE_PATH)
    merged = merged.merge(ashp_df, on="bldg_id", how="left")
    n_missing = merged["is_ashp"].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing:,} buildings have no upgrade-file row; "
              f"treating as non-ASHP.")
        merged["is_ashp"] = merged["is_ashp"].fillna(False)
    n_ashp_kept = int(merged["is_ashp"].sum())
    print(f"  After merge: {n_ashp_kept:,} ASHP / "
          f"{len(merged) - n_ashp_kept:,} non-ASHP buildings\n")

    # ── Sample weights from building_mapping.csv ───────────────
    weights_df = load_weights(MAPPING_PATH)
    merged = merged.merge(weights_df, on="bldg_id", how="left")
    n_no_weight = merged["weight"].isna().sum()
    if n_no_weight:
        print(f"  WARNING: {n_no_weight:,} buildings have no weight in mapping; "
              f"defaulting to 1.")
        merged["weight"] = merged["weight"].fillna(1)
    merged["weight"] = merged["weight"].astype(float)
    print(f"  Weighted total dwellings represented: "
          f"{int(merged['weight'].sum()):,}\n")
    WCOL = "weight"

    # ── Global ASHP vs non-ASHP comparison (independent of peak hour) ──
    print(f"\n{'#' * 70}")
    print(f"  ASHP vs non-ASHP — global comparison")
    print(f"{'#' * 70}")
    merged["_ashp_grp"] = np.where(merged["is_ashp"], "ASHP", "non-ASHP")
    ashp_labels = ["ASHP", "non-ASHP"]
    ashp_colors = ["#e74c3c", "#4a90d9"]
    ashp_stats = summarize(merged, "_ashp_grp", COEFFS, ashp_labels,
                           weight_col=WCOL)
    ashp_stats_path = os.path.join(OUTPUT_DIR, "stats_ashp_vs_rest.csv")
    ashp_stats.to_csv(ashp_stats_path, index=False)
    print(f"  Stats → {ashp_stats_path}")
    for c in COEFFS:
        sub = (ashp_stats[ashp_stats["coefficient"] == c]
               .set_index("group").loc[ashp_labels])
        print(f"\n  {c}:")
        print(sub[["n", "mean", "median", "std", "p25", "p75"]]
              .to_string(float_format=lambda x: f"{x:.6g}"))

    ashp_title = f"{CASE_NAME} — ASHP vs non-ASHP (all buildings)"
    plot_box(
        merged, "_ashp_grp", COEFFS, ashp_labels, ashp_colors,
        f"Box plot: {ashp_title}",
        os.path.join(OUTPUT_DIR, "box_ashp_vs_rest.png"),
        weight_col=WCOL,
    )
    plot_violin(
        merged, "_ashp_grp", COEFFS, ashp_labels, ashp_colors,
        f"Violin plot: {ashp_title}",
        os.path.join(OUTPUT_DIR, "violin_ashp_vs_rest.png"),
        weight_col=WCOL,
    )
    plot_hist(
        merged, "_ashp_grp", COEFFS, ashp_labels, ashp_colors,
        f"Histogram: {ashp_title}",
        os.path.join(OUTPUT_DIR, "hist_ashp_vs_rest.png"),
        weight_col=WCOL,
    )
    print(f"  Plots → {OUTPUT_DIR} (box_, violin_, hist_ashp_vs_rest.png)")

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
            mask = (merged["_grp"] == lbl)
            n_raw = int(mask.sum())
            n_w = int(merged.loc[mask, WCOL].sum())
            print(f"    {lbl}: {n_raw:,} buildings  →  {n_w:,} dwellings")

        stats = summarize(merged, "_grp", COEFFS, ordered_labels,
                          weight_col=WCOL)
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
            weight_col=WCOL,
        )
        plot_violin(
            merged, "_grp", COEFFS, ordered_labels, colors,
            f"Violin plot: {title_suffix}",
            os.path.join(OUTPUT_DIR, f"violin_{peak_type}.png"),
            weight_col=WCOL,
        )
        plot_hist(
            merged, "_grp", COEFFS, ordered_labels, colors,
            f"Histogram: {title_suffix}",
            os.path.join(OUTPUT_DIR, f"hist_{peak_type}.png"),
            weight_col=WCOL,
        )
        print(f"  Plots → {OUTPUT_DIR} (box_, violin_, hist_{peak_type}.png)")

        # ── ASHP-only: same per-hour analysis restricted to ASHPs ──────
        ashp_only = merged[merged["is_ashp"]].copy()
        if len(ashp_only) == 0:
            print(f"  [skip ASHP-only] no ASHP buildings present.")
            continue

        ashp_top_hours = top_n_hours(ashp_only[col], TOP_N_BINS)
        ashp_only["_grp"], ashp_ordered_labels, ashp_h_colors = build_groups(
            ashp_only, col, ashp_top_hours
        )

        print(f"\n  --- ASHP-only ({peak_type}) ---")
        print(f"  Top-{TOP_N_BINS} hours among ASHPs: {ashp_top_hours}")
        for lbl in ashp_ordered_labels:
            mask = (ashp_only["_grp"] == lbl)
            n_raw = int(mask.sum())
            n_w = int(ashp_only.loc[mask, WCOL].sum())
            print(f"    {lbl}: {n_raw:,} buildings  →  {n_w:,} dwellings")

        ashp_peak_stats = summarize(
            ashp_only, "_grp", COEFFS, ashp_ordered_labels, weight_col=WCOL,
        )
        ashp_peak_stats_path = os.path.join(
            OUTPUT_DIR, f"stats_{peak_type}_ashp.csv"
        )
        ashp_peak_stats.to_csv(ashp_peak_stats_path, index=False)
        print(f"  Stats → {ashp_peak_stats_path}")

        ashp_title_suffix = (
            f"{CASE_NAME} — {peak_type.title()} peak — "
            f"ASHPs only — per-hour groups (top {TOP_N_BINS}) + rest"
        )
        plot_box(
            ashp_only, "_grp", COEFFS, ashp_ordered_labels, ashp_h_colors,
            f"Box plot: {ashp_title_suffix}",
            os.path.join(OUTPUT_DIR, f"box_{peak_type}_ashp.png"),
            weight_col=WCOL,
        )
        plot_violin(
            ashp_only, "_grp", COEFFS, ashp_ordered_labels, ashp_h_colors,
            f"Violin plot: {ashp_title_suffix}",
            os.path.join(OUTPUT_DIR, f"violin_{peak_type}_ashp.png"),
            weight_col=WCOL,
        )
        plot_hist(
            ashp_only, "_grp", COEFFS, ashp_ordered_labels, ashp_h_colors,
            f"Histogram: {ashp_title_suffix}",
            os.path.join(OUTPUT_DIR, f"hist_{peak_type}_ashp.png"),
            weight_col=WCOL,
        )
        print(f"  ASHP plots → {OUTPUT_DIR} "
              f"(box_, violin_, hist_{peak_type}_ashp.png)")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()