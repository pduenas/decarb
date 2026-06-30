"""
GenX Cost & Emissions Comparison — Three-Tariff Version
========================================================
Compares three tariff types (flat, tou, flat_cap) across all electrification
levels. Each tariff reads from its own results subfolder.

Produces three figures:
  1. Absolute costs:      grouped bars per component + overlaid total-cost lines
  2. Delta vs baseline:   grouped bars per component + overlaid delta-total lines
  3. Emissions:           grouped bars + overlaid lines (absolute + delta)

Folder naming convention examples:
  elec_060pct_flat_ed_1bus       -> flat tariff, 60% electrification
  elec_060pct_tou_ed_1bus        -> TOU tariff
  elec_060pct_flat_cap_ed_1bus   -> flat_cap tariff

Usage:
  pip install matplotlib pandas
  python compare_genx_costs_4tariff.py
"""

import re
import csv
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── CONFIG ────────────────────────────────────────────────────────────────────

CASES_DIR    = r"D:\shared\ercot_project\out\genx_cases"
BASELINE_PCT = 60
OUTPUT_FILE  = r"D:\shared\ercot_project\out\genx_outputs\genx_costs_4tariff.png"

# Each tariff: (display label, regex to match folder, results subfolder, color)
# Order here = order of bars within each electrification group.
TARIFFS = [
    {
        "key":     "flat",
        "label":   "Flat Rate",
        # match "flat" but NOT followed by "_cap"
        "pattern": re.compile(r"(?<![A-Za-z])flat(?!_cap)", re.IGNORECASE),
        "results": "results_1",
        "color":   "#4C72B0",
    },
    {
        "key":     "tou",
        "label":   "Time-of-use Rate",
        "pattern": re.compile(r"(?<![A-Za-z])tou(?![A-Za-z])", re.IGNORECASE),
        "results": "results",
        "color":   "#55A868",
    },
    {
        "key":     "flat_cap",
        "label":   "Flat Rate + Capacity Charge",
        "pattern": re.compile(r"flat_cap", re.IGNORECASE),
        "results": "results",
        "color":   "#C44E52",
    },
    {
        "key":     "tou_cap",
        "label":   "Time-of-use Rate + Capacity Charge",
        "pattern": re.compile(r"tou_cap", re.IGNORECASE),
        "results": "results",
        "color":   "#D5C655",
    },
]

COST_COMPONENTS = {
    "cFix":   "Investment / Fixed O&M",
    "cVar":   "Variable O&M",
    "cFuel":  "Fuel",
}

ELEC_PATTERN = re.compile(r"elec[_\-](\d+)pct", re.IGNORECASE)

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def parse_elec_pct(name: str) -> int | None:
    m = ELEC_PATTERN.search(name)
    return int(m.group(1)) if m else None


def classify_tariff(folder_name: str) -> dict | None:
    """Return the tariff config that matches this folder, or None.

    Order matters: 'flat_cap' is checked first because 'flat' alone would
    otherwise also match it. The flat regex uses a negative lookahead
    `(?!_cap)` to make this robust either way.
    """
    # Check more specific patterns first
    for tariff in sorted(TARIFFS, key=lambda t: -len(t["key"])):
        if tariff["pattern"].search(folder_name):
            return tariff
    return None


def read_costs(case_dir: Path, results_subfolder: str) -> dict[str, float] | None:
    path = case_dir / results_subfolder / "costs.csv"
    if not path.exists():
        print(f"  [WARN] missing: {path}")
        return None
    data = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header "Costs,Total,Zone1,..."
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                try:
                    data[row[0].strip()] = float(row[1])
                except ValueError:
                    data[row[0].strip()] = 0.0
    return data


def read_emissions(case_dir: Path, results_subfolder: str) -> float | None:
    """Return total annual emissions (tonnes) from AnnualSum row, Total column."""
    path = case_dir / results_subfolder / "emissions.csv"
    if not path.exists():
        print(f"  [WARN] missing: {path}")
        return None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            total_col = next(
                i for i, h in enumerate(header) if h.strip().lower() == "total"
            )
        except StopIteration:
            total_col = len(header) - 1
        for row in reader:
            if row and row[0].strip().lower() == "annualsum":
                try:
                    return float(row[total_col])
                except (ValueError, IndexError):
                    return None
    print(f"  [WARN] AnnualSum row not found in {path}")
    return None


def find_cases(root: str) -> list[Path]:
    return sorted(d for d in Path(root).iterdir() if d.is_dir())


def load_data() -> pd.DataFrame:
    records = []
    for case_dir in find_cases(CASES_DIR):
        pct = parse_elec_pct(case_dir.name)
        if pct is None:
            continue

        tariff = classify_tariff(case_dir.name)
        if tariff is None:
            print(f"  [SKIP] no tariff match: {case_dir.name}")
            continue

        costs = read_costs(case_dir, tariff["results"])
        if costs is None:
            continue

        emissions = read_emissions(case_dir, tariff["results"])

        row = {
            "elec_pct":    pct,
            "tariff_key":  tariff["key"],
            "tariff_lbl":  tariff["label"],
            "folder":      case_dir.name,
        }
        for key in COST_COMPONENTS:
            row[key] = costs.get(key, 0.0) / 1e9          # -> $B
        row["cTotal"]    = costs.get("cTotal", 0.0) / 1e9  # -> $B
        row["emissions"] = (emissions / 1e6) if emissions is not None else None  # -> Mt

        records.append(row)
        print(f"  [OK]  {tariff['key']:10s}  elec={pct:3d}%  "
              f"cTotal={costs.get('cTotal', float('nan')):.4e}  "
              f"emissions={emissions}")

    if not records:
        raise RuntimeError(f"No matching cases found in:\n  {CASES_DIR}")

    df = pd.DataFrame(records).sort_values(
        ["elec_pct", "tariff_key"]
    ).reset_index(drop=True)
    df["x_label"] = df["elec_pct"].astype(str) + "%"
    return df


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _legend_below(fig, handles, labels, ncol=6, **kwargs):
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        fontsize=8,
        framealpha=0.85,
        **kwargs,
    )


def _elec_levels(df: pd.DataFrame) -> list[int]:
    return sorted(df["elec_pct"].unique().tolist())


def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """elec_pct (index) x tariff_key (columns) matrix of value_col."""
    return df.pivot(index="elec_pct", columns="tariff_key", values=value_col)


# ── FIGURE 1: Absolute costs ──────────────────────────────────────────────────

def plot_absolute(df: pd.DataFrame) -> plt.Figure:
    """Grouped bars per cost component (3 bars per elec level), plus a
    second panel with overlaid total-cost lines (one per tariff)."""
    elec_levels = _elec_levels(df)
    n_elec      = len(elec_levels)
    n_tariff    = len(TARIFFS)

    n_components = len(COST_COMPONENTS)
    n_rows, n_cols = 2, 2  # fixed 2x2: 3 components + total

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 9))
    fig.suptitle(
        "System Costs by Tariff and Electrification Level",
        fontsize=14, fontweight="bold",
    )
    axes = axes.flatten()

    bar_width = 0.8 / n_tariff
    x_indices = list(range(n_elec))

    # One panel per cost component
    for ax_idx, (key, display) in enumerate(COST_COMPONENTS.items()):
        ax = axes[ax_idx]
        pivoted = _pivot(df, key).reindex(elec_levels)

        for t_idx, tariff in enumerate(TARIFFS):
            offset = (t_idx - n_tariff / 2 + 0.5) * bar_width
            vals = pivoted.get(tariff["key"], pd.Series([float("nan")] * n_elec)).tolist()
            positions = [x + offset for x in x_indices]
            bars = ax.bar(
                positions, vals,
                width=bar_width * 0.9,
                color=tariff["color"],
                label=tariff["label"],
                edgecolor="white", linewidth=0.4,
            )

        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{p}%" for p in elec_levels], fontsize=9)
        ax.set_xlabel("Electrification", fontsize=10)
        ax.set_ylabel("$B / year", fontsize=10)
        ax.set_title(display, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Final panel: overlaid total-cost lines
    ax_tot = axes[n_components]
    total_pivot = _pivot(df, "cTotal").reindex(elec_levels)
    label_dy = [10, -14, 10]  # stagger annotations to reduce overlap
    for t_idx, tariff in enumerate(TARIFFS):
        if tariff["key"] not in total_pivot.columns:
            continue
        vals = total_pivot[tariff["key"]].tolist()
        ax_tot.plot(
            x_indices, vals,
            marker="o", linewidth=2, markersize=7,
            color=tariff["color"],
            markerfacecolor="white", markeredgewidth=2,
            label=tariff["label"],
        )
        for x, v in zip(x_indices, vals):
            if v is not None and not pd.isna(v):
                ax_tot.annotate(
                    f"${v:.2f}B", (x, v),
                    textcoords="offset points",
                    xytext=(0, label_dy[t_idx % len(label_dy)]),
                    ha="center", fontsize=7.5,
                    color=tariff["color"], fontweight="bold",
                )

    ax_tot.set_xticks(x_indices)
    ax_tot.set_xticklabels([f"{p}%" for p in elec_levels], fontsize=9)
    ax_tot.set_xlabel("Electrification", fontsize=10)
    ax_tot.set_ylabel("$B / year", fontsize=10)
    ax_tot.set_title("Total System Cost", fontsize=11, fontweight="bold")
    ax_tot.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))
    ax_tot.spines[["top", "right"]].set_visible(False)
    ax_tot.grid(axis="y", linestyle="--", alpha=0.4)

    # Hide unused subplots
    for j in range(n_components + 1, len(axes)):
        axes[j].set_visible(False)

    # Single shared legend
    handles = [Patch(color=t["color"], label=t["label"]) for t in TARIFFS]
    labels  = [t["label"] for t in TARIFFS]
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    _legend_below(fig, handles, labels, ncol=n_tariff)
    return fig


# ── FIGURE 1b: Total System Cost (standalone) ────────────────────────────────

def plot_total_only(df: pd.DataFrame) -> plt.Figure:
    """Standalone version of the Total System Cost line panel."""
    elec_levels = _elec_levels(df)
    n_elec      = len(elec_levels)
    x_indices   = list(range(n_elec))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle(
        "Total System Cost by Tariff and Electrification Level",
        fontsize=13, fontweight="bold",
    )

    total_pivot = _pivot(df, "cTotal").reindex(elec_levels)
    label_dy = [10, -14, 10]

    for t_idx, tariff in enumerate(TARIFFS):
        if tariff["key"] not in total_pivot.columns:
            continue
        vals = total_pivot[tariff["key"]].tolist()
        ax.plot(
            x_indices, vals,
            marker="o", linewidth=2, markersize=8,
            color=tariff["color"],
            markerfacecolor="white", markeredgewidth=2,
            label=tariff["label"],
        )
        for x, v in zip(x_indices, vals):
            if v is not None and not pd.isna(v):
                ax.annotate(
                    f"${v:.2f}B", (x, v),
                    textcoords="offset points",
                    xytext=(0, label_dy[t_idx % len(label_dy)]),
                    ha="center", fontsize=8,
                    color=tariff["color"], fontweight="bold",
                )

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{p}%" for p in elec_levels], fontsize=10)
    ax.set_xlabel("Electrification Level", fontsize=11)
    ax.set_ylabel("Total Cost ($B / year)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9, framealpha=0.85)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── FIGURE 2: Delta costs vs baseline ────────────────────────────────────────

def plot_delta(df: pd.DataFrame) -> plt.Figure | None:
    """For each tariff, compute deltas relative to that tariff's own baseline
    electrification level, then plot grouped bars per component plus an
    overlaid delta-total line panel."""
    elec_levels = _elec_levels(df)
    if BASELINE_PCT not in elec_levels:
        print(f"  [WARN] Baseline ({BASELINE_PCT}%) not in data — skipping delta plot.")
        return None

    # Build delta table (elec_pct, tariff_key, d_<comp>, d_cTotal) excluding baseline level
    delta_records = []
    for tariff in TARIFFS:
        sub = df[df["tariff_key"] == tariff["key"]]
        base = sub[sub["elec_pct"] == BASELINE_PCT]
        if base.empty:
            print(f"  [WARN] No baseline row for tariff '{tariff['key']}' — skipped.")
            continue
        base = base.iloc[0]
        for _, r in sub[sub["elec_pct"] != BASELINE_PCT].iterrows():
            row = {"elec_pct": r["elec_pct"], "tariff_key": tariff["key"]}
            for k in COST_COMPONENTS:
                row[f"d_{k}"] = (r[k] - base[k]) * 1000  # $B -> $M
            row["d_cTotal"] = (r["cTotal"] - base["cTotal"]) * 1000
            delta_records.append(row)

    if not delta_records:
        return None

    ddf = pd.DataFrame(delta_records)
    delta_levels = sorted(ddf["elec_pct"].unique().tolist())
    n_elec       = len(delta_levels)
    n_tariff     = len(TARIFFS)
    bar_width    = 0.8 / n_tariff
    x_indices    = list(range(n_elec))

    n_components = len(COST_COMPONENTS)
    n_rows, n_cols = 2, 2  # fixed 2x2: 3 components + total

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 9))
    fig.suptitle(
        f"Cost Changes vs. {BASELINE_PCT}% Baseline (per tariff)",
        fontsize=14, fontweight="bold",
    )
    axes = axes.flatten()

    for ax_idx, (key, display) in enumerate(COST_COMPONENTS.items()):
        ax = axes[ax_idx]
        pivoted = ddf.pivot(index="elec_pct", columns="tariff_key",
                            values=f"d_{key}").reindex(delta_levels)

        for t_idx, tariff in enumerate(TARIFFS):
            offset = (t_idx - n_tariff / 2 + 0.5) * bar_width
            vals = pivoted.get(tariff["key"], pd.Series([float("nan")] * n_elec)).tolist()
            positions = [x + offset for x in x_indices]
            bars = ax.bar(
                positions, vals,
                width=bar_width * 0.9,
                color=tariff["color"],
                label=tariff["label"],
                edgecolor="white", linewidth=0.3,
            )

        ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{p}%" for p in delta_levels], fontsize=9)
        ax.set_xlabel("Electrification", fontsize=10)
        ax.set_ylabel("$M / year", fontsize=10)
        ax.set_title(f"Δ {display}", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fM"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Final panel: overlaid delta-total lines
    ax_tot = axes[n_components]
    tot_pivot = ddf.pivot(index="elec_pct", columns="tariff_key",
                          values="d_cTotal").reindex(delta_levels)
    label_dy = [10, -14, 10]
    for t_idx, tariff in enumerate(TARIFFS):
        if tariff["key"] not in tot_pivot.columns:
            continue
        vals = tot_pivot[tariff["key"]].tolist()
        ax_tot.plot(
            x_indices, vals,
            marker="o", linewidth=2, markersize=7,
            color=tariff["color"],
            markerfacecolor="white", markeredgewidth=2,
            label=tariff["label"],
        )
        for x, v in zip(x_indices, vals):
            if v is not None and not pd.isna(v):
                ax_tot.annotate(
                    f"{v:+.0f}M", (x, v),
                    textcoords="offset points",
                    xytext=(0, label_dy[t_idx % len(label_dy)]),
                    ha="center", fontsize=7.5,
                    color=tariff["color"], fontweight="bold",
                )

    ax_tot.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax_tot.set_xticks(x_indices)
    ax_tot.set_xticklabels([f"{p}%" for p in delta_levels], fontsize=9)
    ax_tot.set_xlabel("Electrification", fontsize=10)
    ax_tot.set_ylabel("$M / year", fontsize=10)
    ax_tot.set_title(f"Δ Total Cost vs. {BASELINE_PCT}%", fontsize=11, fontweight="bold")
    ax_tot.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fM"))
    ax_tot.spines[["top", "right"]].set_visible(False)
    ax_tot.grid(axis="y", linestyle="--", alpha=0.4)

    for j in range(n_components + 1, len(axes)):
        axes[j].set_visible(False)

    handles = [Patch(color=t["color"], label=t["label"]) for t in TARIFFS]
    labels  = [t["label"] for t in TARIFFS]
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    _legend_below(fig, handles, labels, ncol=n_tariff)
    return fig


# ── FIGURE 3: Emissions ───────────────────────────────────────────────────────

def plot_emissions(df: pd.DataFrame) -> plt.Figure | None:
    if df["emissions"].isna().all():
        print("  [WARN] No emissions data found — skipping emissions plot.")
        return None

    elec_levels = _elec_levels(df)
    n_elec      = len(elec_levels)
    n_tariff    = len(TARIFFS)
    bar_width   = 0.8 / n_tariff
    x_indices   = list(range(n_elec))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Annual Emissions by Tariff and Electrification Level",
        fontsize=14, fontweight="bold",
    )

    # Top-left: absolute emissions, grouped bars
    ax_abs = axes[0, 0]
    em_pivot = _pivot(df, "emissions").reindex(elec_levels)
    for t_idx, tariff in enumerate(TARIFFS):
        offset = (t_idx - n_tariff / 2 + 0.5) * bar_width
        vals = em_pivot.get(tariff["key"], pd.Series([float("nan")] * n_elec)).tolist()
        positions = [x + offset for x in x_indices]
        bars = ax_abs.bar(
            positions, vals,
            width=bar_width * 0.9,
            color=tariff["color"],
            label=tariff["label"],
            edgecolor="white", linewidth=0.4,
        )

    ax_abs.set_xticks(x_indices)
    ax_abs.set_xticklabels([f"{p}%" for p in elec_levels], fontsize=9)
    ax_abs.set_xlabel("Electrification", fontsize=10)
    ax_abs.set_ylabel("Mt CO₂ / year", fontsize=10)
    ax_abs.set_title("Total Annual Emissions (bars)", fontsize=11)
    ax_abs.spines[["top", "right"]].set_visible(False)
    ax_abs.grid(axis="y", linestyle="--", alpha=0.4)

    # Top-right: absolute emissions, overlaid lines
    ax_abs_l = axes[0, 1]
    label_dy = [10, -14, 10]
    for t_idx, tariff in enumerate(TARIFFS):
        if tariff["key"] not in em_pivot.columns:
            continue
        vals = em_pivot[tariff["key"]].tolist()
        ax_abs_l.plot(
            x_indices, vals,
            marker="o", linewidth=2, markersize=7,
            color=tariff["color"],
            markerfacecolor="white", markeredgewidth=2,
            label=tariff["label"],
        )
        for x, v in zip(x_indices, vals):
            if v is not None and not pd.isna(v):
                ax_abs_l.annotate(
                    f"{v:.1f}", (x, v),
                    textcoords="offset points",
                    xytext=(0, label_dy[t_idx % len(label_dy)]),
                    ha="center", fontsize=7.5,
                    color=tariff["color"], fontweight="bold",
                )
    ax_abs_l.set_xticks(x_indices)
    ax_abs_l.set_xticklabels([f"{p}%" for p in elec_levels], fontsize=9)
    ax_abs_l.set_xlabel("Electrification", fontsize=10)
    ax_abs_l.set_ylabel("Mt CO₂ / year", fontsize=10)
    ax_abs_l.set_title("Total Annual Emissions (lines)", fontsize=11)
    ax_abs_l.spines[["top", "right"]].set_visible(False)
    ax_abs_l.grid(axis="y", linestyle="--", alpha=0.4)

    # Bottom row: deltas vs baseline (per tariff)
    if BASELINE_PCT not in elec_levels:
        for c in (0, 1):
            axes[1, c].set_visible(False)
    else:
        delta_records = []
        for tariff in TARIFFS:
            sub = df[df["tariff_key"] == tariff["key"]]
            base = sub[sub["elec_pct"] == BASELINE_PCT]
            if base.empty:
                continue
            base_em = base.iloc[0]["emissions"]
            if base_em is None or pd.isna(base_em):
                continue
            for _, r in sub[sub["elec_pct"] != BASELINE_PCT].iterrows():
                if r["emissions"] is None or pd.isna(r["emissions"]):
                    continue
                delta_records.append({
                    "elec_pct":   r["elec_pct"],
                    "tariff_key": tariff["key"],
                    "d_em":       r["emissions"] - base_em,
                })

        if not delta_records:
            for c in (0, 1):
                axes[1, c].set_visible(False)
        else:
            ddf = pd.DataFrame(delta_records)
            delta_levels = sorted(ddf["elec_pct"].unique().tolist())
            x_d = list(range(len(delta_levels)))
            d_pivot = ddf.pivot(index="elec_pct", columns="tariff_key",
                                values="d_em").reindex(delta_levels)

            # Bottom-left: delta bars
            ax_d = axes[1, 0]
            for t_idx, tariff in enumerate(TARIFFS):
                offset = (t_idx - n_tariff / 2 + 0.5) * bar_width
                vals = d_pivot.get(tariff["key"], pd.Series([float("nan")] * len(delta_levels))).tolist()
                positions = [x + offset for x in x_d]
                bars = ax_d.bar(
                    positions, vals,
                    width=bar_width * 0.9,
                    color=tariff["color"],
                    label=tariff["label"],
                    edgecolor="white", linewidth=0.4,
                )
            ax_d.axhline(0, color="#555", linewidth=0.8, linestyle="--")
            ax_d.set_xticks(x_d)
            ax_d.set_xticklabels([f"{p}%" for p in delta_levels], fontsize=9)
            ax_d.set_xlabel("Electrification", fontsize=10)
            ax_d.set_ylabel("Mt CO₂ / year", fontsize=10)
            ax_d.set_title(f"Δ Emissions vs. {BASELINE_PCT}% (bars)", fontsize=11)
            ax_d.spines[["top", "right"]].set_visible(False)
            ax_d.grid(axis="y", linestyle="--", alpha=0.4)

            # Bottom-right: delta lines
            ax_d_l = axes[1, 1]
            label_dy = [10, -14, 10]
            for t_idx, tariff in enumerate(TARIFFS):
                if tariff["key"] not in d_pivot.columns:
                    continue
                vals = d_pivot[tariff["key"]].tolist()
                ax_d_l.plot(
                    x_d, vals,
                    marker="o", linewidth=2, markersize=7,
                    color=tariff["color"],
                    markerfacecolor="white", markeredgewidth=2,
                    label=tariff["label"],
                )
                for x, v in zip(x_d, vals):
                    if v is not None and not pd.isna(v):
                        ax_d_l.annotate(
                            f"{v:+.1f}", (x, v),
                            textcoords="offset points",
                            xytext=(0, label_dy[t_idx % len(label_dy)]),
                            ha="center", fontsize=7.5,
                            color=tariff["color"], fontweight="bold",
                        )
            ax_d_l.axhline(0, color="#555", linewidth=0.8, linestyle="--")
            ax_d_l.set_xticks(x_d)
            ax_d_l.set_xticklabels([f"{p}%" for p in delta_levels], fontsize=9)
            ax_d_l.set_xlabel("Electrification", fontsize=10)
            ax_d_l.set_ylabel("Mt CO₂ / year", fontsize=10)
            ax_d_l.set_title(f"Δ Emissions vs. {BASELINE_PCT}% (lines)", fontsize=11)
            ax_d_l.spines[["top", "right"]].set_visible(False)
            ax_d_l.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [Patch(color=t["color"], label=t["label"]) for t in TARIFFS]
    labels  = [t["label"] for t in TARIFFS]
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    _legend_below(fig, handles, labels, ncol=n_tariff)
    return fig


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Scanning: {CASES_DIR}")
    df = load_data()

    print("\nLoaded summary:")
    summary_cols = ["elec_pct", "tariff_key"] + list(COST_COMPONENTS) + ["cTotal", "emissions"]
    print(df[summary_cols].to_string(index=False))
    print()

    fig1  = plot_absolute(df)
    fig1b = plot_total_only(df)
    fig2  = plot_delta(df)
    fig3  = plot_emissions(df)

    if OUTPUT_FILE:
        out = Path(OUTPUT_FILE)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out}")
        if fig1b is not None:
            p = out.with_stem(out.stem + "_total")
            fig1b.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved -> {p}")
        if fig2 is not None:
            p = out.with_stem(out.stem + "_vs_baseline")
            fig2.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved -> {p}")
        if fig3 is not None:
            p = out.with_stem(out.stem + "_emissions")
            fig3.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved -> {p}")
    else:
        plt.show()