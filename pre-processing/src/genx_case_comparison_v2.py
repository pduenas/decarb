"""
GenX Cost & Emissions Comparison — Flat Tariff Cases
=====================================================
Produces three figures:
  1. Absolute costs: stacked bar + total cost line
  2. Cost changes vs. baseline in $M
  3. Annual emissions: absolute + delta vs. baseline

Folder naming convention:
  elec_060pct_flat_ed_1bus  ->  60% electrification, flat tariff

Usage:
  pip install matplotlib pandas
  python compare_genx_costs.py
"""

import re
import csv
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# ── CONFIG ────────────────────────────────────────────────────────────────────

CASES_DIR     = r"D:\shared\ercot_project\out\genx_cases"
TARIFF_FILTER = "tou_cap"
BASELINE_PCT  = 60
OUTPUT_FILE   = r"D:\shared\ercot_project\out\genx_outputs\genx_costs_tou_cap.png"
RESULTS_CASE  = "results"
# Additional outputs saved automatically:
#   genx_costs_flat_vs_baseline.png
#   genx_costs_flat_emissions.png

COST_COMPONENTS = {
    "cFix":  "Investment / Fixed O&M",
    "cVar":  "Variable O&M",
    "cFuel": "Fuel",
    "cNSE":  "Non-Served Energy",
    "cStart":"Start-up",
}

COLORS = {
    "cFix":  "#4C72B0",
    "cVar":  "#55A868",
    "cFuel": "#C44E52",
    "cNSE":  "#DD8452",
    "cStart":"#8172B2",
}

ELEC_PATTERN = re.compile(r"elec[_\-](\d+)pct", re.IGNORECASE)

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def find_cases(root: str) -> list[Path]:
    return sorted(
        d for d in Path(root).iterdir()
        if d.is_dir() and TARIFF_FILTER.lower() in d.name.lower()
    )


def parse_elec_pct(name: str) -> int | None:
    m = ELEC_PATTERN.search(name)
    return int(m.group(1)) if m else None


def read_costs(case_dir: Path) -> dict[str, float] | None:
    path = case_dir / RESULTS_CASE / "costs.csv"
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


def read_emissions(case_dir: Path) -> float | None:
    """Return total annual emissions (tonnes) from AnnualSum row, Total column."""
    path = case_dir / RESULTS_CASE/ "emissions.csv"
    if not path.exists():
        print(f"  [WARN] missing: {path}")
        return None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Find "Total" column (case-insensitive), fall back to last column
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


def load_data() -> pd.DataFrame:
    records = []
    for case_dir in find_cases(CASES_DIR):
        pct = parse_elec_pct(case_dir.name)
        if pct is None:
            print(f"  [SKIP] can't parse elec% from: {case_dir.name}")
            continue

        costs = read_costs(case_dir)
        if costs is None:
            continue

        emissions = read_emissions(case_dir)

        row = {"elec_pct": pct, "folder": case_dir.name}
        for key in COST_COMPONENTS:
            row[key] = costs.get(key, 0.0) / 1e9          # -> $B
        row["cTotal"]    = costs.get("cTotal", 0.0) / 1e9  # -> $B
        row["emissions"] = (emissions / 1e6) if emissions is not None else None  # -> Mt

        records.append(row)
        costs_path = case_dir / RESULTS_CASE / "costs.csv"
        print(f"  [OK]  {costs_path}")
        print(f"        elec={pct}%  cTotal={costs.get('cTotal', float('nan')):.4e}"
              f"  emissions={emissions}")

    if not records:
        raise RuntimeError(
            f"No '{TARIFF_FILTER}' cases found in:\n  {CASES_DIR}"
        )

    df = pd.DataFrame(records).sort_values("elec_pct").reset_index(drop=True)
    df["x_label"] = df["elec_pct"].astype(str) + "%"
    return df


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _legend_below(fig, ax=None, ncol=6, handles=None, labels=None, **kwargs):
    """Place a legend centered below the entire figure."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        fontsize=8,
        framealpha=0.85,
        **kwargs,
    )


# ── FIGURE 1: Absolute costs ──────────────────────────────────────────────────

def plot_absolute(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.suptitle(
        f"System Costs — {TARIFF_FILTER.title()} Tariff",
        fontsize=14, fontweight="bold",
    )

    x      = range(len(df))
    labels = df["x_label"].tolist()

    # Left: stacked bar
    ax = axes[0]
    bottoms = [0.0] * len(df)
    for key, display in COST_COMPONENTS.items():
        vals = df[key].tolist()
        bars = ax.bar(
            x, vals, bottom=bottoms,
            label=display, color=COLORS[key],
            edgecolor="white", linewidth=0.4, width=0.6,
        )
        for bar, b, v in zip(bars, bottoms, vals):
            if v > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, b + v / 2,
                    f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    for i, total in enumerate(df["cTotal"]):
        ax.text(
            i, bottoms[i] + 0.04, f"${total:.2f}B",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#222",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Electrification Level", fontsize=11)
    ax.set_ylabel("Cost ($B / year)", fontsize=11)
    ax.set_title("Cost Breakdown by Component", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(bottoms) * 1.15)
    # Right: total cost line
    ax2 = axes[1]
    ax2.plot(
        labels, df["cTotal"],
        marker="o", linewidth=2, markersize=7,
        color="#4C72B0", markerfacecolor="white", markeredgewidth=2,
    )
    for lbl, v in zip(labels, df["cTotal"]):
        ax2.annotate(
            f"${v:.2f}B", (lbl, v),
            textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=8, color="#333",
        )
    ax2.set_xlabel("Electrification Level", fontsize=11)
    ax2.set_ylabel("Total Cost ($B / year)", fontsize=11)
    ax2.set_title("Total System Cost", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    _legend_below(fig, ax, ncol=6)
    return fig


# ── FIGURE 2: Delta costs vs baseline ────────────────────────────────────────

def plot_delta(df: pd.DataFrame) -> plt.Figure | None:
    baseline_row = df[df["elec_pct"] == BASELINE_PCT]
    if baseline_row.empty:
        print(f"  [WARN] Baseline ({BASELINE_PCT}%) not found — skipping delta plot.")
        return None

    baseline = baseline_row.iloc[0]
    df_delta = df[df["elec_pct"] != BASELINE_PCT].copy()

    for key in list(COST_COMPONENTS) + ["cTotal"]:
        df_delta[f"d_{key}"] = (df_delta[key] - baseline[key]) * 1000  # $B -> $M

    x2      = range(len(df_delta))
    labels2 = df_delta["x_label"].tolist()

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.suptitle(
        f"System Cost Changes vs. {BASELINE_PCT}% Baseline — {TARIFF_FILTER.title()} Tariff",
        fontsize=14, fontweight="bold",
    )

    # Left: grouped bars per component, delta in $M
    ax3 = axes[0]
    n_comp    = len(COST_COMPONENTS)
    bar_width = 0.7 / n_comp
    offsets   = [(i - n_comp / 2 + 0.5) * bar_width for i in range(n_comp)]

    for offset, (key, display) in zip(offsets, COST_COMPONENTS.items()):
        dvals         = df_delta[f"d_{key}"].tolist()
        bar_positions = [xi + offset for xi in x2]
        bars = ax3.bar(
            bar_positions, dvals,
            width=bar_width * 0.9,
            label=display, color=COLORS[key],
            edgecolor="white", linewidth=0.3,
        )
        for bar, v in zip(bars, dvals):
            if abs(v) > 1:  # skip < $1M labels
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + (5 if v >= 0 else -5),
                    f"{v:+.0f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.5, color="#333",
                )

    ax3.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax3.set_xticks(list(x2))
    ax3.set_xticklabels(labels2, fontsize=10)
    ax3.set_xlabel("Electrification Level", fontsize=11)
    ax3.set_ylabel(f"Delta Cost vs. {BASELINE_PCT}% ($M / year)", fontsize=11)
    ax3.set_title("Change in Cost Components", fontsize=11)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fM"))
    ax3.spines[["top", "right"]].set_visible(False)
    # Right: delta total cost line
    ax4 = axes[1]
    d_totals = df_delta["d_cTotal"].tolist()
    ax4.plot(
        labels2, d_totals,
        marker="o", linewidth=2, markersize=7,
        color="#C44E52", markerfacecolor="white", markeredgewidth=2,
    )
    ax4.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    for lbl, v in zip(labels2, d_totals):
        ax4.annotate(
            f"{v:+.0f}M", (lbl, v),
            textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=8, color="#333",
        )
    ax4.set_xlabel("Electrification Level", fontsize=11)
    ax4.set_ylabel(f"Delta Total Cost vs. {BASELINE_PCT}% ($M / year)", fontsize=11)
    ax4.set_title("Change in Total System Cost", fontsize=11)
    ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fM"))
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    _legend_below(fig, ax3, ncol=6)
    return fig


# ── FIGURE 3: Emissions ───────────────────────────────────────────────────────

def plot_emissions(df: pd.DataFrame) -> plt.Figure | None:
    if df["emissions"].isna().all():
        print("  [WARN] No emissions data found — skipping emissions plot.")
        return None

    labels     = df["x_label"].tolist()
    all_em     = df["emissions"].tolist()
    bar_colors = [
        "#999999" if pct == BASELINE_PCT else "#2ca02c"
        for pct in df["elec_pct"].tolist()
    ]
    em_max = max(v for v in all_em if v is not None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(
        f"Annual Emissions — {TARIFF_FILTER.title()} Tariff",
        fontsize=14, fontweight="bold",
    )

    # Left: absolute emissions
    ax = axes[0]
    ax.bar(labels, all_em, color=bar_colors, edgecolor="white",
           linewidth=0.4, width=0.6)
    for lbl, v in zip(labels, all_em):
        if v is not None:
            ax.text(lbl, v + em_max * 0.01, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xlabel("Electrification Level", fontsize=11)
    ax.set_ylabel("Annual Emissions (Mt CO\u2082)", fontsize=11)
    ax.set_title("Total Annual Emissions", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, em_max * 1.15)


    # Right: delta emissions vs baseline
    baseline_row = df[df["elec_pct"] == BASELINE_PCT]
    ax2 = axes[1]
    if baseline_row.empty:
        ax2.set_visible(False)
    else:
        base_em  = baseline_row.iloc[0]["emissions"]
        df_delta = df[df["elec_pct"] != BASELINE_PCT].copy()
        df_delta["d_em"] = df_delta["emissions"] - base_em
        labels2  = df_delta["x_label"].tolist()
        d_vals   = df_delta["d_em"].tolist()
        d_max    = max(abs(v) for v in d_vals if v is not None)

        ax2.bar(labels2, d_vals, color="#2ca02c", edgecolor="white",
                linewidth=0.4, width=0.6)
        ax2.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        for lbl, v in zip(labels2, d_vals):
            if v is not None:
                ax2.text(lbl, v + (d_max * 0.02 if v >= 0 else -d_max * 0.02),
                         f"{v:+.1f}",
                         ha="center", va="bottom" if v >= 0 else "top",
                         fontsize=8, color="#333")
        ax2.set_xlabel("Electrification Level", fontsize=11)
        ax2.set_ylabel(f"Delta Emissions vs. {BASELINE_PCT}% (Mt CO\u2082)", fontsize=11)
        ax2.set_title("Change in Annual Emissions", fontsize=11)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.legend(
        handles=[Patch(color="#999999", label=f"Baseline ({BASELINE_PCT}%)"),
                 Patch(color="#2ca02c", label="Other cases")],
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=2, fontsize=8, framealpha=0.85,
    )
    return fig


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Scanning: {CASES_DIR}")
    df = load_data()
    print(
        f"\nLoaded {len(df)} cases:\n"
        f"{df[['elec_pct'] + list(COST_COMPONENTS) + ['cTotal', 'emissions']].to_string(index=False)}\n"
    )

    fig1 = plot_absolute(df)
    fig2 = plot_delta(df)
    fig3 = plot_emissions(df)

    if OUTPUT_FILE:
        out = Path(OUTPUT_FILE)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out}")
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