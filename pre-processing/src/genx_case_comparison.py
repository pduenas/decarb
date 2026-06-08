"""
GenX Cost Comparison — Flat Tariff Cases
==========================================
Scans CASES_DIR for folders matching TARIFF_FILTER, reads results/costs.csv,
parses electrification % from folder names, and produces comparison plots.

Folder naming convention:
  elec_060pct_flat_ed_1bus  →  60% electrification, flat tariff

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

# ── CONFIG ────────────────────────────────────────────────────────────────────

CASES_DIR     = r"D:\shared\ercot_project\out\genx_cases"
TARIFF_FILTER = "flat"          # substring match on folder name (case-insensitive)
OUTPUT_FILE   = r"D:\shared\ercot_project\out\genx_outputs\genx_costs_flat.png"  # set None to plt.show() instead

# Cost rows in costs.csv to include (order = stack order in bar chart)
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
        [d for d in Path(root).iterdir()
         if d.is_dir() and TARIFF_FILTER.lower() in d.name.lower()]
    )


def parse_elec_pct(name: str) -> int | None:
    m = ELEC_PATTERN.search(name)
    return int(m.group(1)) if m else None


def read_costs(case_dir: Path) -> dict[str, float] | None:
    path = case_dir / "results" / "costs.csv"
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
        row = {"elec_pct": pct, "folder": case_dir.name}
        for key in COST_COMPONENTS:
            row[key] = costs.get(key, 0.0) / 1e9   # convert to $B
        row["cTotal"] = costs.get("cTotal", 0.0) / 1e9
        records.append(row)
        print(f"  [OK]  {case_dir.name}  ({pct}%)")

    if not records:
        raise RuntimeError(
            f"No '{TARIFF_FILTER}' cases found in:\n  {CASES_DIR}"
        )

    df = pd.DataFrame(records).sort_values("elec_pct").reset_index(drop=True)
    df["x_label"] = df["elec_pct"].astype(str) + "%"
    return df


# ── PLOTTING ──────────────────────────────────────────────────────────────────

BASELINE_PCT = 60   # electrification % to use as baseline


def plot(df: pd.DataFrame):
    # ── Figure 1: absolute costs ──────────────────────────────────────────────
    fig1, axes = plt.subplots(
        1, 2,
        figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig1.suptitle(
        f"GenX System Costs — {TARIFF_FILTER.title()} Tariff",
        fontsize=14, fontweight="bold", y=1.01,
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
            label=display,
            color=COLORS[key],
            edgecolor="white", linewidth=0.4,
            width=0.6,
        )
        for bar, b, v in zip(bars, bottoms, vals):
            if v > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + v / 2,
                    f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    for i, total in enumerate(df["cTotal"]):
        ax.text(
            i, bottoms[i] + 0.04,
            f"${total:.2f}B",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#222",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Electrification Level", fontsize=11)
    ax.set_ylabel("Cost ($B / year)", fontsize=11)
    ax.set_title("Cost Breakdown by Component", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(bottoms) * 1.12)

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

    fig1.tight_layout()

    # ── Figure 2: changes vs baseline ────────────────────────────────────────
    baseline_row = df[df["elec_pct"] == BASELINE_PCT]
    if baseline_row.empty:
        print(f"  [WARN] Baseline ({BASELINE_PCT}%) not found — skipping delta plot.")
        fig2 = None
    else:
        baseline = baseline_row.iloc[0]
        # Only non-baseline cases on x-axis
        df_delta = df[df["elec_pct"] != BASELINE_PCT].copy()
        for key in list(COST_COMPONENTS) + ["cTotal"]:
            df_delta[f"d_{key}"] = df_delta[key] - baseline[key]

        x2     = range(len(df_delta))
        labels2 = df_delta["x_label"].tolist()

        fig2, axes2 = plt.subplots(
            1, 2,
            figsize=(13, 5.5),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        fig2.suptitle(
            f"GenX Cost Changes vs. {BASELINE_PCT}% Baseline — {TARIFF_FILTER.title()} Tariff",
            fontsize=14, fontweight="bold", y=1.01,
        )

        # Left: grouped bars, one per component, showing delta
        ax3 = axes2[0]
        n_components = len(COST_COMPONENTS)
        bar_width    = 0.7 / n_components
        offsets      = [(i - n_components / 2 + 0.5) * bar_width
                        for i in range(n_components)]

        for offset, (key, display) in zip(offsets, COST_COMPONENTS.items()):
            dvals = df_delta[f"d_{key}"].tolist()
            bar_positions = [xi + offset for xi in x2]
            bars = ax3.bar(
                bar_positions, dvals,
                width=bar_width * 0.9,
                label=display,
                color=COLORS[key],
                edgecolor="white", linewidth=0.3,
            )
            for bar, v in zip(bars, dvals):
                if abs(v) > 0.02:
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + (0.015 if v >= 0 else -0.015),
                        f"{v:+.2f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=6.5, color="#333",
                    )

        ax3.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        ax3.set_xticks(list(x2))
        ax3.set_xticklabels(labels2, fontsize=10)
        ax3.set_xlabel("Electrification Level", fontsize=11)
        ax3.set_ylabel(f"ΔCost vs. {BASELINE_PCT}% ($B / year)", fontsize=11)
        ax3.set_title("Change in Cost Components", fontsize=11)
        ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2fB"))
        ax3.legend(loc="upper left", fontsize=8, framealpha=0.85)
        ax3.spines[["top", "right"]].set_visible(False)

        # Right: delta total cost line
        ax4 = axes2[1]
        d_totals = df_delta["d_cTotal"].tolist()
        ax4.plot(
            labels2, d_totals,
            marker="o", linewidth=2, markersize=7,
            color="#C44E52", markerfacecolor="white", markeredgewidth=2,
        )
        ax4.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        for lbl, v in zip(labels2, d_totals):
            ax4.annotate(
                f"{v:+.2f}B", (lbl, v),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color="#333",
            )

        ax4.set_xlabel("Electrification Level", fontsize=11)
        ax4.set_ylabel(f"ΔTotal Cost vs. {BASELINE_PCT}% ($B / year)", fontsize=11)
        ax4.set_title("Change in Total System Cost", fontsize=11)
        ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2fB"))
        ax4.spines[["top", "right"]].set_visible(False)
        ax4.grid(axis="y", linestyle="--", alpha=0.4)

        fig2.tight_layout()

    # ── Save / show ───────────────────────────────────────────────────────────
    if OUTPUT_FILE:
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
        print(f"Saved → {OUTPUT_FILE}")
        if fig2 is not None:
            delta_path = Path(OUTPUT_FILE).with_stem(
                Path(OUTPUT_FILE).stem + "_vs_baseline"
            )
            fig2.savefig(delta_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {delta_path}")
    else:
        plt.show()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Scanning: {CASES_DIR}")
    df = load_data()
    print(f"\nLoaded {len(df)} cases:\n{df[['elec_pct'] + list(COST_COMPONENTS) + ['cTotal']].to_string(index=False)}\n")
    plot(df)