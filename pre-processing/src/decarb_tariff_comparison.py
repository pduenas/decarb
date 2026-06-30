import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# --- Data ---
cases = {
    "Flat":       Path(r"Z:\ercot_project\out\decarb_results\elec_all_flat\summary.csv"),
    "TOU":        Path(r"Z:\ercot_project\out\decarb_results\elec_all_tou\summary.csv"),
    "Flat w/ CC": Path(r"Z:\ercot_project\out\decarb_results\elec_all_flat_cap\summary.csv"),
    "TOU w/ CC": Path(r"Z:\ercot_project\out\decarb_results\elec_all_tou_cap\summary.csv"),
}

COLORS = {
    "Flat":       "#378ADD",
    "TOU":        "#D85A30",
    "Flat w/ CC": "#1D9E75",
    "TOU w/ CC": "#D5C655",
}

def load(path):
    df = pd.read_csv(path)
    df["elec_pct"] = df["electrification_pct"].str.extract(r"(\d+)").astype(int)
    df["peak_gw"]  = df["peak_buy"] / 1000
    df["cost_b"]   = df["total_cost"] / 1e9
    df["peak_date"] = pd.to_datetime(df["peak_buy_datetime"]).dt.strftime("%#d %b")
    return df.sort_values("elec_pct")

dfs = {name: load(p) for name, p in cases.items()}

# --- Figure ---
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

for name, df in dfs.items():
    c = COLORS[name]
    ax1.plot(df["elec_pct"], df["peak_gw"], color=c, linewidth=2,
             marker="o", markersize=5, label=name)
    ax2.plot(df["elec_pct"], df["cost_b"], color=c, linewidth=2,
             marker="o", markersize=5, linestyle="--")

# --- Axes styling ---
for ax in [ax1, ax2]:
    ax.tick_params(labelsize=10)
    ax.spines[["top"]].set_visible(False)

ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

ax1.set_xlabel("Electrification Level",      fontsize=11)
ax1.set_ylabel("System Peak (GW)",           fontsize=11)
ax2.set_ylabel("Consumer Expenditure ($B)",  fontsize=11)

ax1.set_xticks(dfs["Flat"]["elec_pct"])
ax1.set_xticklabels([f"{x}%" for x in dfs["Flat"]["elec_pct"]])
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"${v:.0f}B"))

ax1.grid(axis="y", linestyle="--", alpha=0.4)

# --- Title ---
fig.suptitle(
    "Demand-Side Summary under Heating Electrification in Texas",
    fontsize=14, fontweight="bold",
)

# --- Legend (two separate legends, one per row) ---
case_handles = [
    Line2D([0], [0], color=c, linewidth=2, marker="o", markersize=5, label=name)
    for name, c in COLORS.items()
]

style_handles = [
    Line2D([0], [0], color="#333333", linewidth=2, linestyle="-",  label="System Peak"),
    Line2D([0], [0], color="#333333", linewidth=2, linestyle="--", label="Consumer Expenditure"),
]

leg1 = fig.legend(
    handles=case_handles,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.04),
)
fig.add_artist(leg1)

fig.legend(
    handles=style_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, -0.02),
)

# --- Save ---
fig.tight_layout(rect=[0, 0.13, 1, 0.96])

out = Path(r"Z:\ercot_project\out\decarb_results\case_comparison\system_peak_by_tariff_v3.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, facecolor="white")
print(f"Saved → {out}")
plt.show()