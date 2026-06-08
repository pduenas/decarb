"""
plot_buildings.py
─────────────────
Generates per-building demand and HVAC plots comparing pre- and post-
decarbonisation simulation outputs.

Outputs (written to each building's …/<UPDATE_FOLDER>/out/ directory):
  plot_demand_<bdg>.png  –  hourly total electricity demand (before vs after)
  plot_hvac_<bdg>.png    –  hourly heating & cooling loads (before vs after)
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ── Settings ───────────────────────────────────────────────────────────────
CASE_DIR        = r"C:\Users\onurt\OneDrive - Massachusetts Institute of Technology\Documents\research\decarb\elec_0_flat_ihg"
CONSUMPTION_DIR = r"Z:\ercot_project\out\consumption_files"
UPDATE_FOLDER   = "update_0"
DEBUG           = False   # True → building 19 only | False → all buildings
# ──────────────────────────────────────────────────────────────────────────



# ── Helpers ────────────────────────────────────────────────────────────────

def sum_every4(series: pd.Series) -> "np.ndarray":
    """Sum consecutive groups of 4 sub-hourly rows into hourly totals."""
    import numpy as np
    arr = series.to_numpy(dtype=float)
    n   = len(arr) // 4
    return arr[: n * 4].reshape(n, 4).sum(axis=1)


def make_xticks(dates: pd.Series) -> tuple[list, list]:
    """
    Return (tick_positions, tick_labels) for the 1st and 15th of every month
    (midnight only), giving ~24 evenly-spaced labels across a full year.
    """
    mask   = dates.dt.day.isin([1, 15]) & (dates.dt.hour == 0) & (dates.dt.minute == 0)
    idx    = dates[mask].index.tolist()
    labels = dates[mask].dt.strftime("%m/%d").tolist()
    return idx, labels


def load_outputs(out_dir: Path) -> dict:
    """Read the optimisation time-series output (ts.csv)."""
    df = pd.read_csv(
        out_dir / "ts.csv",
        parse_dates=["Date"],
        date_format="%m/%d/%Y %H:%M",
    )
    return {
        "dates":   df["Date"],
        "demand":  df["buy"].astype(float).values,
        "heating": (df["hvac_HT"].astype(float).values +
                    df["CHPht"].astype(float).values),
        "cooling": df["hvac_AC"].astype(float).values,
    }


def load_inputs(consumption_dir: Path, bdg: str) -> dict:
    """Read the baseline consumption parquet and resample to hourly."""
    df = pd.read_parquet(consumption_dir / f"{bdg}-0.parquet", engine="fastparquet")

    # Heating = sum of all fuel sources serving heating (skip any missing columns)
    heating_cols = [
        "out.natural_gas.heating.energy_consumption..kwh",
        "out.electricity.heating.energy_consumption..kwh",
        "out.fuel_oil.heating.energy_consumption..kwh",
        "out.propane.heating.energy_consumption..kwh",
    ]
    heating_series = sum(
        sum_every4(df[c].astype(float))
        for c in heating_cols
        if c in df.columns
    )

    return {
        "demand":  sum_every4(df["out.electricity.total.energy_consumption..kwh"].astype(float)),
        "heating": heating_series,
        "cooling": sum_every4(df["out.electricity.cooling.energy_consumption..kwh"].astype(float)),
    }


# ── Summary stats ─────────────────────────────────────────────────────────

def print_summary(dates: pd.Series, inputs: dict, outputs: dict, bdg: str) -> None:
    """
    Print seasonal and annual energy totals (kWh) for demand, heating, and
    cooling — comparing baseline inputs to optimisation outputs.

    Cooling season : May 1 – Sep 30  (month 5 ≤ m < 10)
    Heating season : Oct 1 – Apr 30  (month ≥ 10 or month ≤ 4)
    Annual         : full year
    """
    # Boolean masks aligned to the hourly output index
    m = dates.dt.month.values

    cool_mask = (m >= 5) & (m < 10)          # May 1 – Sep 30
    heat_mask = (m >= 10) | (m <= 4)         # Oct 1 – Apr 30

    def _sum(arr, mask=None):
        return float(arr[mask].sum()) if mask is not None else float(arr.sum())

    # ── cooling
    cc_before = _sum(inputs["cooling"],  cool_mask)
    cc_after  = _sum(outputs["cooling"], cool_mask)
    cc_yr_bef = _sum(inputs["cooling"])
    cc_yr_aft = _sum(outputs["cooling"])

    # ── heating
    hh_before = _sum(inputs["heating"],  heat_mask)
    hh_after  = _sum(outputs["heating"], heat_mask)

    # ── total demand
    td_before = _sum(inputs["demand"])
    td_after  = _sum(outputs["demand"])

    print(f"\n  ── Building {bdg} summary ──────────────────────────────")
    print(f"  Cooling  (May–Sep)   before: {cc_before:>10,.0f} kWh  |  after: {cc_after:>10,.0f} kWh  |  Δ {cc_after - cc_before:>+,.0f} kWh")
    print(f"  Cooling  (annual)    before: {cc_yr_bef:>10,.0f} kWh  |  after: {cc_yr_aft:>10,.0f} kWh  |  Δ {cc_yr_aft - cc_yr_bef:>+,.0f} kWh")
    print(f"  Heating  (Oct–Apr)   before: {hh_before:>10,.0f} kWh  |  after: {hh_after:>10,.0f} kWh  |  Δ {hh_after - hh_before:>+,.0f} kWh")
    print(f"  Demand   (annual)    before: {td_before:>10,.0f} kWh  |  after: {td_after:>10,.0f} kWh  |  Δ {td_after - td_before:>+,.0f} kWh")
    print()


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_demand(x, inp, out, tick_idx, tick_labels, bdg: str, out_dir: Path) -> None:
    """Save total electricity demand comparison plot."""
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(x, inp["demand"], color="black", linestyle="--", linewidth=1.2, label="Before")
    ax.plot(x, out["demand"], color="red",   linestyle="-",  linewidth=1.2, label="After")
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel("Demand (kW)")
    ax.set_title(f"Building {bdg} — Total Demand")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_demand_{bdg}.png", dpi=150)
    plt.close(fig)


def plot_hvac(x, inp, out, tick_idx, tick_labels, bdg: str, out_dir: Path) -> None:
    """Save HVAC heating + cooling comparison plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax1.plot(x, inp["heating"], color="orange", linestyle="--", linewidth=1.2, label="Heating before")
    ax1.plot(x, out["heating"], color="red",    linestyle="-",  linewidth=1.2, label="Heating after")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title("Heating")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2.plot(x, inp["cooling"], color="green", linestyle="--", linewidth=1.2, label="Cooling before")
    ax2.plot(x, out["cooling"], color="blue",  linestyle="-",  linewidth=1.2, label="Cooling after")
    ax2.set_ylabel("Power (kW)")
    ax2.set_title("Cooling")
    ax2.legend(loc="upper right")
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax2.grid(True)

    fig.suptitle(f"Building {bdg} — HVAC Demand")
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_hvac_{bdg}.png", dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def discover_buildings(case_dir: str, update_folder: str, debug: bool) -> list[str]:
    if debug:
        return ["100"]
    return [
        d for d in os.listdir(case_dir)
        if os.path.isdir(os.path.join(case_dir, d, update_folder, "in"))
    ]


def main() -> None:
    buildings = discover_buildings(CASE_DIR, UPDATE_FOLDER, DEBUG)
    print(f"Found {len(buildings)} building(s) to process.\n")

    for bdg in buildings:
        print(f"Plotting building {bdg}...")
        out_dir = Path(CASE_DIR) / bdg / UPDATE_FOLDER / "out"

        try:
            outputs = load_outputs(out_dir)
            inputs  = load_inputs(Path(CONSUMPTION_DIR), bdg)
        except FileNotFoundError as e:
            print(f"  [SKIP] Missing file: {e}")
            continue

        tick_idx, tick_labels = make_xticks(outputs["dates"])
        x = range(len(outputs["demand"]))

        print_summary(outputs["dates"], inputs, outputs, bdg)

        plot_demand(x, inputs, outputs, tick_idx, tick_labels, bdg, out_dir)
        plot_hvac  (x, inputs, outputs, tick_idx, tick_labels, bdg, out_dir)

        print(f"  Saved plots → {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()