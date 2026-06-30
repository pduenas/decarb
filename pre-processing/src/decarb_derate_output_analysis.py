# """
# Aggregate building timeseries across ERCOT buildings for ONE OR MORE cases,
# then compare the cases against each other and against the ResStock baseline.

# NO scaling, NO counts, NO nearest-neighbor mapping:
#     - For each case, simply SUM every ts.csv that exists in the case dir.
#     - The ResStock baseline is summed once over the same set of buildings
#       (it is case-independent) and used as a common comparison point.

# Units: ts.csv buy is kW -> GW; HVAC/CHP and ResStock loads are kWh -> GWh.
# """

# from pathlib import Path
# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")  # non-interactive backend: writes PNGs, no GUI/thread issues
# import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor


# # =============================================================================
# # Building discovery
# # =============================================================================
# def find_case_bldg_ids(case_dir, upgrade):
#     """Return the list of bldg_ids that actually have a ts.csv in this case."""
#     bldg_ids = []
#     if not case_dir.exists():
#         print(f"  WARNING: case dir does not exist: {case_dir}")
#         return bldg_ids
#     for child in case_dir.iterdir():
#         if not child.is_dir():
#             continue
#         ts_path = child / upgrade / "out" / "ts.csv"
#         if ts_path.exists():
#             try:
#                 bldg_ids.append(int(child.name))
#             except ValueError:
#                 continue
#     bldg_ids.sort()
#     return bldg_ids


# # =============================================================================
# # ts.csv aggregation (the decarb case)
# # =============================================================================
# def _load_single_ts(args):
#     """Load a single building's ts.csv. No scaling."""
#     bldg_id, case_dir, upgrade, ts_columns = args
#     ts_file = case_dir / str(int(bldg_id)) / upgrade / "out" / "ts.csv"
#     try:
#         ts_data = pd.read_csv(ts_file, usecols=ts_columns)
#     except pd.errors.EmptyDataError:
#         print(f"WARNING: Empty file skipped: {ts_file}")
#         ts_data = pd.DataFrame(0, index=range(8760), columns=ts_columns)
#     buy = ts_data["buy"].values
#     hvac_ht = ts_data["HVACht"].values
#     hvac_ac = ts_data["HVACac"].values
#     chp_ht = ts_data["CHPht"].values
#     return np.column_stack([buy, hvac_ht, hvac_ac, chp_ht])


# def aggregate_timeseries(bldg_ids, case_dir, upgrade, ts_columns, output_columns,
#                          use_threading=False, n_workers=20):
#     """
#     Sum the ts.csv of every building in bldg_ids.
#     Combines HVACht + CHPht into heating. Converts kW -> GW.
#     """
#     total = len(bldg_ids)
#     args_list = [(bid, case_dir, upgrade, ts_columns) for bid in bldg_ids]

#     aggregated = None

#     def _accumulate(iterable):
#         nonlocal aggregated
#         for i, stacked in enumerate(iterable):
#             if aggregated is None:
#                 aggregated = stacked.astype(np.float64)
#             else:
#                 aggregated += stacked
#             if (i + 1) % 2000 == 0:
#                 print(f"  Processed {i + 1}/{total}...")

#     if use_threading:
#         print(f"  Summing ts.csv for {total} buildings with {n_workers} workers...")
#         with ProcessPoolExecutor(max_workers=n_workers) as executor:
#             _accumulate(executor.map(_load_single_ts, args_list))
#     else:
#         print(f"  Summing ts.csv for {total} buildings (sequential)...")
#         _accumulate(_load_single_ts(a) for a in args_list)

#     result = pd.DataFrame({
#         output_columns[0]: aggregated[:, 0] / 1_000_000,                       # buy: kW -> GW
#         output_columns[1]: (aggregated[:, 1] + aggregated[:, 3]) / 1_000_000,  # HVACht + CHPht -> GWh
#         output_columns[2]: aggregated[:, 2] / 1_000_000,                       # HVACac -> GWh
#     })
#     return result


# # =============================================================================
# # ResStock baseline aggregation
# # =============================================================================
# KBTU_TO_KWH = 0.293071


# def _load_single_resstock(args):
#     """Load a single ResStock building's parquet. No scaling."""
#     bldg_id, resstock_dir = args
#     pq_file = resstock_dir / f"{int(bldg_id)}-0.parquet"
#     df = pd.read_parquet(pq_file, engine="fastparquet")
#     elec = df["out.electricity.total.energy_consumption..kwh"].fillna(0).values.astype(np.float64)
#     heat = df["out.load.heating.energy_delivered..kbtu"].fillna(0).values.astype(np.float64)
#     cool = df["out.load.cooling.energy_delivered..kbtu"].fillna(0).values.astype(np.float64)
#     # Downsample 15-min -> hourly by summing every 4 rows
#     elec_h = elec.reshape(-1, 4).sum(axis=1)
#     heat_h = heat.reshape(-1, 4).sum(axis=1) * KBTU_TO_KWH
#     cool_h = cool.reshape(-1, 4).sum(axis=1) * KBTU_TO_KWH
#     return np.column_stack([elec_h, heat_h, cool_h])


# def aggregate_resstock(bldg_ids, resstock_dir, output_columns,
#                        use_threading=False, n_workers=20):
#     """Sum the ResStock parquet of every building in bldg_ids. No scaling."""
#     total = len(bldg_ids)
#     args_list = [(bid, resstock_dir) for bid in bldg_ids]

#     aggregated = None

#     def _accumulate(iterable):
#         nonlocal aggregated
#         for i, stacked in enumerate(iterable):
#             if aggregated is None:
#                 aggregated = stacked.astype(np.float64)
#             else:
#                 aggregated += stacked
#             if (i + 1) % 2000 == 0:
#                 print(f"  Processed {i + 1}/{total}...")

#     if use_threading:
#         print(f"  Summing ResStock for {total} buildings with {n_workers} workers...")
#         with ProcessPoolExecutor(max_workers=n_workers) as executor:
#             _accumulate(executor.map(_load_single_resstock, args_list))
#     else:
#         print(f"  Summing ResStock for {total} buildings (sequential)...")
#         _accumulate(_load_single_resstock(a) for a in args_list)

#     result = pd.DataFrame({
#         output_columns[0]: aggregated[:, 0] / 1_000_000,  # kWh (hourly) = kW -> GW
#         output_columns[1]: aggregated[:, 1] / 1_000_000,  # kWh -> GWh
#         output_columns[2]: aggregated[:, 2] / 1_000_000,  # kWh -> GWh
#     })
#     return result


# # =============================================================================
# # Plotting
# # =============================================================================
# def plot_results(aggregated, output_columns, label, plot_path):
#     """Plot the three timeseries for a single dataset in three subplots."""
#     fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
#     fig.suptitle(f"Aggregated Building Timeseries — {label}", fontsize=14)
#     colors = ["#1f77b4", "#d62728", "#2ca02c"]

#     for ax, col, color in zip(axes, output_columns, colors):
#         ax.plot(aggregated[col], linewidth=0.5, color=color)
#         ax.set_ylabel("GW" if "GW]" in col else "GWh")
#         ax.set_title(col)
#         ax.grid(True, alpha=0.3)

#     axes[-1].set_xlabel("Hour")
#     fig.tight_layout()
#     fig.savefig(plot_path, dpi=150)
#     plt.close(fig)
#     print(f"  Plot saved to: {plot_path}")


# def plot_multi_case_comparison(case_results, resstock_agg, output_columns, plot_path):
#     """
#     Overlay all cases (and the ResStock baseline) on the same three subplots.
#     case_results: dict {case_name: aggregated_df}
#     """
#     fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
#     fig.suptitle("Case Comparison vs ResStock Baseline", fontsize=14)

#     # Distinct colors for cases; baseline always gray.
#     cmap = plt.get_cmap("tab10")

#     for ax, col in zip(axes, output_columns):
#         if resstock_agg is not None:
#             ax.plot(resstock_agg[col], linewidth=0.6, color="gray",
#                     alpha=0.7, label="ResStock Baseline")
#         for idx, (case_name, agg) in enumerate(case_results.items()):
#             ax.plot(agg[col], linewidth=0.5, color=cmap(idx % 10), label=case_name)
#         ax.set_ylabel("GW" if "GW]" in col else "GWh")
#         ax.set_title(col)
#         ax.grid(True, alpha=0.3)
#         ax.legend(loc="upper right", fontsize=8)

#     axes[-1].set_xlabel("Hour")
#     fig.tight_layout()
#     fig.savefig(plot_path, dpi=150)
#     plt.close(fig)
#     print(f"\nComparison plot saved to: {plot_path}")


# # =============================================================================
# # Per-case driver
# # =============================================================================
# def process_case(case_name, ercot_root, upgrade, ts_columns, output_columns,
#                  use_threading, n_workers):
#     """Run the full ts.csv aggregation for one case and save its outputs."""
#     print("\n" + "=" * 70)
#     print(f"CASE: {case_name}")
#     print("=" * 70)

#     case_dir = ercot_root / "in" / "decarb_inputs" / case_name
#     results_dir = ercot_root / "out" / "decarb_results" / case_name
#     results_dir.mkdir(parents=True, exist_ok=True)

#     bldg_ids = find_case_bldg_ids(case_dir, upgrade)
#     print(f"  Buildings with ts.csv: {len(bldg_ids)}")
#     if not bldg_ids:
#         print("  No buildings found — skipping case.")
#         return None, []

#     aggregated = aggregate_timeseries(
#         bldg_ids, case_dir, upgrade, ts_columns, output_columns,
#         use_threading=use_threading, n_workers=n_workers,
#     )

#     output_path = results_dir / f"{case_name}_aggregated_ts.csv"
#     plot_path = results_dir / f"{case_name}_aggregated_ts.png"
#     aggregated.to_csv(output_path, index=False)
#     print(f"  Output saved to: {output_path}  (shape {aggregated.shape})")
#     plot_results(aggregated, output_columns, case_name, plot_path)

#     return aggregated, bldg_ids


# # =============================================================================
# # MAIN
# # =============================================================================
# if __name__ == "__main__":

#     # ---- Config -------------------------------------------------------------
#     # List one or more cases to run and compare.
#     case_names = [
#         "elec_0_flat_derate_10",
#         "elec_0_flat_derate_20",
#         "elec_0_flat_derate_30",
#         "elec_0_flat_derate_40",
#         "elec_0_flat_derate_50",
#     ]
#     upgrade = "update_0"
#     use_threading = True
#     n_workers = 20
#     include_resstock = True  # set False to skip the baseline entirely

#     # ---- Paths --------------------------------------------------------------
#     script_path = Path(__file__).resolve()
#     ercot_root = script_path.parents[1]
#     resstock_dir = ercot_root / "out" / "consumption_files"
#     comparison_dir = ercot_root / "in" / "decarb_inputs" / "derate_comp"
#     comparison_dir.mkdir(parents=True, exist_ok=True)

#     # ---- Columns ------------------------------------------------------------
#     ts_columns = ["buy", "HVACht", "HVACac", "CHPht"]
#     output_columns = [
#         "Grid Purchases [GW]",
#         "Heating Generated [GWh]",
#         "Cooling Generated [GWh]",
#     ]

#     # ---- Run each case ------------------------------------------------------
#     case_results = {}      # {case_name: aggregated_df}
#     all_bldg_ids = set()   # union of buildings seen, for the ResStock baseline
#     for case_name in case_names:
#         agg, bldg_ids = process_case(
#             case_name, ercot_root, upgrade, ts_columns, output_columns,
#             use_threading, n_workers,
#         )
#         if agg is not None:
#             case_results[case_name] = agg
#             all_bldg_ids.update(bldg_ids)

#     if not case_results:
#         raise SystemExit("No cases produced results — nothing to compare.")

#     # ---- ResStock baseline (summed once over all buildings seen) ------------
#     resstock_agg = None
#     if include_resstock and all_bldg_ids:
#         print("\n" + "=" * 70)
#         print("RESSTOCK BASELINE")
#         print("=" * 70)
#         resstock_bldg_ids = sorted(all_bldg_ids)
#         resstock_agg = aggregate_resstock(
#             resstock_bldg_ids, resstock_dir, output_columns,
#             use_threading=use_threading, n_workers=n_workers,
#         )
#         resstock_path = comparison_dir / "resstock_baseline_ts.csv"
#         resstock_agg.to_csv(resstock_path, index=False)
#         print(f"ResStock output saved to: {resstock_path}")
#         plot_results(resstock_agg, output_columns, "ResStock Baseline",
#                      comparison_dir / "resstock_baseline_ts.png")

#     # ---- Overlay comparison plot --------------------------------------------
#     plot_multi_case_comparison(
#         case_results, resstock_agg, output_columns,
#         comparison_dir / "case_comparison_ts.png",
#     )

#     # ---- Annual totals comparison table -------------------------------------
#     metrics = [
#         ("Annual Grid Purchases [TWh]", output_columns[0]),
#         ("Annual Heating [TWh]",        output_columns[1]),
#         ("Annual Cooling [TWh]",        output_columns[2]),
#     ]
#     table = {"Metric": [m[0] for m in metrics]}

#     for case_name, agg in case_results.items():
#         table[case_name] = [agg[col].sum() / 1_000 for _, col in metrics]  # GWh -> TWh

#     if resstock_agg is not None:
#         table["ResStock Baseline"] = [
#             resstock_agg[col].sum() / 1_000 for _, col in metrics
#         ]

#     annual_comparison = pd.DataFrame(table)

#     # Add a % difference vs ResStock for each case, if baseline exists.
#     if resstock_agg is not None:
#         for case_name in case_results:
#             annual_comparison[f"{case_name} vs RS [%]"] = (
#                 (annual_comparison[case_name] - annual_comparison["ResStock Baseline"])
#                 / annual_comparison["ResStock Baseline"] * 100
#             ).round(2)

#     print("\n" + "=" * 70)
#     print("ANNUAL COMPARISON ACROSS CASES")
#     print("=" * 70)
#     print(annual_comparison.to_string(index=False))

#     comparison_csv_path = comparison_dir / "annual_comparison.csv"
#     annual_comparison.to_csv(comparison_csv_path, index=False)
#     print(f"\nAnnual comparison saved to: {comparison_csv_path}")

"""
Aggregate building timeseries across ERCOT buildings for ONE OR MORE cases,
then compare the cases against each other and against the ResStock baseline.

NO scaling, NO counts, NO nearest-neighbor mapping:
    - For each case, simply SUM every ts.csv that exists in the case dir.
    - The ResStock baseline is summed once over the same set of buildings
      (it is case-independent) and used as a common comparison point.

Units: ts.csv buy is kW -> GW; HVAC/CHP and ResStock loads are kWh -> GWh.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: writes PNGs, no GUI/thread issues
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


# =============================================================================
# Building discovery
# =============================================================================
def find_case_bldg_ids(case_dir, upgrade):
    """Return the list of bldg_ids that actually have a ts.csv in this case."""
    bldg_ids = []
    if not case_dir.exists():
        print(f"  WARNING: case dir does not exist: {case_dir}")
        return bldg_ids
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        ts_path = child / upgrade / "out" / "ts.csv"
        if ts_path.exists():
            try:
                bldg_ids.append(int(child.name))
            except ValueError:
                continue
    bldg_ids.sort()
    return bldg_ids


# =============================================================================
# ts.csv aggregation (the decarb case)
# =============================================================================
def _load_single_ts(args):
    """Load a single building's ts.csv. No scaling."""
    bldg_id, case_dir, upgrade, ts_columns = args
    ts_file = case_dir / str(int(bldg_id)) / upgrade / "out" / "ts.csv"
    try:
        ts_data = pd.read_csv(ts_file, usecols=ts_columns)
    except pd.errors.EmptyDataError:
        print(f"WARNING: Empty file skipped: {ts_file}")
        ts_data = pd.DataFrame(0, index=range(8760), columns=ts_columns)
    buy = ts_data["buy"].values
    hvac_ht = ts_data["HVACht"].values
    hvac_ac = ts_data["HVACac"].values
    chp_ht = ts_data["CHPht"].values
    return np.column_stack([buy, hvac_ht, hvac_ac, chp_ht])


def aggregate_timeseries(bldg_ids, case_dir, upgrade, ts_columns, output_columns,
                         use_threading=False, n_workers=20):
    """
    Sum the ts.csv of every building in bldg_ids.
    Combines HVACht + CHPht into heating. Converts kW -> GW.
    """
    total = len(bldg_ids)
    args_list = [(bid, case_dir, upgrade, ts_columns) for bid in bldg_ids]

    aggregated = None

    def _accumulate(iterable):
        nonlocal aggregated
        for i, stacked in enumerate(iterable):
            if aggregated is None:
                aggregated = stacked.astype(np.float64)
            else:
                aggregated += stacked
            if (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{total}...")

    if use_threading:
        print(f"  Summing ts.csv for {total} buildings with {n_workers} workers...")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            _accumulate(executor.map(_load_single_ts, args_list))
    else:
        print(f"  Summing ts.csv for {total} buildings (sequential)...")
        _accumulate(_load_single_ts(a) for a in args_list)

    result = pd.DataFrame({
        output_columns[0]: aggregated[:, 0],                       # buy: kW
        output_columns[1]: aggregated[:, 1] + aggregated[:, 3],    # HVACht + CHPht: kWh
        output_columns[2]: aggregated[:, 2],                       # HVACac: kWh
    })
    return result


# =============================================================================
# ResStock baseline aggregation
# =============================================================================
KBTU_TO_KWH = 0.293071


def _load_single_resstock(args):
    """Load a single ResStock building's parquet. No scaling."""
    bldg_id, resstock_dir = args
    pq_file = resstock_dir / f"{int(bldg_id)}-0.parquet"
    df = pd.read_parquet(pq_file, engine="fastparquet")
    elec = df["out.electricity.total.energy_consumption..kwh"].fillna(0).values.astype(np.float64)
    heat = df["out.load.heating.energy_delivered..kbtu"].fillna(0).values.astype(np.float64)
    cool = df["out.load.cooling.energy_delivered..kbtu"].fillna(0).values.astype(np.float64)
    # Downsample 15-min -> hourly by summing every 4 rows
    elec_h = elec.reshape(-1, 4).sum(axis=1)
    heat_h = heat.reshape(-1, 4).sum(axis=1) * KBTU_TO_KWH
    cool_h = cool.reshape(-1, 4).sum(axis=1) * KBTU_TO_KWH
    return np.column_stack([elec_h, heat_h, cool_h])


def aggregate_resstock(bldg_ids, resstock_dir, output_columns,
                       use_threading=False, n_workers=20):
    """Sum the ResStock parquet of every building in bldg_ids. No scaling."""
    total = len(bldg_ids)
    args_list = [(bid, resstock_dir) for bid in bldg_ids]

    aggregated = None

    def _accumulate(iterable):
        nonlocal aggregated
        for i, stacked in enumerate(iterable):
            if aggregated is None:
                aggregated = stacked.astype(np.float64)
            else:
                aggregated += stacked
            if (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{total}...")

    if use_threading:
        print(f"  Summing ResStock for {total} buildings with {n_workers} workers...")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            _accumulate(executor.map(_load_single_resstock, args_list))
    else:
        print(f"  Summing ResStock for {total} buildings (sequential)...")
        _accumulate(_load_single_resstock(a) for a in args_list)

    result = pd.DataFrame({
        output_columns[0]: aggregated[:, 0],  # kW
        output_columns[1]: aggregated[:, 1],  # kWh
        output_columns[2]: aggregated[:, 2],  # kWh
    })
    return result


# =============================================================================
# Plotting
# =============================================================================
def plot_results(aggregated, output_columns, label, plot_path):
    """Plot the three timeseries for a single dataset in three subplots."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Aggregated Building Timeseries — {label}", fontsize=14)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for ax, col, color in zip(axes, output_columns, colors):
        ax.plot(aggregated[col], linewidth=0.5, color=color)
        ax.set_ylabel(col.split("[")[-1].rstrip("]"))
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Hour")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {plot_path}")


def plot_multi_case_comparison(case_results, resstock_agg, output_columns, plot_path):
    """
    Overlay all cases (and the ResStock baseline) on the same three subplots.
    case_results: dict {case_name: aggregated_df}
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Case Comparison vs ResStock Baseline", fontsize=14)

    # Distinct colors for cases; baseline always gray.
    cmap = plt.get_cmap("tab10")

    for ax, col in zip(axes, output_columns):
        if resstock_agg is not None:
            ax.plot(resstock_agg[col], linewidth=0.6, color="gray",
                    alpha=0.7, label="ResStock Baseline")
        for idx, (case_name, agg) in enumerate(case_results.items()):
            ax.plot(agg[col], linewidth=0.5, color=cmap(idx % 10), label=case_name)
        ax.set_ylabel(col.split("[")[-1].rstrip("]"))
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Hour")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nComparison plot saved to: {plot_path}")


def plot_50_vs_resstock(case_results, resstock_agg, output_columns, plot_path):
    """Overlay elec_0_flat_derate_50 and ResStock baseline only."""
    derate_50_key = next((k for k in case_results if k == "elec_0_flat_derate_50"), None)
    if derate_50_key is None:
        print("WARNING: elec_0_flat_derate_50 not in case_results — skipping 50 vs ResStock plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("elec_0_flat_derate_50 vs ResStock Baseline", fontsize=14)

    for ax, col in zip(axes, output_columns):
        if resstock_agg is not None:
            ax.plot(resstock_agg[col], linewidth=0.6, color="gray", alpha=0.7, label="ResStock Baseline")
        ax.plot(case_results[derate_50_key][col], linewidth=0.5, color="#d62728", label="derate_50")
        ax.set_ylabel(col.split("[")[-1].rstrip("]"))
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Hour")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n50 vs ResStock plot saved to: {plot_path}")


def plot_peak_days(case_results, resstock_agg, output_columns, plot_path):
    """
    Find the summer and winter peak days using the ResStock Grid Purchases profile,
    then plot all three series (ResStock, elec_0_flat, elec_0_flat_derate_50)
    for those 24 hours in a combined 3-row x 2-column figure.

    Peak day search windows (hourly, Jan 1 start):
        Summer: hours 2160-6551  (Apr-Sep)
        Winter: hours 0-2159 and 6552-8759  (Jan-Mar, Oct-Dec)
    """
    grid_col = output_columns[0]

    noderate_key = next((k for k in case_results if k == "elec_0_flat_results"), None)
    derate50_key = next((k for k in case_results if k == "elec_0_flat_derate_50"), None)

    missing = [name for name, key in [("elec_0_flat_results", noderate_key),
                                       ("elec_0_flat_derate_50", derate50_key)] if key is None]
    if missing or resstock_agg is None:
        print(f"WARNING: peak day plot skipped — missing: {missing or ['ResStock']}")
        return

    buy_rs = resstock_agg[grid_col].values

    # Summer peak day
    summer_hours = np.arange(2160, 6552)
    summer_day_start = (summer_hours[np.argmax(buy_rs[summer_hours])] // 24) * 24
    summer_slice = slice(summer_day_start, summer_day_start + 24)

    # Winter peak day
    winter_hours = np.concatenate([np.arange(0, 2160), np.arange(6552, 8760)])
    winter_day_start = (winter_hours[np.argmax(buy_rs[winter_hours])] // 24) * 24
    winter_slice = slice(winter_day_start, winter_day_start + 24)

    print(f"  Summer peak day starts at hour {summer_day_start}")
    print(f"  Winter peak day starts at hour {winter_day_start}")

    hours = np.arange(24)
    series = [
        ("ResStock Baseline",       resstock_agg,                  "gray"),
        ("elec_0_flat_results",             case_results[noderate_key],    "#1f77b4"),
        ("elec_0_flat_derate_50",   case_results[derate50_key],    "#d62728"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharey="row")
    fig.suptitle("Peak Day Profiles — determined from ResStock Grid Purchases", fontsize=14)
    axes[0, 0].set_title(f"Summer Peak Day (hours {summer_day_start}–{summer_day_start + 23})")
    axes[0, 1].set_title(f"Winter Peak Day (hours {winter_day_start}–{winter_day_start + 23})")

    for row, col in enumerate(output_columns):
        for label, df, color in series:
            axes[row, 0].plot(hours, df[col].values[summer_slice], linewidth=1.2,
                              color=color, label=label)
            axes[row, 1].plot(hours, df[col].values[winter_slice], linewidth=1.2,
                              color=color, label=label)
        for c in range(2):
            axes[row, c].set_ylabel(col.split("[")[-1].rstrip("]"))
            axes[row, c].set_xlabel("Hour of day")
            axes[row, c].grid(True, alpha=0.3)
        axes[row, 0].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Peak day plot saved to: {plot_path}")


# =============================================================================
# Per-case driver
# =============================================================================
def process_case(case_name, ercot_root, upgrade, ts_columns, output_columns,
                 use_threading, n_workers, bldg_ids=None):
    """Run the full ts.csv aggregation for one case and save its outputs.

    If bldg_ids is provided, use that list directly instead of discovering
    buildings from the case directory (used to pin the baseline to the same
    set of buildings as the derate cases).
    """
    print("\n" + "=" * 70)
    print(f"CASE: {case_name}")
    print("=" * 70)

    case_dir = ercot_root / "in" / "decarb_inputs" / case_name
    results_dir = ercot_root / "out" / "decarb_results" / case_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if bldg_ids is None:
        bldg_ids = find_case_bldg_ids(case_dir, upgrade)
    else:
        print(f"  Using pre-supplied building list ({len(bldg_ids)} buildings)")

    print(f"  Buildings with ts.csv: {len(bldg_ids)}")
    if not bldg_ids:
        print("  No buildings found — skipping case.")
        return None, []

    aggregated = aggregate_timeseries(
        bldg_ids, case_dir, upgrade, ts_columns, output_columns,
        use_threading=use_threading, n_workers=n_workers,
    )

    output_path = results_dir / f"{case_name}_aggregated_ts.csv"
    plot_path = results_dir / f"{case_name}_aggregated_ts.png"
    aggregated.to_csv(output_path, index=False)
    print(f"  Output saved to: {output_path}  (shape {aggregated.shape})")
    plot_results(aggregated, output_columns, case_name, plot_path)

    return aggregated, bldg_ids


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # ---- Config -------------------------------------------------------------
    # List one or more cases to run and compare.
    case_upgrade = {
        "elec_0_flat_results":   "update_0",
        "elec_0_flat_derate_10": "update_0",
        "elec_0_flat_derate_20": "update_0",
        "elec_0_flat_derate_30": "update_0",
        "elec_0_flat_derate_40": "update_0",
        "elec_0_flat_derate_50": "update_0",
    }
    use_threading = True
    n_workers = 20
    include_resstock = True  # set False to skip the baseline entirely

    # ---- Paths --------------------------------------------------------------
    script_path = Path(__file__).resolve()
    ercot_root = script_path.parents[1]
    resstock_dir = ercot_root / "out" / "consumption_files"
    comparison_dir = ercot_root / "in" / "decarb_inputs" / "derate_comp"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # ---- Columns ------------------------------------------------------------
    ts_columns = ["buy", "HVACht", "HVACac", "CHPht"]
    output_columns = [
        "Grid Purchases [kW]",
        "Heating Generated [kWh]",
        "Cooling Generated [kWh]",
    ]

    # Separate the baseline case from the derate cases
    baseline_case = "elec_0_flat_results"
    derate_cases = {k: v for k, v in case_upgrade.items() if k != baseline_case}

    # ---- Run derate cases first to establish the building list --------------
    case_results = {}      # {case_name: aggregated_df}
    all_bldg_ids = set()   # union of buildings across derate cases
    for case_name, upgrade in derate_cases.items():
        agg, bldg_ids = process_case(
            case_name, ercot_root, upgrade, ts_columns, output_columns,
            use_threading, n_workers,
        )
        if agg is not None:
            case_results[case_name] = agg
            all_bldg_ids.update(bldg_ids)

    if not case_results:
        raise SystemExit("No derate cases produced results — nothing to compare.")

    # ---- Run baseline pinned to the same 100 buildings ----------------------
    pinned_bldg_ids = sorted(all_bldg_ids)
    agg, _ = process_case(
        baseline_case, ercot_root, case_upgrade[baseline_case], ts_columns, output_columns,
        use_threading, n_workers, bldg_ids=pinned_bldg_ids,
    )
    if agg is not None:
        case_results[baseline_case] = agg

    # ---- ResStock baseline (summed once over all buildings seen) ------------
    resstock_agg = None
    if include_resstock and all_bldg_ids:
        print("\n" + "=" * 70)
        print("RESSTOCK BASELINE")
        print("=" * 70)
        resstock_bldg_ids = sorted(all_bldg_ids)
        resstock_agg = aggregate_resstock(
            resstock_bldg_ids, resstock_dir, output_columns,
            use_threading=use_threading, n_workers=n_workers,
        )
        resstock_path = comparison_dir / "resstock_baseline_ts.csv"
        resstock_agg.to_csv(resstock_path, index=False)
        print(f"ResStock output saved to: {resstock_path}")
        plot_results(resstock_agg, output_columns, "ResStock Baseline",
                     comparison_dir / "resstock_baseline_ts.png")

    # ---- Overlay comparison plot --------------------------------------------
    plot_multi_case_comparison(
        case_results, resstock_agg, output_columns,
        comparison_dir / "case_comparison_ts.png",
    )

    # ---- Annual totals comparison table -------------------------------------
    metrics = [
        ("Annual Grid Purchases [TWh]", output_columns[0]),
        ("Annual Heating [TWh]",        output_columns[1]),
        ("Annual Cooling [TWh]",        output_columns[2]),
    ]
    table = {"Metric": [m[0] for m in metrics]}

    for case_name, agg in case_results.items():
        table[case_name] = [agg[col].sum() / 1_000_000_000 for _, col in metrics]  # kWh -> TWh

    if resstock_agg is not None:
        table["ResStock Baseline"] = [
            resstock_agg[col].sum() / 1_000_000_000 for _, col in metrics  # kWh -> TWh
        ]

    annual_comparison = pd.DataFrame(table)

    # Add a % difference vs ResStock for each case, if baseline exists.
    if resstock_agg is not None:
        for case_name in case_results:
            annual_comparison[f"{case_name} vs RS [%]"] = (
                (annual_comparison[case_name] - annual_comparison["ResStock Baseline"])
                / annual_comparison["ResStock Baseline"] * 100
            ).round(2)

    print("\n" + "=" * 70)
    print("ANNUAL COMPARISON ACROSS CASES")
    print("=" * 70)
    print(annual_comparison.to_string(index=False))

    comparison_csv_path = comparison_dir / "annual_comparison.csv"
    annual_comparison.to_csv(comparison_csv_path, index=False)
    print(f"\nAnnual comparison saved to: {comparison_csv_path}")

    # ---- derate_50 vs ResStock only -----------------------------------------
    plot_50_vs_resstock(
        case_results, resstock_agg, output_columns,
        comparison_dir / "derate_50_vs_resstock.png",
    )

    # ---- Peak day profiles --------------------------------------------------
    print("\nFinding peak days...")
    plot_peak_days(
        case_results, resstock_agg, output_columns,
        comparison_dir / "peak_day_profiles.png",
    )