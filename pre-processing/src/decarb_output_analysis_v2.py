"""
Aggregate building timeseries data across ERCOT buildings.

For each building in the substation-NREL map:
- If ts.csv exists: org_bldg_id = new_bldg_id = bldg_id
- If ts.csv missing: match to nearest building (by Euclidean distance
    in summer/winter peak space) that DOES have a ts.csv

Then scale each building's timeseries by its count, aggregate, and convert kW -> GW.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from fastparquet import ParquetFile
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


def load_parquets(map_path, tx_path):
    """Load the substation map and TX_upgrade0 parquet files."""
    map_df = ParquetFile(str(map_path)).to_pandas(columns=["bldg_id", "count"])

    tx_df = ParquetFile(str(tx_path)).to_pandas(
        columns=[
            "bldg_id",
            "out.qoi.electricity.maximum_daily_peak_summer..kw",
            "out.qoi.electricity.maximum_daily_peak_winter..kw",
        ]
    )
    tx_df = tx_df.rename(columns={
        "out.qoi.electricity.maximum_daily_peak_summer..kw": "summer_peak",
        "out.qoi.electricity.maximum_daily_peak_winter..kw": "winter_peak",
    })

    return map_df, tx_df


def build_mapping(map_df, tx_df, case_dir, upgrade):
    """
    Build the (org_bldg_id, new_bldg_id, count) mapping dataframe.
    
    Buildings with ts.csv map to themselves.
    Buildings without ts.csv are matched to the nearest building
    (by Euclidean distance in summer/winter peak space) that has a ts.csv.
    """
    unique_bldg_ids = map_df["bldg_id"].unique()

    has_ts = []
    no_ts = []
    for bldg_id in unique_bldg_ids:
        ts_path = case_dir / str(int(bldg_id)) / upgrade / "out" / "ts.csv"
        if ts_path.exists():
            has_ts.append(int(bldg_id))
        else:
            no_ts.append(int(bldg_id))

    print(f"Buildings with ts.csv:    {len(has_ts)}")
    print(f"Buildings without ts.csv: {len(no_ts)}")
    if has_ts:
        print(f"  Sample has_ts path: {case_dir / str(has_ts[0]) / upgrade / 'out' / 'ts.csv'}")
    if no_ts:
        print(f"  Sample no_ts path:  {case_dir / str(no_ts[0]) / upgrade / 'out' / 'ts.csv'}")

    print(f"Buildings with ts.csv:    {len(has_ts)}")
    print(f"Buildings without ts.csv: {len(no_ts)}")

    # Aggregate counts per building
    count_per_bldg = map_df.groupby("bldg_id")["count"].sum().reset_index()
    count_per_bldg = count_per_bldg.rename(columns={"count": "total_count"})

    # Buildings WITH ts.csv: org = new = bldg_id
    rows = []
    for bldg_id in has_ts:
        c = count_per_bldg.loc[
            count_per_bldg["bldg_id"] == bldg_id, "total_count"
        ]
        total = c.values[0] if len(c) > 0 else 0
        rows.append({
            "org_bldg_id": bldg_id,
            "new_bldg_id": bldg_id,
            "count": total,
        })

    # Buildings WITHOUT ts.csv: nearest-neighbor match
    if no_ts:
        has_ts_set = set(has_ts)
        candidates = tx_df[tx_df["bldg_id"].isin(has_ts_set)].copy()
        cand_summers = candidates["summer_peak"].values
        cand_winters = candidates["winter_peak"].values
        cand_ids = candidates["bldg_id"].values

        for bldg_id in no_ts:
            row = tx_df[tx_df["bldg_id"] == bldg_id]
            if row.empty:
                print(f"  WARNING: bldg_id {bldg_id} not found in TX_upgrade0.parquet, skipping")
                continue

            s_peak = row["summer_peak"].values[0]
            w_peak = row["winter_peak"].values[0]

            dists = np.sqrt(
                (cand_summers - s_peak) ** 2 + (cand_winters - w_peak) ** 2
            )
            best_idx = np.argmin(dists)
            matched_id = cand_ids[best_idx]

            c = count_per_bldg.loc[
                count_per_bldg["bldg_id"] == bldg_id, "total_count"
            ]
            total = c.values[0] if len(c) > 0 else 0

            rows.append({
                "org_bldg_id": bldg_id,
                "new_bldg_id": matched_id,
                "count": total,
            })

    mapping_df = pd.DataFrame(rows)
    print(f"\nMapping table: {len(mapping_df)} rows")
    print(f"  Unique new_bldg_ids: {mapping_df['new_bldg_id'].nunique()}")
    print(mapping_df.head(10))

    return mapping_df


def _load_single_ts(args):
    """Worker function to load and scale a single building's timeseries."""
    new_id, total_count, case_dir, upgrade, ts_columns = args
    ts_file = case_dir / str(int(new_id)) / upgrade / "out" / "ts.csv"
    # ts_data = pd.read_csv(ts_file, usecols=ts_columns)
    try:
        ts_data = pd.read_csv(ts_file, usecols=ts_columns)
    except pd.errors.EmptyDataError:
        print(f"WARNING: Empty file skipped: {ts_file}")
        # return a zero-filled DataFrame with the expected columns
        ts_data = pd.DataFrame(0, index=range(8760), columns=ts_columns)
    buy = ts_data["buy"].values
    hvac_ht = ts_data["HVACht"].values
    hvac_ac = ts_data["HVACac"].values
    chp_ht = ts_data["CHPht"].values
    stacked = np.column_stack([buy, hvac_ht, hvac_ac, chp_ht])
    return stacked * total_count


def aggregate_timeseries(mapping_df, case_dir, upgrade, ts_columns, output_columns,
                        use_threading=False, n_workers=20):
    """
    Load each unique new_bldg_id's ts.csv once, multiply by the total
    count across all buildings mapped to it, then sum.
    Combines HVACht + CHPht into heating. Converts kWh -> GWh.
    """
    # Sum counts per new_bldg_id (many org buildings can map to the same new)
    grouped = mapping_df.groupby("new_bldg_id")["count"].sum()
    total = len(grouped)

    args_list = [
        (new_id, total_count, case_dir, upgrade, ts_columns)
        for new_id, total_count in grouped.items()
    ]

    aggregated = None

    if use_threading:
        print(f"\nAggregating timeseries for {total} unique buildings with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, scaled in enumerate(executor.map(_load_single_ts, args_list)):
                if aggregated is None:
                    aggregated = scaled.copy()
                else:
                    aggregated += scaled
                if (i + 1) % 2000 == 0:
                    print(f"  Processed {i + 1}/{total}...")
    else:
        print(f"\nAggregating timeseries for {total} unique buildings (sequential)...")
        for i, args in enumerate(args_list):
            scaled = _load_single_ts(args)
            if aggregated is None:
                aggregated = scaled.copy()
            else:
                aggregated += scaled
            if (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{total}...")

    # aggregated columns: buy, HVACht, HVACac, CHPht
    # Combine HVACht + CHPht into heating, keep buy and HVACac
    result = pd.DataFrame({
        output_columns[0]: aggregated[:, 0] / 1_000_000,  # buy: kW -> GW
        output_columns[1]: (aggregated[:, 1] + aggregated[:, 3]) / 1_000_000,  # HVACht + CHPht: kWh -> GWh
        output_columns[2]: aggregated[:, 2] / 1_000_000,  # HVACac: kWh -> GWh
    })

    return result


def _compute_single_peak(args):
    """Worker function for a single building's peak conditions."""
    bldg_id, case_dir, upgrade = args
    ts_file = case_dir / str(int(bldg_id)) / upgrade / "out" / "ts.csv"
    buy = pd.read_csv(ts_file, usecols=["buy"])["buy"].values

    # Assuming 8760 hourly timesteps starting Jan 1
    # Winter = Jan, Feb, Mar, Oct, Nov, Dec (indices 0-2159, 6552-8759)
    # Summer = Apr-Sep (indices 2160-6551)
    winter_vals = np.concatenate([buy[:2160], buy[6552:]])
    summer_vals = buy[2160:6552]

    winter_peak = winter_vals.max()
    summer_peak = summer_vals.max()

    return {
        "bldg_id": int(bldg_id),
        "winter_peak": winter_peak,
        "summer_peak": summer_peak,
        "winter_dominant": 1 if winter_peak > summer_peak else 0,
    }


def compute_peak_conditions(mapping_df, case_dir, upgrade, results_dir,
                            use_threading=False, n_workers=20):
    """
    For each unique building with a ts.csv, compute winter and summer peak
    from the buy column.
    """
    unique_bldg_ids = mapping_df.loc[
        mapping_df["org_bldg_id"] == mapping_df["new_bldg_id"], "org_bldg_id"
    ].unique()

    args_list = [(bldg_id, case_dir, upgrade) for bldg_id in unique_bldg_ids]

    results = []
    if use_threading:
        print(f"\nComputing peak conditions for {len(unique_bldg_ids)} buildings with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, result in enumerate(executor.map(_compute_single_peak, args_list)):
                results.append(result)
                if (i + 1) % 2000 == 0:
                    print(f"  Processed {i + 1}/{len(unique_bldg_ids)}...")
    else:
        print(f"\nComputing peak conditions for {len(unique_bldg_ids)} buildings (sequential)...")
        for i, args in enumerate(args_list):
            results.append(_compute_single_peak(args))
            if (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{len(unique_bldg_ids)}...")

    peak_df = pd.DataFrame(results)
    peak_path = results_dir / "peak_conditions.csv"
    peak_df.to_csv(peak_path, index=False)
    print(f"Peak conditions saved to: {peak_path}")
    print(f"  Winter dominant: {peak_df['winter_dominant'].sum()} / {len(peak_df)}")

    return peak_df


def plot_results(aggregated, output_columns, case_name, plot_path):
    """Plot all three timeseries in three subplots."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Aggregated Building Timeseries — {case_name} (GW)", fontsize=14)

    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for ax, col, color in zip(axes, output_columns, colors):
        ax.plot(aggregated[col], linewidth=0.5, color=color)
        ax.set_ylabel("GW")
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.tight_layout()

    fig.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to: {plot_path}")


# =============================================================================
# ResStock baseline functions
# =============================================================================

KBTU_TO_KWH = 0.293071

def _load_single_resstock(args):
    """Worker function to load and scale a single ResStock building's parquet."""
    bldg_id, total_count, resstock_dir = args
    pq_file = resstock_dir / f"{int(bldg_id)}-0.parquet"
    df = pd.read_parquet(pq_file, engine="fastparquet")
    # Select columns by name, fill NaN with 0, force float64
    elec = df["out.electricity.total.energy_consumption..kwh"].fillna(0).values.astype(np.float64)
    heat = df["out.load.heating.energy_delivered..kbtu"].fillna(0).values.astype(np.float64)
    cool = df["out.load.cooling.energy_delivered..kbtu"].fillna(0).values.astype(np.float64)
    # Downsample 15-min -> hourly by summing every 4 rows
    elec_h = elec.reshape(-1, 4).sum(axis=1)
    heat_h = heat.reshape(-1, 4).sum(axis=1) * KBTU_TO_KWH
    cool_h = cool.reshape(-1, 4).sum(axis=1) * KBTU_TO_KWH
    # Stack and scale
    hourly = np.column_stack([elec_h, heat_h, cool_h])
    return hourly * total_count


def aggregate_resstock(mapping_df, resstock_dir, output_columns,
                    use_threading=False, n_workers=20):
    """
    Aggregate ResStock baseline parquet files using the same mapping/scaling.
    Converts electricity kWh -> GW, heating/cooling kWh -> GWh.
    """
    grouped = mapping_df.groupby("new_bldg_id")["count"].sum()
    total = len(grouped)

    args_list = [
        (new_id, total_count, resstock_dir)
        for new_id, total_count in grouped.items()
    ]

    aggregated = None

    if use_threading:
        print(f"\nAggregating ResStock for {total} unique buildings with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, scaled in enumerate(executor.map(_load_single_resstock, args_list)):
                if aggregated is None:
                    aggregated = scaled.astype(np.float64)
                    print(f"  DEBUG first result: shape={scaled.shape}, col_sums={scaled.sum(axis=0)}")
                else:
                    aggregated += scaled
                if (i + 1) % 2000 == 0:
                    print(f"  Processed {i + 1}/{total}...")
    else:
        print(f"\nAggregating ResStock for {total} unique buildings (sequential)...")
        for i, args in enumerate(args_list):
            scaled = _load_single_resstock(args)
            if aggregated is None:
                aggregated = scaled.astype(np.float64)
                print(f"  DEBUG first result: shape={scaled.shape}, col_sums={scaled.sum(axis=0)}")
            else:
                aggregated += scaled
            if (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{total}...")

    result = pd.DataFrame({
        output_columns[0]: aggregated[:, 0] / 1_000_000,  # kWh (hourly) = kW -> GW
        output_columns[1]: aggregated[:, 1] / 1_000_000,  # kWh -> GWh
        output_columns[2]: aggregated[:, 2] / 1_000_000,  # kWh -> GWh
    })

    return result


def plot_resstock(resstock_agg, output_columns, plot_path):
    """Plot ResStock baseline timeseries."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("ResStock Baseline (GW / GWh)", fontsize=14)

    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for ax, col, color in zip(axes, output_columns, colors):
        ax.plot(resstock_agg[col], linewidth=0.5, color=color)
        ax.set_ylabel("GW" if "GW]" in col else "GWh")
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Hour")
    fig.tight_layout()

    fig.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to: {plot_path}")


def plot_comparison(aggregated, resstock_agg, output_columns, case_name, plot_path):
    """Overlay decarb case vs ResStock baseline."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{case_name} vs ResStock Baseline", fontsize=14)

    for ax, col in zip(axes, output_columns):
        ax.plot(resstock_agg[col], linewidth=0.5, color="gray", alpha=0.7, label="ResStock Baseline")
        ax.plot(aggregated[col], linewidth=0.5, color="#d62728", label=case_name)
        ax.set_ylabel("GW" if "GW]" in col else "GWh")
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Hour")
    fig.tight_layout()

    fig.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved to: {plot_path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # Config variables
    case_name = "elec_0_tou_results"
    network_config = "1bus"
    upgrade = "update_0"
    use_threading = True
    n_workers = 20

    # File paths
    script_path = Path(__file__).resolve()
    ercot_root = script_path.parents[1]
    case_dir = ercot_root / "in" / "decarb_inputs_alt" / case_name
    map_path = ercot_root / "out" / "ercot_substation_nrel_map.parquet"
    tx_path = ercot_root / "in" / "TX_upgrade0.parquet"
    resstock_dir = ercot_root / "out" / "consumption_files"
    results_dir = ercot_root / "out" / "decarb_results_alt" / case_name
    output_path = results_dir / f"{case_name}_aggregated_ts.csv"
    plot_path = results_dir / f"{case_name}_aggregated_ts.png"
    resstock_output_path = results_dir / "resstock_baseline_ts.csv"
    resstock_plot_path = results_dir / "resstock_baseline_ts.png"
    comparison_plot_path = results_dir / f"{case_name}_vs_resstock.png"

    # Input columns from ts.csv
    ts_columns = ["buy", "HVACht", "HVACac", "CHPht"]
    # Output column names
    output_columns = [
        "Grid Purchases [GW]",
        "Heating Generated [GWh]",
        "Cooling Generated [GWh]",
    ]

    # Load data
    map_df, tx_df = load_parquets(map_path, tx_path)

    # Build building mapping
    mapping_df = build_mapping(map_df, tx_df, case_dir, upgrade)

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save mapping table
    mapping_path = results_dir / "building_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Mapping saved to: {mapping_path}")

    # Aggregate timeseries
    aggregated = aggregate_timeseries(mapping_df, case_dir, upgrade, ts_columns, output_columns,
                                    use_threading=use_threading, n_workers=n_workers)

    # Compute peak conditions per building
    peak_df = compute_peak_conditions(mapping_df, case_dir, upgrade, results_dir,
                                    use_threading=use_threading, n_workers=n_workers)

    # Save aggregated timeseries
    aggregated.to_csv(output_path, index=False)
    print(f"\nOutput saved to: {output_path}")
    print(f"Shape: {aggregated.shape}")
    print(aggregated.head())

    # Plot decarb case
    plot_results(aggregated, output_columns, case_name, plot_path)

    # Aggregate ResStock baseline
    resstock_agg = aggregate_resstock(mapping_df, resstock_dir, output_columns,
                                    use_threading=use_threading, n_workers=n_workers)
    resstock_agg.to_csv(resstock_output_path, index=False)
    print(f"\nResStock output saved to: {resstock_output_path}")

    # Plot ResStock baseline
    plot_resstock(resstock_agg, output_columns, resstock_plot_path)

    # Plot comparison overlay
    plot_comparison(aggregated, resstock_agg, output_columns, case_name, comparison_plot_path)

    # Annual comparison: heating and cooling consumption
    annual_comparison = pd.DataFrame({
        "Metric": [
            "Annual Heating [TWh]",
            "Annual Cooling [TWh]",
            "Annual Grid Purchases [TWh]",
        ],
        case_name: [
            aggregated[output_columns[1]].sum() / 1_000,  # GWh -> TWh
            aggregated[output_columns[2]].sum() / 1_000,
            aggregated[output_columns[0]].sum() / 1_000,
        ],
        "ResStock Baseline": [
            resstock_agg[output_columns[1]].sum() / 1_000,
            resstock_agg[output_columns[2]].sum() / 1_000,
            resstock_agg[output_columns[0]].sum() / 1_000,
        ],
    })
    annual_comparison["Difference"] = annual_comparison[case_name] - annual_comparison["ResStock Baseline"]
    annual_comparison["Difference [%]"] = (
        (annual_comparison[case_name] - annual_comparison["ResStock Baseline"])
        / annual_comparison["ResStock Baseline"] * 100
    ).round(2)

    print("\n" + "=" * 70)
    print("ANNUAL COMPARISON: Decarb Case vs ResStock Baseline")
    print("=" * 70)
    print(annual_comparison.to_string(index=False))

    comparison_csv_path = results_dir / "annual_comparison.csv"
    annual_comparison.to_csv(comparison_csv_path, index=False)
    print(f"\nAnnual comparison saved to: {comparison_csv_path}")