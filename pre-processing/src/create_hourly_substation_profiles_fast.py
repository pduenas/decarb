from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
import multiprocessing as mp
from functools import partial


def map_substations(root_dir: Path, save_csv=False):
    """
    Process all P<number>R / P<number>U folders under root_dir.

    If save_csv is True:
        Save substation profiles to:
            root_dir/residential_profiles
        Skip substations with existing CSVs

    Always:
        Maintain sub_bldg_lookup.json mapping:
            substation -> { nrel_climate_match : count }
    """

    zone_pattern = re.compile(r"P\d+(R|U)$")

    output_dir = root_dir / "residential_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    lookup_file = output_dir / "sub_bldg_lookup.json"

    sub_bldg_lookup = {}
    if lookup_file.exists():
        try:
            with open(lookup_file, "r") as f:
                sub_bldg_lookup = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: {lookup_file} is empty or corrupted. Starting fresh.")
            sub_bldg_lookup = {}

    for zone_dir in root_dir.iterdir():
        if not zone_dir.is_dir():
            continue

        if not zone_pattern.match(zone_dir.name):
            continue

        feeder_summaries_dir = zone_dir / "feeder_summaries"
        if not feeder_summaries_dir.exists():
            continue

        zone_name = zone_dir.name
        print(f"Processing zone: {zone_name}")

        substation_files = {}

        for csv_file in feeder_summaries_dir.glob("feeder_summary*.csv"):
            match = re.search(r"feeder_summary_(.*?)--", csv_file.name)
            if not match:
                continue

            substation = match.group(1)
            substation_files.setdefault(substation, []).append(csv_file)

        for substation, files in substation_files.items():
            output_file = output_dir / "mapping_files" / f"res_profile_{substation}.csv"

            if save_csv and output_file.exists():
                print(f"  Skipping {substation} (already exists)")
                continue

            print(f"  Processing substation: {substation}")

            counts = {}

            for file in files:
                df = pd.read_csv(file)
                required_cols = {"nrel_climate_match", "yearly"}
                missing = required_cols - set(df.columns)
                if missing:
                    raise ValueError(f"Missing columns {missing} in {file}")

                df = df[df["yearly"].astype(str).str.startswith("res", na=False)]

                if df.empty:
                    continue

                vc = df["nrel_climate_match"].dropna().value_counts()

                for bldg_id, count in vc.items():
                    counts[int(bldg_id)] = counts.get(int(bldg_id), 0) + int(count)

            if save_csv:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                out_df = (
                    pd.DataFrame(
                        {
                            "nrel_climate_match": list(counts.keys()),
                            "count": list(counts.values()),
                        }
                    )
                    .sort_values("nrel_climate_match")
                    .reset_index(drop=True)
                )

                out_df.to_csv(output_file, index=False)
                print(f"    Saved: {output_file}")

            sub_bldg_lookup[substation] = {
                str(k): int(v) for k, v in counts.items()
            }

            with open(lookup_file, "w") as f:
                json.dump(sub_bldg_lookup, f, indent=2)


def to_hourly_max(series_15min: pd.Series) -> pd.Series:
    if len(series_15min) % 4 != 0:
        raise ValueError(
            "15-minute series length is not divisible by 4; "
            "cannot aggregate to hourly."
        )

    return (
        series_15min
        .groupby(series_15min.index // 4)
        .max()
        .reset_index(drop=True)
    )


def process_single_substation(substation_data, parquet_root, reference_len, hourly_output):
    """
    Process a single substation's load data.
    Designed to be run in parallel.
    """
    substation, climate_counts = substation_data
    
    substation_kwh = None

    for bldg_id_str, count in climate_counts.items():
        bldg_id = int(bldg_id_str)
        count = int(count)

        parquet_file = parquet_root / f"{bldg_id}-0.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(f"Missing parquet: {parquet_file}")

        # Read only the column we need
        df = pd.read_parquet(
            parquet_file, 
            columns=["out.electricity.total.energy_consumption..kwh"]
        )
        
        energy_kwh = df["out.electricity.total.energy_consumption..kwh"].astype(float) * count

        if substation_kwh is None:
            substation_kwh = energy_kwh.values  # Use numpy array for speed
        else:
            substation_kwh += energy_kwh.values

    if substation_kwh is None:
        substation_mw = np.zeros(reference_len)
        has_load = False
    else:
        substation_mw = (substation_kwh * 4.0) / 1000.0
        has_load = True

    if hourly_output:
        # Vectorized hourly max aggregation
        substation_mw = substation_mw.reshape(-1, 4).max(axis=1)

    return substation, substation_mw, has_load


def create_load_data_csv(
    lookup_json,
    parquet_root,
    output_csv,
    pretext_csv,
    hourly_output,
    n_workers=None,
):
    """
    Create Load_data_6716_bus.csv by aggregating substation loads.
    
    Optimizations:
    - Parallel processing of substations
    - Read only needed columns from parquet
    - Vectorized operations with numpy
    - Efficient hourly aggregation
    
    Args:
        n_workers: Number of parallel workers (default: CPU count - 1)
    """
    
    with open(lookup_json, "r") as f:
        sub_bldg_lookup = json.load(f)

    # Read the pretext file
    pretext_df = pd.read_csv(pretext_csv)
    num_metadata_rows = len(pretext_df)

    # Determine reference length from first parquet file
    first_substation = next(iter(sub_bldg_lookup.values()))
    first_bldg_id = int(next(iter(first_substation.keys())))
    first_file = parquet_root / f"{first_bldg_id}-0.parquet"
    
    reference_len = len(pd.read_parquet(first_file, columns=[]))
    expected_timesteps = 8760 if hourly_output else reference_len
    
    print(f"Processing {len(sub_bldg_lookup)} substations...")
    print(f"Reference length: {reference_len} intervals")
    print(f"Output timesteps: {expected_timesteps}")
    
    # Set up parallel processing
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {n_workers} parallel workers")
    
    # Prepare data for parallel processing
    substation_items = list(sub_bldg_lookup.items())
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_substation,
        parquet_root=parquet_root,
        reference_len=reference_len,
        hourly_output=hourly_output
    )
    
    # Process substations in parallel
    substation_series = {}
    zero_load_substations = []
    
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(process_func, substation_items)
        
        for substation, load_data, has_load in results:
            substation_series[f"Load_MW_{substation}"] = load_data
            if not has_load:
                zero_load_substations.append({
                    "substation": substation, 
                    "reason": "no_residential_load"
                })
                print(f"WARNING: Substation '{substation}' has no residential load")
    else:
        # Sequential processing (useful for debugging)
        for item in substation_items:
            substation, load_data, has_load = process_func(item)
            substation_series[f"Load_MW_{substation}"] = load_data
            if not has_load:
                zero_load_substations.append({
                    "substation": substation, 
                    "reason": "no_residential_load"
                })
                print(f"WARNING: Substation '{substation}' has no residential load")
    
    print(f"Completed aggregation for {len(substation_series)} substations")

    # Create the load data columns efficiently
    load_columns_df = pd.DataFrame(substation_series)
    
    if len(load_columns_df) != expected_timesteps:
        raise ValueError(
            f"Load data has {len(load_columns_df)} rows, "
            f"expected {expected_timesteps} timesteps"
        )

    # Determine which pretext columns should have values only in first N rows
    metadata_cols = []
    repeating_cols = []
    
    for col in pretext_df.columns:
        if 'Time_Index' in col or 'Sub_Weights' in col or 'Timesteps' in col:
            repeating_cols.append(col)
        else:
            metadata_cols.append(col)
    
    # Create expanded pretext dataframe efficiently
    pretext_expanded = {}
    
    for col in pretext_df.columns:
        if col in metadata_cols:
            # Metadata columns: fill only first N rows, rest are empty
            values = [''] * expected_timesteps
            for i in range(min(num_metadata_rows, expected_timesteps)):
                values[i] = pretext_df.loc[i, col]
            pretext_expanded[col] = values
        else:
            # Repeating columns
            if col == 'Time_Index':
                pretext_expanded[col] = list(range(1, expected_timesteps + 1))
            elif len(pretext_df[col].dropna()) == num_metadata_rows:
                pretext_expanded[col] = [pretext_df[col].iloc[-1]] * expected_timesteps
            else:
                values = [''] * expected_timesteps
                for i in range(min(num_metadata_rows, expected_timesteps)):
                    val = pretext_df.loc[i, col]
                    if pd.notna(val) and val != '':
                        values[i] = val
                pretext_expanded[col] = values
    
    pretext_expanded_df = pd.DataFrame(pretext_expanded)

    # Combine pretext and load data
    load_df = pd.concat(
        [pretext_expanded_df, load_columns_df],
        axis=1,
    )

    load_df.to_csv(output_csv, index=False)
    print(f"Saved load data to: {output_csv}")
    print(f"  Total rows: {len(load_df)}")
    print(f"  Metadata rows: {num_metadata_rows}")
    print(f"  Load columns: {len(substation_series)}")

    if zero_load_substations:
        report_df = pd.DataFrame(zero_load_substations)
        report_path = output_csv.parent / "zero_load_substations.csv"
        report_df.to_csv(report_path, index=False)
        print(
            f"Saved zero-load substation report to: {report_path} "
            f"({len(report_df)} substations)"
        )


if __name__ == "__main__":
    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]

    input_dir = parent_dir / "in"
    output_dir = parent_dir / "out"

    map_substations(output_dir, save_csv=True)

    consumption_path = output_dir / "consumption_files"
    lookup_json = output_dir / "residential_profiles" / "sub_bldg_lookup.json"
    load_csv = output_dir / "residential_profiles" / "Load_data_6716_bus.csv"
    pretext_csv = input_dir / "genx_demand_data_pretext.csv"

    create_load_data_csv(
        lookup_json=lookup_json,
        parquet_root=consumption_path,
        output_csv=load_csv,
        pretext_csv=pretext_csv,
        hourly_output=False,
        n_workers=32,  # Auto-detect CPU cores, or set to specific number like 4
    )