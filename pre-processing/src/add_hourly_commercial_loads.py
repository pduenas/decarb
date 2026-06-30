from pathlib import Path
import pandas as pd
import numpy as np
import re
import multiprocessing as mp
from functools import partial


def to_hourly_max(series_15min: pd.Series) -> pd.Series:
    """
    Convert 15-minute series to hourly by taking max of each 4 intervals.
    Handles edge cases where data might already be hourly or different resolution.
    Always pads with zeros if needed to make data divisible by 4.
    """
    length = len(series_15min)
    
    # Check if already hourly (8760 values)
    if length == 8760:
        print("  Data appears to be already hourly, skipping aggregation")
        return series_15min
    
    # Check if it's approximately 15-minute data (around 35040 values)
    # Allow for small variations due to missing data or different year lengths
    if 35000 <= length <= 35100:
        # Check if divisible by 4
        if length % 4 != 0:
            # Pad with zeros to make it divisible by 4
            remainder = length % 4
            pad_length = 4 - remainder
            print(f"  Warning: Padding {pad_length} zeros to length {length} to make divisible by 4")
            
            # Pad with zeros
            padded = np.zeros(length + pad_length)
            padded[:length] = series_15min.values
            series_15min = pd.Series(padded)
            length = len(series_15min)
        
        # Vectorized hourly max aggregation
        return pd.Series(series_15min.values.reshape(-1, 4).max(axis=1))
    
    # Check if it's approximately hourly but slightly off
    if 8700 <= length <= 8800:
        print(f"  Warning: Data length {length} is close to hourly (8760)")
        # Pad to exactly 8760
        if length < 8760:
            padded = np.zeros(8760)
            padded[:length] = series_15min.values
            return pd.Series(padded)
        elif length > 8760:
            # Trim if too long
            return series_15min.iloc[:8760]
        return series_15min
    
    # Unknown resolution
    raise ValueError(
        f"Unexpected series length: {length}. "
        f"Expected ~8760 (hourly) or ~35040 (15-min)"
    )


def process_single_substation_commercial(
    substation_data, 
    output_dir, 
    com_root, 
    zone_pattern,
    hourly_output
):
    """
    Process a single substation's commercial load data.
    Designed to be run in parallel.
    """
    substation, zone_files = substation_data
    
    sub_kw = None
    zone_name = None

    for zone_dir_name, files in zone_files.items():
        zone_name = zone_dir_name
        
        for file in files:
            df = pd.read_csv(file)
            if not {"yearly", "kw"}.issubset(df.columns):
                raise ValueError(f"{file} missing required columns")

            df_com = df[df["yearly"].astype(str).str.startswith("com", na=False)]
            if df_com.empty:
                continue

            loadshape_dir = (
                com_root
                / zone_name
                / "scenarios"
                / "base_timeseries"
                / "opendss"
                / "opendss_loadshape_files"
            )
            if not loadshape_dir.exists():
                raise FileNotFoundError(f"Missing loadshape dir: {loadshape_dir}")

            for _, row in df_com.iterrows():
                shape_file = loadshape_dir / (row["yearly"] + ".csv")
                if not shape_file.exists():
                    raise FileNotFoundError(shape_file)

                ts_df = pd.read_csv(shape_file)
                num_cols = ts_df.select_dtypes(include="number").columns
                if len(num_cols) == 0:
                    raise ValueError(f"No numeric data in {shape_file}")

                ts = ts_df[num_cols[0]].astype(float).values  # Use numpy array
                scaled_kw = ts * float(row["kw"])

                if sub_kw is None:
                    sub_kw = scaled_kw
                else:
                    sub_kw += scaled_kw

    if sub_kw is None:
        return substation, None, 0

    # Convert kW -> MW
    sub_mw = sub_kw / 1000.0

    reference_len = len(sub_mw)

    # Pad 15-minute data to 35040 if needed (before hourly conversion)
    if 35000 <= reference_len <= 35100 and reference_len % 4 != 0:
        pad_length = 4 - (reference_len % 4)
        print(f"  {substation}: Padding {pad_length} intervals to make {reference_len} → {reference_len + pad_length}")
        # Repeat the last value instead of padding with zeros
        last_value = sub_mw[-1]
        sub_mw = np.append(sub_mw, np.full(pad_length, last_value))
        reference_len = len(sub_mw)

    if hourly_output:
        sub_mw_series = pd.Series(sub_mw)
        sub_mw = to_hourly_max(sub_mw_series).values

    return substation, sub_mw, reference_len


def add_commercial_profiles(
    output_dir: Path,
    com_root: Path,
    save_csv: bool = True,
    hourly_output: bool = True,
    n_workers: int = None,
):
    """
    Aggregate commercial demand profiles per substation and globally.

    - Iterate over feeder_summary CSVs
    - For 'com' rows, load OpenDSS loadshapes
    - Compute demand at 15-min resolution
    - Convert kW -> MW
    - Optionally aggregate to hourly (max of 4 intervals)
    - Save per-substation CSVs and a global Load_data file
    
    Args:
        n_workers: Number of parallel workers (default: CPU count - 1)
    """

    zone_pattern = re.compile(r"P\d+(R|U)$")
    out_dir = output_dir / "commercial_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: collect all substations and their files by zone
    substation_zone_files = {}
    
    for zone_dir in output_dir.iterdir():
        if not zone_dir.is_dir():
            continue
        if not zone_pattern.match(zone_dir.name):
            continue

        zone_name = zone_dir.name
        feeder_summaries_dir = zone_dir / "feeder_summaries"
        if not feeder_summaries_dir.exists():
            print(f"Skipping {zone_name}, no feeder_summaries")
            continue

        print(f"Collecting from zone: {zone_name}")

        for csv_file in feeder_summaries_dir.glob("feeder_summary*.csv"):
            m = re.search(r"feeder_summary_(.*?)--", csv_file.name)
            if m:
                substation = m.group(1)
                if substation not in substation_zone_files:
                    substation_zone_files[substation] = {}
                if zone_name not in substation_zone_files[substation]:
                    substation_zone_files[substation][zone_name] = []
                substation_zone_files[substation][zone_name].append(csv_file)

    if not substation_zone_files:
        print("No commercial load found")
        return

    print(f"\nProcessing {len(substation_zone_files)} substations...")
    
    # Set up parallel processing
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {n_workers} parallel workers")
    
    # Prepare data for parallel processing
    substation_items = list(substation_zone_files.items())
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_substation_commercial,
        output_dir=output_dir,
        com_root=com_root,
        zone_pattern=zone_pattern,
        hourly_output=hourly_output
    )
    
    # Process substations
    substation_series = {}
    reference_len = None
    zero_load_count = 0
    
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(process_func, substation_items)
        
        for substation, sub_mw, sub_len in results:
            if sub_mw is None:
                print(f"WARNING: {substation} has no commercial load")
                zero_load_count += 1
                continue
            
            if reference_len is None:
                reference_len = sub_len if not hourly_output else len(sub_mw)
            
            col_name = f"Load_MW_{substation}"
            substation_series[col_name] = sub_mw
    else:
        # Sequential processing
        for item in substation_items:
            substation, sub_mw, sub_len = process_func(item)
            if sub_mw is None:
                print(f"WARNING: {substation} has no commercial load")
                zero_load_count += 1
                continue
            
            if reference_len is None:
                reference_len = sub_len if not hourly_output else len(sub_mw)
            
            col_name = f"Load_MW_{substation}"
            substation_series[col_name] = sub_mw

    if not substation_series:
        print("No commercial load data generated")
        return

    print(f"\nCompleted processing {len(substation_series)} substations")
    print(f"Skipped {zero_load_count} substations with no commercial load")

    # Create DataFrame efficiently
    load_df = pd.DataFrame(substation_series)
    load_df["Load_MW_ALL_SUBSTATIONS"] = load_df.sum(axis=1)

    freq = "H" if hourly_output else "15min"
    periods = len(load_df)

    load_df.insert(
        0,
        "time",
        pd.date_range(
            start="2018-01-01 00:00:00",
            periods=periods,
            freq=freq,
        ),
    )

    if save_csv:
        # Save individual substation files
        for col in substation_series.keys():
            substation = col.replace("Load_MW_", "")
            out_df = pd.DataFrame({
                "time": load_df["time"],
                col: load_df[col]
            })
            out_df.to_csv(
                out_dir / f"com_profile_{substation}.csv", 
                index=False
            )
        
        # Save global file
        global_file = out_dir / "com_data_6716_bus.csv"
        load_df.to_csv(global_file, index=False)
        print(f"Saved global load file: {global_file}")
        print(f"  Total rows: {len(load_df)}")
        print(f"  Load columns: {len(substation_series)}")


def combine_residential_and_commercial(
    output_dir: Path,
    save_csv: bool = True,
):
    """
    Combine residential and commercial load profiles into a single file.
    
    - Reads Load_data_6716_bus.csv (residential)
    - Reads com_data_6716_bus.csv (commercial)
    - Matches substations and sums loads
    - Saves combined Load_data_6716_bus_combined.csv
    """
    
    res_profiles_dir = output_dir / "residential_profiles"
    com_profiles_dir = output_dir / "commercial_profiles"
    
    res_file = res_profiles_dir / "Load_data_6716_bus.csv"
    com_file = com_profiles_dir / "com_data_6716_bus.csv"
    
    if not res_file.exists():
        raise FileNotFoundError(f"Residential load file not found: {res_file}")
    
    if not com_file.exists():
        raise FileNotFoundError(f"Commercial load file not found: {com_file}")
    
    print("Loading residential load data...")
    res_df = pd.read_csv(res_file)
    
    print("Loading commercial load data...")
    com_df = pd.read_csv(com_file)
    
    # Get load columns (those starting with Load_MW_)
    res_load_cols = [col for col in res_df.columns if col.startswith("Load_MW_")]
    com_load_cols = [col for col in com_df.columns if col.startswith("Load_MW_") and col != "Load_MW_ALL_SUBSTATIONS"]
    
    # Get metadata columns from residential file (everything before load columns)
    metadata_cols = [col for col in res_df.columns if not col.startswith("Load_MW_")]
    
    print(f"Found {len(res_load_cols)} residential load columns")
    print(f"Found {len(com_load_cols)} commercial load columns")
    
    # Check if lengths match
    if len(res_df) != len(com_df):
        raise ValueError(
            f"Row count mismatch: residential has {len(res_df)} rows, "
            f"commercial has {len(com_df)} rows"
        )
    
    # Start with metadata columns from residential
    combined_df = res_df[metadata_cols].copy()
    
    # Get all unique substations from both datasets
    res_substations = {col.replace("Load_MW_", "") for col in res_load_cols}
    com_substations = {col.replace("Load_MW_", "") for col in com_load_cols}
    all_substations = sorted(res_substations | com_substations)
    
    print(f"Processing {len(all_substations)} unique substations...")
    
    # Combine loads for each substation
    for substation in all_substations:
        col_name = f"Load_MW_{substation}"
        
        res_col = col_name if col_name in res_load_cols else None
        com_col = col_name if col_name in com_load_cols else None
        
        if res_col and com_col:
            # Both residential and commercial load
            combined_df[col_name] = res_df[res_col] + com_df[com_col]
        elif res_col:
            # Only residential load
            combined_df[col_name] = res_df[res_col]
        elif com_col:
            # Only commercial load
            combined_df[col_name] = com_df[com_col]
    
    print(f"Combined {len(all_substations)} substations")
    print(f"  Residential only: {len(res_substations - com_substations)}")
    print(f"  Commercial only: {len(com_substations - res_substations)}")
    print(f"  Both res + com: {len(res_substations & com_substations)}")
    
    if save_csv:
        combined_file = res_profiles_dir / "Load_data_6716_bus_combined.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\nSaved combined load file: {combined_file}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Load columns: {len(all_substations)}")
    
    return combined_df


if __name__ == "__main__":
    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]

    output_dir = parent_dir / "out"
    com_dir = Path(r"F:\full_texas")

    # Process commercial profiles
    add_commercial_profiles(
        output_dir=output_dir,
        com_root=com_dir,
        save_csv=True,
        hourly_output=False,
        n_workers=None,  # Auto-detect CPU cores
    )
    
    # Combine residential and commercial
    print("\n" + "="*60)
    print("Combining residential and commercial loads...")
    print("="*60 + "\n")
    
    combine_residential_and_commercial(
        output_dir=output_dir,
        save_csv=True,
    )