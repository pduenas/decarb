from pathlib import Path
import pandas as pd
import re
import json


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


def create_load_data_csv(
    lookup_json,
    parquet_root,
    output_csv,
    pretext_csv,
    hourly_output=True,
):
    """
    Create Load_data_6716_bus.csv by aggregating substation loads.

    - Uses metadata structure from genx_demand_data_pretext.csv (4 rows)
    - Computes demand at 15-min resolution
    - Optionally aggregates to hourly (max of 4 intervals)
    - Outputs 8760 rows with metadata in first rows, empty cells below
    """

    with open(lookup_json, "r") as f:
        sub_bldg_lookup = json.load(f)

    # Read the pretext file (contains metadata structure, e.g., 4 rows)
    pretext_df = pd.read_csv(pretext_csv)
    num_metadata_rows = len(pretext_df)

    substation_series = {}
    zero_load_substations = []
    reference_len = None

    for substation, climate_counts in sub_bldg_lookup.items():
        print(f"Aggregating load for substation: {substation}")

        substation_kwh = None

        for bldg_id_str, count in climate_counts.items():
            bldg_id = int(bldg_id_str)
            count = int(count)

            parquet_file = parquet_root / f"{bldg_id}-0.parquet"
            if not parquet_file.exists():
                raise FileNotFoundError(f"Missing parquet: {parquet_file}")

            df = pd.read_parquet(parquet_file)

            if reference_len is None:
                reference_len = len(df)

            col = "out.electricity.total.energy_consumption..kwh"
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {parquet_file}")

            energy_kwh = df[col].astype(float) * count

            if substation_kwh is None:
                substation_kwh = energy_kwh.copy()
            else:
                substation_kwh = substation_kwh.add(
                    energy_kwh, fill_value=0.0
                )

        if substation_kwh is None:
            print(
                f"WARNING: Substation '{substation}' has no residential load. "
                "Filling with zeros."
            )

            zero_load_substations.append(
                {"substation": substation, "reason": "no_residential_load"}
            )

            substation_mw = pd.Series(0.0, index=range(reference_len))
        else:
            substation_mw = (substation_kwh * 4.0) / 1000.0

        substation_mw = substation_mw.reset_index(drop=True)

        if hourly_output:
            substation_mw = to_hourly_max(substation_mw)

        substation_series[f"Load_MW_{substation}"] = substation_mw

    # Expected number of timesteps
    expected_timesteps = 8760 if hourly_output else reference_len

    # Create the load data columns (8760 rows)
    load_columns_df = pd.DataFrame(substation_series).reset_index(drop=True)
    
    if len(load_columns_df) != expected_timesteps:
        raise ValueError(
            f"Load data has {len(load_columns_df)} rows, "
            f"expected {expected_timesteps} timesteps"
        )

    # Determine which pretext columns should have values only in first N rows
    # vs. which should repeat (like Time_Index)
    metadata_cols = []
    repeating_cols = []
    
    for col in pretext_df.columns:
        # Check if column is meant to repeat (e.g., Time_Index, Sub_Weights)
        # These typically have sequential or repeating values
        if 'Time_Index' in col or 'Sub_Weights' in col or 'Timesteps' in col:
            repeating_cols.append(col)
        else:
            metadata_cols.append(col)
    
    # Create expanded pretext dataframe with 8760 rows
    pretext_expanded = pd.DataFrame(index=range(expected_timesteps))
    
    for col in pretext_df.columns:
        if col in metadata_cols:
            # Metadata columns: fill only first N rows, rest are empty
            pretext_expanded[col] = ''
            for i in range(min(num_metadata_rows, expected_timesteps)):
                pretext_expanded.loc[i, col] = pretext_df.loc[i, col]
        else:
            # Repeating columns: check if we need to generate values
            if col == 'Time_Index':
                # Generate sequential time index
                pretext_expanded[col] = range(1, expected_timesteps + 1)
            elif len(pretext_df[col].dropna()) == num_metadata_rows:
                # Column has values in all metadata rows, repeat the last value
                pretext_expanded[col] = pretext_df[col].iloc[-1]
            else:
                # Use the pattern from pretext if available
                pretext_expanded[col] = ''
                for i in range(min(num_metadata_rows, expected_timesteps)):
                    val = pretext_df.loc[i, col]
                    if pd.notna(val) and val != '':
                        pretext_expanded.loc[i, col] = val

    # Combine pretext and load data
    load_df = pd.concat(
        [
            pretext_expanded.reset_index(drop=True),
            load_columns_df.reset_index(drop=True),
        ],
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
        hourly_output=True,
    )
