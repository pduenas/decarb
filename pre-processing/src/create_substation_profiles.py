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

    # Global output directory
    output_dir = root_dir / "residential_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON lookup file
    lookup_file = output_dir / "sub_bldg_lookup.json"

    # Load existing lookup if it exists and is valid, otherwise start empty
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

        # Map: substation -> list of feeder files
        substation_files = {}

        for csv_file in feeder_summaries_dir.glob("feeder_summary*.csv"):
            # feeder_summary_<substation>--<feeder>.csv
            match = re.search(r"feeder_summary_(.*?)--", csv_file.name)
            if not match:
                continue

            substation = match.group(1)
            substation_files.setdefault(substation, []).append(csv_file)

        # Process each substation in this zone
        for substation, files in substation_files.items():
            output_file = output_dir / "mapping_files" / f"res_profile_{substation}.csv"

            # Skip only when CSV saving is enabled
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
                    raise ValueError(
                        f"Missing columns {missing} in {file}"
                    )

                # Keep only residential rows
                df = df[df["yearly"].astype(str).str.startswith("res", na=False)]

                # Skip this feeder if no residential load
                if df.empty:
                    continue

                vc = df["nrel_climate_match"].dropna().value_counts()

                for bldg_id, count in vc.items():
                    bldg_id = int(bldg_id)
                    counts[bldg_id] = counts.get(bldg_id, 0) + int(count)

            # Save CSV only if requested
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

                out_df["nrel_climate_match"] = out_df["nrel_climate_match"].astype(int)
                out_df["count"] = out_df["count"].astype(int)

                out_df.to_csv(output_file, index=False)
                print(f"    Saved: {output_file}")

            # Update JSON lookup for this substation
            sub_bldg_lookup[substation] = {
                str(k): int(v) for k, v in counts.items()
            }

            # Persist lookup after each substation
            with open(lookup_file, "w") as f:
                json.dump(sub_bldg_lookup, f, indent=2)


def create_load_data_csv(lookup_json, parquet_root, output_csv):
    """
    Create Load_data_6716_bus.csv by aggregating substation loads.

    For each substation:
      - Read building parquet(s) by climate ID
      - Element-wise sum energy (kWh) scaled by building count
      - Convert 15-min kWh → MW: (kWh * 4) / 1000
      - Save as column: Load_MW_<substation>
    """

    with open(lookup_json, "r") as f:
        sub_bldg_lookup = json.load(f)

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

        # Convert 15-minute kWh → MW
        if substation_kwh is None:
            if reference_len is None:
                raise RuntimeError(
                    "No parquet data found to determine time series length"
                )

            print(
                f"WARNING: Substation '{substation}' has no residential load. "
                "Filling load time series with zeros."
            )

            zero_load_substations.append(
                {
                    "substation": substation,
                    "reason": "no_residential_load",
                }
            )

            substation_mw = pd.Series(0.0, index=range(reference_len))
        else:
            substation_mw = (substation_kwh * 4.0) / 1000.0

        substation_series[f"Load_MW_{substation}"] = substation_mw.reset_index(drop=True)

    # Combine all substations into one DataFrame
    load_df = pd.DataFrame(substation_series)

    load_df.to_csv(output_csv, index=False)
    print(f"Saved load data to: {output_csv}")

    # Write zero-load summary report
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
    # consumption_path = output_dir / "consumption_files"
    # lookup_json = output_dir / "residential_profiles" / "sub_bldg_lookup.json"
    # load_csv = output_dir / "residential_profiles" / "Load_data_6716_bus.csv"

    # create_load_data_csv(lookup_json, consumption_path, load_csv)
