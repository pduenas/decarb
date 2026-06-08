from pathlib import Path
import pandas as pd
import re

def add_commercial_profiles(output_dir: Path, com_root: Path, save_csv: bool = True):
    """
    Aggregate commercial demand profiles per substation and globally.

    - Iterate over feeder_summary CSVs exactly like the first script.
    - For 'com' rows, load the corresponding timeseries from com_root/<zone>/scenarios/.../opendss_loadshape_files
    - Convert kW -> MW
    - Add 15-min timestamps starting 2018-01-01 00:00
    - Save per-substation CSVs and a global Load_data_6716_bus.csv
    """

    zone_pattern = re.compile(r"P\d+(R|U)$")
    out_dir = output_dir / "commercial_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    substation_series = {}
    reference_len = None

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

        print(f"Processing zone: {zone_name}")

        # Map substation -> list of feeder_summary files
        substation_files = {}
        for csv_file in feeder_summaries_dir.glob("feeder_summary*.csv"):
            m = re.search(r"feeder_summary_(.*?)--", csv_file.name)
            if m:
                substation_files.setdefault(m.group(1), []).append(csv_file)

        for substation, files in substation_files.items():
            print(f"  Processing substation: {substation}")
            sub_kw = None

            # Iterate over each feeder_summary for this substation
            for file in files:
                df = pd.read_csv(file)
                if not {"yearly", "kw"}.issubset(df.columns):
                    raise ValueError(f"{file} missing required columns")

                # Commercial rows only
                df_com = df[df["yearly"].astype(str).str.startswith("com", na=False)]
                if df_com.empty:
                    continue

                # Dynamically build loadshape path for this zone
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

                # Iterate over commercial rows
                for _, row in df_com.iterrows():
                    shape_file = loadshape_dir / (row["yearly"]+".csv")
                    if not shape_file.exists():
                        raise FileNotFoundError(shape_file)

                    ts_df = pd.read_csv(shape_file)
                    num_cols = ts_df.select_dtypes(include="number").columns
                    if len(num_cols) == 0:
                        raise ValueError(f"No numeric data in {shape_file}")

                    ts = ts_df[num_cols[0]].astype(float)
                    scaled_kw = ts * float(row["kw"])

                    if sub_kw is None:
                        sub_kw = scaled_kw.copy()
                    else:
                        sub_kw = sub_kw.add(scaled_kw, fill_value=0.0)

            if sub_kw is None:
                if reference_len is None:
                    print(f"WARNING: {substation} has no commercial load")
                    continue
                sub_kw = pd.Series(0.0, index=range(reference_len))

            if reference_len is None:
                reference_len = len(sub_kw)
            elif len(sub_kw) != reference_len:
                raise ValueError(f"Timeseries length mismatch at {substation}")

            # Convert kW -> MW
            sub_mw = sub_kw / 1000.0
            col_name = f"Load_MW_{substation}"
            substation_series[col_name] = sub_mw.reset_index(drop=True)

            # Save per-substation CSV
            if save_csv:
                time_index = pd.date_range(
                    start="2018-01-01 00:00:00",
                    periods=reference_len,
                    freq="15min"
                )
                out_df = pd.DataFrame({"time": time_index, col_name: sub_mw.reset_index(drop=True)})
                out_df.to_csv(out_dir / f"com_profile_{substation}.csv", index=False)

    if not substation_series:
        print("No commercial load found")
        return

    # Global aggregation
    load_df = pd.DataFrame(substation_series)
    load_df["Load_MW_ALL_SUBSTATIONS"] = load_df.sum(axis=1)
    load_df.insert(0, "time", pd.date_range(start="2018-01-01 00:00:00", periods=reference_len, freq="15min"))

    if save_csv:
        global_file = out_dir / "com_data_6716_bus.csv"
        load_df.to_csv(global_file, index=False)
        print(f"Saved global load file: {global_file}")


if __name__ == "__main__":
    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]

    input_dir = parent_dir / "in"
    output_dir = parent_dir / "out"

    com_dir = Path(r"F:\full_texas")

    add_commercial_profiles(output_dir=output_dir, com_root=com_dir, save_csv=True)
