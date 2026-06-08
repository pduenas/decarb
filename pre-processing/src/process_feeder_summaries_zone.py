import os
import pandas as pd
import re
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

# ----------------------------
# PROCESS AND SUMMARIZE FEEDER SUMMARIES
# ----------------------------
def process_and_summarize_feeder_summaries_old(zone_name, output_root, match_tag, save_in_zone_folder=False, zone_folder=None):
    """
    Processes feeder summary files, saves one unique file per substation,
    and creates aggregated zone-level files.
    """
    base_path = fr"D:\shared\ercot_project\{zone_name}\scenarios\base_timeseries\opendss"

    # Decide output directory for building IDs
    if save_in_zone_folder:
        bldg_ids_folder = os.path.join(output_root, f"{zone_name}_all_bldg_ids")
    else:
        bldg_ids_folder = os.path.join(output_root, zone_name, "bldg_ids")
    os.makedirs(bldg_ids_folder, exist_ok=True)

    # Decide where to look for feeder summaries
    if zone_folder:
        print(f"📂 Using folder: {zone_folder}")
        feeder_summaries_folder = zone_folder
    elif os.path.exists(base_path):
        feeder_summaries_folder = os.path.join(base_path, zone_name, "feeder_summaries")
    else:
        print(f"❌ Error: Path does not exist — {base_path}")
        return

    print(f"Processing feeder summaries for zone: {zone_name}")
    print(f"Output directory for building IDs: {bldg_ids_folder}\n")

    # List all files starting with "feeder_summary" in the feeder_summaries folder
    feeder_files = [
        f for f in os.listdir(feeder_summaries_folder)
        if f.lower().startswith("feeder_summary") and f.lower().endswith((".csv", ".xlsx"))
    ]
    
    if not feeder_files:
        print(f"❌ No feeder summary files found in {feeder_summaries_folder}")
        return

    # To store all residential IDs across all feeders for zone-level summary
    all_ids_counter = {}

    # Process each feeder summary file
    for feeder_file in feeder_files:
        feeder_path = os.path.join(feeder_summaries_folder, feeder_file)
        
        # Extract the substation name using regex (text between the first underscore and first dash)
        match = re.search(r"feeder_summary_(.*?)--", feeder_file)
        if match:
            substation = match.group(1)
        else:
            print(f"❌ Could not extract substation from filename: {feeder_file}")
            continue

        # print(f"📂 Processing {feeder_file} for substation: {substation}")

        try:
            # Load the feeder summary file
            df = pd.read_csv(feeder_path) if feeder_file.lower().endswith(".csv") else pd.read_excel(feeder_path)

            # Ensure the match_tag column exists
            if match_tag not in df.columns:
                print(f"⚠️ Missing required column '{match_tag}' in {feeder_file}")
                continue

            # Apply residential filtering based on 'yearly' column starting with "res"
            residential_df = df[df["yearly"].str.startswith("res", na=False)]

            if residential_df.empty:
                print(f"⚠️ No residential data found in {feeder_file}. Skipping...")
                continue

            # Count the occurrences of each building ID in the match_tag column for residential data
            match_counts = residential_df[match_tag].value_counts().reset_index()
            match_counts.columns = [match_tag, "num_match"]

            # Add feeder filename column for identification
            match_counts.insert(0, "feeder", feeder_file)

            # Save the result for each substation (one file per substation)
            output_file = os.path.join(bldg_ids_folder, f"{substation}_bldg_id.csv")
            match_counts.to_csv(output_file, index=False)
            print(f"  ✅ Processed {feeder_file} — {len(match_counts)} residential IDs")
            # print(f"    💾 Saved: {output_file}")

            # Collect residential data for zone-level aggregation
            for bldg_id, count in residential_df[match_tag].value_counts().items():
                all_ids_counter[bldg_id] = all_ids_counter.get(bldg_id, 0) + count

        except Exception as e:
            print(f"❌ Error processing {feeder_file}: {e}")

    # Save the zone-level combined CSV: <zone>_bldg_id.csv (combined data for all feeders)
    if all_ids_counter:
        zone_bldg_ids_df = pd.DataFrame(sorted(all_ids_counter.items()), columns=[match_tag, "num_match"])
        zone_bldg_ids_file = os.path.join(bldg_ids_folder, f"{zone_name}_bldg_id.csv")
        zone_bldg_ids_df.to_csv(zone_bldg_ids_file, index=False)
        print(f"💾 Saved zone-level building IDs: {zone_bldg_ids_file}")

        # Save the zone-level file with aggregated unique building IDs: <zone>_bldg_ids_all.csv
        zone_ids_df = pd.DataFrame(sorted(all_ids_counter.items()), columns=[match_tag, "num_matches"])
        zone_ids_all_file = os.path.join(bldg_ids_folder, f"{zone_name}_bldg_ids_all.csv")
        zone_ids_df.to_csv(zone_ids_all_file, index=False)
        print(f"💾 Saved zone-level unique building IDs: {zone_ids_all_file}")
        print(f"✅ Total unique building IDs in zone: {len(all_ids_counter)}")
    else:
        print(f"⚠️ No residential building IDs found for zone: {zone_name}.")
    

def process_and_summarize_feeder_summaries(zone_name, output_root, match_tag, save_in_zone_folder=False, zone_folder=None):
    """
    Processes feeder summary files, saves one unique file per substation,
    creates aggregated zone-level files, and builds bus-to-building mapping.
    """
    base_path = fr"D:\shared\ercot_project\{zone_name}\scenarios\base_timeseries\opendss"

    # Decide output directory for building IDs
    if save_in_zone_folder:
        bldg_ids_folder = os.path.join(output_root, f"{zone_name}_all_bldg_ids")
    else:
        bldg_ids_folder = os.path.join(output_root, zone_name, "bldg_ids")
    os.makedirs(bldg_ids_folder, exist_ok=True)

    # Decide where to look for feeder summaries
    if zone_folder:
        print(f"📂 Using folder: {zone_folder}")
        feeder_summaries_folder = zone_folder
    elif os.path.exists(base_path):
        feeder_summaries_folder = os.path.join(base_path, "feeder_summaries")  # ✅ fixed path
    else:
        print(f"❌ Error: Path does not exist — {base_path}")
        return

    print(f"Processing feeder summaries for zone: {zone_name}")
    print(f"Output directory for building IDs: {bldg_ids_folder}\n")

    feeder_files = [
        f for f in os.listdir(feeder_summaries_folder)
        if f.lower().startswith("feeder_summary") and f.lower().endswith((".csv", ".xlsx"))
    ]
    if not feeder_files:
        print(f"❌ No feeder summary files found in {feeder_summaries_folder}")
        return

    all_ids_counter = {}
    all_mapping_records = []  # 🆕 stores (feeder, bus_name, bldg_id)
    total_buses = 0

    for feeder_file in feeder_files:
        feeder_path = os.path.join(feeder_summaries_folder, feeder_file)
        feeder_name = re.sub(r'^feeder_summary_|\.csv$|\.xlsx$', '', feeder_file)

        # Extract substation name
        match = re.search(r"feeder_summary_(.*?)--", feeder_file)
        substation = match.group(1) if match else "unknown"

        try:
            df = pd.read_csv(feeder_path, low_memory=False) if feeder_file.lower().endswith(".csv") else pd.read_excel(feeder_path)

            # Validate presence of columns
            if match_tag not in df.columns:
                print(f"⚠️ Missing required column '{match_tag}' in {feeder_file}")
                continue
            if "Name" not in df.columns:
                print(f"⚠️ Missing required column 'Name' in {feeder_file}")
                continue

            # Filter to residential only
            residential_df = df[df["yearly"].str.startswith("res", na=False)]
            if residential_df.empty:
                print(f"⚠️ No residential data found in {feeder_file}. Skipping...")
                continue

            # --- 🆕 build mapping ---
            mapping_df = residential_df[["Name", match_tag]].dropna(subset=[match_tag]).copy()
            mapping_df.insert(0, "feeder", feeder_name)
            mapping_df.rename(columns={"Name": "bus_name", match_tag: "bldg_id"}, inplace=True)
            all_mapping_records.append(mapping_df)
            total_buses += len(mapping_df)

            # --- build counts for zone summary ---
            vc = residential_df[match_tag].value_counts()
            for bldg_id, count in vc.items():
                all_ids_counter[bldg_id] = all_ids_counter.get(bldg_id, 0) + count

            print(f"  ✅ Processed {feeder_file} — {len(mapping_df)} residential buses")

        except Exception as e:
            print(f"❌ Error processing {feeder_file}: {e}")

    # --- 🆕 Combine and save mapping file ---
    if all_mapping_records:
        mapping_df = pd.concat(all_mapping_records, ignore_index=True)
        mapping_file = os.path.join(bldg_ids_folder, f"{zone_name}_bldg_id_mapping.csv")
        mapping_df.to_csv(mapping_file, index=False)
        print(f"💾 Saved bus-to-building mapping: {mapping_file}")

    # --- Zone-level unique building IDs ---
    if all_ids_counter:
        zone_ids_df = pd.DataFrame(sorted(all_ids_counter.items()), columns=[match_tag, "num_matches"])
        zone_ids_all_file = os.path.join(bldg_ids_folder, f"{zone_name}_bldg_ids_all.csv")
        zone_ids_df.to_csv(zone_ids_all_file, index=False)
        print(f"💾 Saved zone-level unique building IDs: {zone_ids_all_file}")

        unique_bldgs = len(all_ids_counter)
        percent = 100 * unique_bldgs / total_buses if total_buses else 0
        print(f"\n✅ Zone {zone_name} summary:")
        print(f"   Total buses processed: {total_buses}")
        print(f"   Unique building IDs matched: {unique_bldgs}")
        print(f"   Match ratio: {percent:.3g}%\n")
    else:
        print(f"⚠️ No residential building IDs found for zone: {zone_name}.")

# ----------------------------------------
# DOWNLOAD BUILDING CONSUMPTION FILES
# ----------------------------------------
def download_building_consumption_files(zone_name, output_root, match_tag):
    """
    Download building consumption parquet files for a given zone.
    Uses the zone-level building ID CSV: <zone>_bldg_ids_all.csv
    """
    bldg_ids_folder = os.path.join(output_root, zone_name, "bldg_ids")
    bldg_consumption_folder = os.path.join(output_root, zone_name, "bldg_consumption_files")
    os.makedirs(bldg_consumption_folder, exist_ok=True)

    zone_ids_file = os.path.join(bldg_ids_folder, f"{zone_name}_bldg_ids_all.csv")
    if not os.path.exists(zone_ids_file):
        print(f"❌ Zone-level building ID file not found: {zone_ids_file}")
        return

    # Load unique building IDs
    df_ids = pd.read_csv(zone_ids_file)
    if match_tag not in df_ids.columns:
        print(f"❌ Column '{match_tag}' not found in {zone_ids_file}")
        return

    all_ids = df_ids[match_tag].astype(str).unique()
    print(f"🔍 Found {len(all_ids)} unique building IDs in zone: {zone_name}\n")
    print(f"Downloading consumption files to: {bldg_consumption_folder}\n")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "oedi-data-lake"

    downloaded = 0
    missing = []

    for bldg_id in sorted(all_ids):
        key = f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1/timeseries_individual_buildings/by_state/upgrade=0/state=TX/{bldg_id}-0.parquet"
        save_path = os.path.join(bldg_consumption_folder, f"{bldg_id}-0.parquet")

        if os.path.exists(save_path):
            print(f"  ⏭️ Skipping {bldg_id} — already downloaded.")
            continue

        try:
            s3.head_object(Bucket=bucket_name, Key=key)
        except Exception as e:
            if e.response["Error"]["Code"] == "404":
                print(f"  ❌ Missing: {bldg_id}-0.parquet")
                missing.append(bldg_id)
                continue
                
        try:
            s3.download_file(bucket_name, key, save_path)
            downloaded += 1
            print(f"  ✅ Downloaded {bldg_id}-0.parquet")
        except Exception as e:
            print(f"  ⚠️ Failed to download {bldg_id}: {e}")
            missing.append(bldg_id)

    # Summary and save missing list
    total = len(all_ids)
    print(f"\n📦 Downloaded {downloaded}/{total} files. Missing {len(missing)}.\n")

    if missing:
        missing_file = os.path.join(bldg_consumption_folder, f"{zone_name}_missing_files.csv")
        pd.DataFrame({"missing_bldg_id": missing}).to_csv(missing_file, index=False)
        print(f"💾 Saved list of missing building files: {missing_file}")

    print(f"✅ Finished downloading consumption files for zone: {zone_name}")


# ----------------------------
# MAIN FUNCTION
# ----------------------------
if __name__ == "__main__":
    zone_name = "P1R"
    match_tag = "nrel_climate_match"
    
    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]
    output_root = parent_dir / "out"

    # Optionally specify a custom folder for feeder summaries
    zone_folder = output_root / zone_name / "feeder_summaries"  # None if using default

    # Process and summarize feeder summaries
    process_and_summarize_feeder_summaries(zone_name, output_root, match_tag, zone_folder=zone_folder)

    # Download building consumption parquet files
    # download_building_consumption_files(zone_name, output_root, match_tag)
