import os
import re
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

# =======================================================
# PROCESS AND SUMMARIZE FEEDER SUMMARIES (PER ZONE)
# =======================================================
def process_and_summarize_feeder_summaries(
    zone_name, output_root, match_tag, save_in_zone_folder=False, zone_folder=None
):
    """
    Processes feeder summary files, saves one unique file per substation,
    creates aggregated zone-level files, and builds bus-to-building mapping.
    Returns zone-level summary stats: total_buses, unique_bldgs
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
        feeder_summaries_folder = os.path.join(base_path, "feeder_summaries")
    else:
        print(f"❌ Error: Path does not exist — {base_path}")
        return None

    print(f"Processing feeder summaries for zone: {zone_name}")
    print(f"Output directory for building IDs: {bldg_ids_folder}\n")

    feeder_files = [
        f for f in os.listdir(feeder_summaries_folder)
        if f.lower().startswith("feeder_summary") and f.lower().endswith((".csv", ".xlsx"))
    ]
    if not feeder_files:
        print(f"❌ No feeder summary files found in {feeder_summaries_folder}")
        return None

    all_ids_counter = {}
    all_mapping_records = []
    total_buses = 0

    for feeder_file in feeder_files:
        feeder_path = os.path.join(feeder_summaries_folder, feeder_file)
        feeder_name = re.sub(r'^feeder_summary_|\.csv$|\.xlsx$', '', feeder_file)
        match = re.search(r"feeder_summary_(.*?)--", feeder_file)
        substation = match.group(1) if match else "unknown"

        try:
            df = (
                pd.read_csv(feeder_path, low_memory=False)
                if feeder_file.lower().endswith(".csv")
                else pd.read_excel(feeder_path)
            )

            # Validate columns
            if match_tag not in df.columns:
                print(f"⚠️ Missing column '{match_tag}' in {feeder_file}")
                continue
            if "Name" not in df.columns:
                print(f"⚠️ Missing column 'Name' in {feeder_file}")
                continue

            # Filter residential
            residential_df = df[df["yearly"].str.startswith("res", na=False)]
            if residential_df.empty:
                print(f"⚠️ No residential data found in {feeder_file}. Skipping...")
                continue

            # --- Mapping: bus <-> building ID
            mapping_df = residential_df[["Name", match_tag]].dropna(subset=[match_tag]).copy()
            mapping_df[match_tag] = mapping_df[match_tag].astype(str).str.strip()  # Convert to string and strip spaces
            # Attempt to cast the building IDs to integers
            try:
                mapping_df[match_tag] = mapping_df[match_tag].astype(int)  # Cast the building ID column to integers
            except ValueError as e:
                print(f"⚠️ Error casting '{match_tag}' to integer: {e}")
                # In case casting fails, filter out rows with non-numeric building IDs
                mapping_df = mapping_df[pd.to_numeric(mapping_df[match_tag], errors='coerce').notna()]
                mapping_df[match_tag] = mapping_df[match_tag].astype(float).astype(int)


            mapping_df.insert(0, "feeder", feeder_name)
            mapping_df.rename(columns={"Name": "bus_name", match_tag: "bldg_id"}, inplace=True)
            all_mapping_records.append(mapping_df)
            total_buses += len(mapping_df)

            # --- Count matches
            vc = mapping_df["bldg_id"].value_counts()
            for bldg_id, count in vc.items():
                all_ids_counter[bldg_id] = all_ids_counter.get(bldg_id, 0) + count

            print(f"  ✅ Processed {feeder_file} — {len(mapping_df)} residential buses")

        except Exception as e:
            print(f"❌ Error processing {feeder_file}: {e}")

    # --- Combine and save mapping
    if all_mapping_records:
        mapping_df = pd.concat(all_mapping_records, ignore_index=True)
        mapping_file = os.path.join(bldg_ids_folder, f"{zone_name}_bldg_id_mapping.csv")
        mapping_df.to_csv(mapping_file, index=False)
        print(f"💾 Saved bus-to-building mapping: {mapping_file}")

    # --- Zone-level unique building IDs
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
        return {"zone": zone_name, "total_buses": total_buses, "unique_bldgs": unique_bldgs}
    else:
        print(f"⚠️ No residential building IDs found for zone: {zone_name}.")
        return {"zone": zone_name, "total_buses": total_buses, "unique_bldgs": 0}


# =======================================================
# AGGREGATE ALL ZONE SUMMARIES
# =======================================================
def summarize_all_zones_old(output_root, match_tag):
    """
    Reads all zone-level summaries and produces one combined summary CSV
    with total unique building IDs and total buses across all zones.
    """
    print("\n📊 Aggregating results across all zones...\n")

    zone_dirs = [d for d in Path(output_root).iterdir() if d.is_dir()]
    all_zone_summaries = []
    all_building_ids = set()
    total_buses_all_zones = 0

    for zone_dir in zone_dirs:
        bldg_ids_file = zone_dir / "bldg_ids" / f"{zone_dir.name}_bldg_ids_all.csv"
        mapping_file = zone_dir / "bldg_ids" / f"{zone_dir.name}_bldg_id_mapping.csv"

        if not bldg_ids_file.exists():
            continue

        try:
            df_zone = pd.read_csv(bldg_ids_file)
            all_building_ids.update(df_zone[match_tag].astype(str).unique())

            if mapping_file.exists():
                df_mapping = pd.read_csv(mapping_file)
                num_buses = len(df_mapping)
            else:
                num_buses = df_zone["num_matches"].sum()

            all_zone_summaries.append({
                "zone": zone_dir.name,
                "unique_bldgs": len(df_zone),
                "total_buses": num_buses
            })
            total_buses_all_zones += num_buses

        except Exception as e:
            print(f"⚠️ Error reading {bldg_ids_file}: {e}")

    # Save master summary
    summary_df = pd.DataFrame(all_zone_summaries)
    summary_df["match_ratio_%"] = (summary_df["unique_bldgs"] / summary_df["total_buses"] * 100).round(2)
    master_summary_path = Path(output_root) / "all_zones_summary.csv"
    summary_df.to_csv(master_summary_path, index=False)

    print(f"💾 Saved all-zones summary: {master_summary_path}")
    print(f"🌍 Total unique buildings across all zones: {len(all_building_ids)}")
    print(f"🚏 Total buses across all zones: {total_buses_all_zones}\n")

    return summary_df

def summarize_all_zones(output_root, match_tag):
    """
    Reads all zone-level summaries and produces one combined summary CSV
    with total unique building IDs and total buses across all zones.
    Also saves a file with all unique building IDs and their usage counts.
    """
    print("\n📊 Aggregating results across all zones...\n")

    zone_dirs = [d for d in Path(output_root).iterdir() if d.is_dir()]
    all_zone_summaries = []
    building_id_counts = {}  # Track count of each building ID across all zones
    total_buses_all_zones = 0

    for zone_dir in zone_dirs:
        bldg_ids_file = zone_dir / "bldg_ids" / f"{zone_dir.name}_bldg_ids_all.csv"
        mapping_file = zone_dir / "bldg_ids" / f"{zone_dir.name}_bldg_id_mapping.csv"

        if not bldg_ids_file.exists():
            continue

        try:
            df_zone = pd.read_csv(bldg_ids_file)
            
            # Count occurrences of each building ID
            for bldg_id in df_zone[match_tag].astype(float).astype(int).unique():
                building_id_counts[bldg_id] = building_id_counts.get(bldg_id, 0) + 1

            if mapping_file.exists():
                df_mapping = pd.read_csv(mapping_file)
                num_buses = len(df_mapping)
            else:
                num_buses = df_zone["num_matches"].sum()

            all_zone_summaries.append({
                "zone": zone_dir.name,
                "unique_bldgs": len(df_zone),
                "total_buses": num_buses
            })
            total_buses_all_zones += num_buses

        except Exception as e:
            print(f"⚠️ Error reading {bldg_ids_file}: {e}")

    # Save master summary
    summary_df = pd.DataFrame(all_zone_summaries)
    summary_df["match_ratio_%"] = (summary_df["unique_bldgs"] / summary_df["total_buses"] * 100).round(2)
    master_summary_path = Path(output_root) / "all_zones_summary.csv"
    summary_df.to_csv(master_summary_path, index=False)

    # Save building ID usage counts across all zones
    building_usage_df = pd.DataFrame([
        {"bldg_id": bldg_id, "zones_used_in": count}
        for bldg_id, count in building_id_counts.items()
    ]).sort_values("zones_used_in", ascending=False)
    
    building_usage_path = Path(output_root) / "all_zones_bldg_count.csv"
    building_usage_df.to_csv(building_usage_path, index=False)

    print(f"💾 Saved all-zones summary: {master_summary_path}")
    print(f"💾 Saved building ID usage: {building_usage_path}")
    print(f"🌍 Total unique buildings across all zones: {len(building_id_counts)}")
    print(f"🚏 Total buses across all zones: {total_buses_all_zones}\n")

    return summary_df


# =======================================================
# DOWNLOAD BUILDING CONSUMPTION FILES
# =======================================================
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
            continue

        try:
            s3.head_object(Bucket=bucket_name, Key=key)
        except Exception as e:
            if hasattr(e, "response") and e.response["Error"]["Code"] == "404":
                missing.append(bldg_id)
                continue

        try:
            s3.download_file(bucket_name, key, save_path)
            downloaded += 1
        except Exception as e:
            missing.append(bldg_id)

    total = len(all_ids)
    print(f"\n📦 Downloaded {downloaded}/{total} files. Missing {len(missing)}.\n")
    if missing:
        missing_file = os.path.join(bldg_consumption_folder, f"{zone_name}_missing_files.csv")
        pd.DataFrame({"missing_bldg_id": missing}).to_csv(missing_file, index=False)
        print(f"💾 Saved missing list: {missing_file}")

    print(f"✅ Finished downloading consumption files for zone: {zone_name}")

def download_building_consumption_files_all_zones(output_root, match_tag):
    """
    Download ALL building consumption parquet files across ALL zones.
    Saves files into a single folder: /out/consumption_files
    """

    consumption_folder = os.path.join(output_root, "consumption_files")
    os.makedirs(consumption_folder, exist_ok=True)

    print("\n🔍 Gathering building IDs from all zones...")

    # Find all zone directories
    zone_dirs = [d for d in Path(output_root).iterdir() if d.is_dir()]

    all_ids = set()

    # --- Collect building IDs from each zone ---
    for zone_dir in zone_dirs:
        bldg_ids_file = zone_dir / "bldg_ids" / f"{zone_dir.name}_bldg_ids_all.csv"
        if not bldg_ids_file.exists():
            continue

        try:
            df_ids = pd.read_csv(bldg_ids_file)
            if match_tag not in df_ids.columns:
                print(f"⚠️ Missing column '{match_tag}' in {bldg_ids_file}")
                continue

            ids = df_ids[match_tag].astype(str).unique()
            all_ids.update(ids)

        except Exception as e:
            print(f"⚠️ Error reading {bldg_ids_file}: {e}")

    all_ids = sorted(all_ids)
    print(f"🌍 Total unique building IDs across all zones: {len(all_ids)}\n")

    # --- Begin downloads ---
    print(f"⬇️ Downloading consumption files into: {consumption_folder}\n")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "oedi-data-lake"

    downloaded = 0
    missing = []

    for bldg_id in all_ids:
        key = f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1/timeseries_individual_buildings/by_state/upgrade=0/state=TX/{bldg_id}-0.parquet"
        save_path = os.path.join(consumption_folder, f"{bldg_id}-0.parquet")

        if os.path.exists(save_path):
            continue

        try:
            s3.head_object(Bucket=bucket_name, Key=key)
        except Exception as e:
            if hasattr(e, "response") and e.response["Error"]["Code"] == "404":
                missing.append(bldg_id)
                continue

        try:
            s3.download_file(bucket_name, key, save_path)
            downloaded += 1
        except Exception:
            missing.append(bldg_id)

    print(f"\n📦 Downloaded {downloaded}/{len(all_ids)} files. Missing {len(missing)}.\n")

    if missing:
        missing_file = os.path.join(consumption_folder, "missing_files.csv")
        pd.DataFrame({"missing_bldg_id": missing}).to_csv(missing_file, index=False)
        print(f"💾 Saved missing list: {missing_file}")

    print("✅ Finished downloading all consumption files.")


# =======================================================
# MAIN — PROCESS ALL ZONES + MASTER SUMMARY
# =======================================================
if __name__ == "__main__":
    match_tag = "nrel_climate_match"
    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]
    output_root = parent_dir / "out"
    zones_root = output_root

    # Find all zones to process
    zone_dirs = [d for d in zones_root.iterdir() if d.is_dir()]
    print(f"📍 Found {len(zone_dirs)} zones to process: {[z.name for z in zone_dirs]}")

    all_zone_stats = []
    for zone_dir in zone_dirs:
        zone_name = zone_dir.name
        zone_folder = zone_dir / "feeder_summaries"
        if not zone_folder.exists():
            print(f"⏭️ Skipping {zone_name}: no feeder_summaries folder found.")
            continue

        stats = process_and_summarize_feeder_summaries(
            zone_name, output_root, match_tag, zone_folder=zone_folder
        )
        if stats:
            all_zone_stats.append(stats)

    # --- Create combined summary for all zones ---
    if all_zone_stats:
        summarize_all_zones(output_root, match_tag)

        # --- Download all consumption files at once ---
        # download_building_consumption_files_all_zones(output_root, match_tag)

    else:
        print("⚠️ No zones were successfully processed.")