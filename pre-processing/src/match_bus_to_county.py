import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_county_shapefile(shapefile_path):
    """
    Loads the Texas county shapefile and ensures EPSG:4326 projection.
    """
    counties = gpd.read_file(shapefile_path)
    counties = counties.to_crs(epsg=4326)
    return counties


def get_county_from_coordinates(lat, lon, counties):
    """
    Given lat/lon, return the (county_name, county_code) that contains the point.
    """
    if pd.isna(lat) or pd.isna(lon):
        return "Not found", None

    point = Point(lon, lat)
    possible = counties[counties.geometry.intersects(point)]

    if not possible.empty:
        county = possible.iloc[0]
        name = county['name']
        code = f"G{county['statefp']}0{county['countyfp']}0"
        return name, code

    return "Not found", None


def process_feeder(feeder_path, feeder_name, output_dir, counties):
    """
    Enrich Loads.csv with coordinates and county info and save to CSV.
    """
    loads_file = os.path.join(feeder_path, 'Loads.csv')
    buscoords_file = os.path.join(feeder_path, 'Buscoords.csv')

    if not os.path.exists(loads_file) or not os.path.exists(buscoords_file):
        print(f"⚠️ Missing Loads.csv or Buscoords.csv in {feeder_path}")
        return

    output_file = os.path.join(output_dir, f"feeder_summary_{feeder_name}.csv")
    os.makedirs(output_dir, exist_ok=True)

    loads_df = pd.read_csv(loads_file)
    buscoords_df = pd.read_csv(buscoords_file)

    # Rename buscoords column to avoid Name_x / Name_y
    buscoords_df = buscoords_df.rename(columns={'Name': 'Bus_Name'})

    # Match bus names
    loads_df['bus1_base'] = loads_df['bus1'].apply(lambda x: x.split('.')[0])
    merged_df = pd.merge(
        loads_df,
        buscoords_df[['Bus_Name', 'Longitude', 'Latitude']],
        left_on='bus1_base',
        right_on='Bus_Name',
        how='left'
    )

    # Add county info
    merged_df[['county', 'county_code']] = merged_df.apply(
        lambda row: pd.Series(get_county_from_coordinates(row['Latitude'], row['Longitude'], counties)),
        axis=1
    )

    # Drop helper column
    merged_df.drop(columns=['Bus_Name'], inplace=True)

    merged_df.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")


def process_all_substations(base_path, shapefile_path=None, save_location="local", out_dir=None):
    """
    Process all substations and feeders inside a given zone folder.
    """
    if shapefile_path is None:
        print("❌ Shapefile path not provided.")
        return

    counties = load_county_shapefile(shapefile_path)
    print(f"Loaded {len(counties)} counties from shapefile.")

    zone_path = base_path
    opendss_path = os.path.join(zone_path, 'scenarios', 'base_timeseries', 'opendss')
    print(f"Looking for OpenDSS data at: {opendss_path}")

    if not os.path.isdir(opendss_path):
        print(f"❌ Directory does not exist: {opendss_path}")
        return

    substations = [
        name for name in os.listdir(opendss_path)
        if os.path.isdir(os.path.join(opendss_path, name)) and name != 'opendss_loadshape_files'
    ]

    if not substations:
        print(f"⚠️ No substations found in {opendss_path}")
        return

    # Create zone-level folder for summaries if requested
    if save_location == "out_folder" and out_dir is not None:
        zone_summary_dir = os.path.join(out_dir, "feeder_summaries")
        os.makedirs(zone_summary_dir, exist_ok=True)
    else:
        zone_summary_dir = None

    for substation in substations:
        substation_path = os.path.join(opendss_path, substation)
        print(f"\n📂 Processing substation: {substation}")

        local_output_dir = os.path.join(substation_path, "feeder_summaries")
        os.makedirs(local_output_dir, exist_ok=True)

        feeders = [
            name for name in os.listdir(substation_path)
            if os.path.isdir(os.path.join(substation_path, name))
            and name != 'feeder_summaries'
        ]

        if not feeders:
            print(f"⚠️ No feeders found in {substation}")
            continue

        for feeder in feeders:
            feeder_path = os.path.join(substation_path, feeder)
            print(f"  🔄 Feeder: {feeder}")

            output_dir = zone_summary_dir if (save_location == "out_folder" and zone_summary_dir) else local_output_dir
            process_feeder(feeder_path, feeder, output_dir, counties)

    print(f"\n🎉 Finished processing all substations and feeders in zone: {os.path.basename(base_path)}")


def process_all_zones_parallel(base_folder, shapefile_path, save_location="out_folder", out_base=None, max_workers=None):
    """
    Processes all zone folders inside the given `base_folder` in parallel.
    """
    if not os.path.isdir(base_folder):
        print(f"❌ Base folder does not exist: {base_folder}")
        return

    zones = [z for z in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, z))]
    if not zones:
        print(f"⚠️ No zones found in {base_folder}")
        return

    print(f"📍 Found {len(zones)} zones to process: {', '.join(zones)}")
    print(f"🧵 Running in parallel using up to {max_workers or os.cpu_count()} workers...\n")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for zone in zones:
            zone_path = os.path.join(base_folder, zone)
            out_dir = os.path.join(out_base, zone) if out_base else None
            futures.append(executor.submit(
                process_all_substations,
                zone_path,
                shapefile_path,
                save_location,
                out_dir
            ))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error processing a zone: {e}")

    print("\n✅ All zones processed successfully!")


# Run all zones
if __name__ == "__main__":
    print("Starting full multi-zone parallel processing...")

    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]
    in_dir = parent_dir / "in"
    out_base = parent_dir / "out"
    base_folder = parent_dir / "full_texas"

    shapefile_path = in_dir / "us-county-boundaries" / "us-county-boundaries.shp"

    process_all_zones_parallel(
        base_folder=base_folder,
        shapefile_path=shapefile_path,
        save_location="out_folder",
        out_base=out_base,
        max_workers=None  # or set to a number like 4
    )

    print("\n🏁 Done.")
