import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import time
import fnmatch
import re


def transform_geoid(geoid: str) -> str:
    return geoid[:3] + '0' + geoid[3:] + '0'


def build_trees(df, climate_tag, summer_demand_tag, winter_demand_tag):
    """Build KDTree index per (climate_zone, summer_dominant) pair."""
    trees = {}
    for zone in df[climate_tag].unique():
        for summer_dom in [True, False]:
            subset = df[
                (df[climate_tag] == zone)
                & ((df[summer_demand_tag] >= df[winter_demand_tag]) == summer_dom)
            ]
            if not subset.empty:
                vectors = subset[[summer_demand_tag, winter_demand_tag]].to_numpy(dtype=float)
                trees[(zone, summer_dom)] = (subset.copy(), cKDTree(vectors))
    return trees


def find_closest_demand_match(
    nrel_df, valid_df, feeder_df, climate_gdf, usage_limit=500,
    county_tag='in.county', climate_tag='in.building_america_climate_zone',
    summer_demand_tag='out.qoi.electricity.maximum_daily_peak_summer..kw',
    winter_demand_tag='out.qoi.electricity.maximum_daily_peak_winter..kw'
):
    """Find county and climate demand matches for feeders, with fallback logic."""

    # --- Required columns ---
    required_cols = ['county_code', 'kw']
    for col in required_cols:
        if col not in feeder_df.columns:
            raise ValueError(f"Feeder summary missing required column: {col}")

    # --- Output columns ---
    match_columns = [
        "nrel_county_match",
        "nrel_county_summer_kw",
        "nrel_county_winter_kw",
        "nrel_climate_match",
        "nrel_climate_summer_kw",
        "nrel_climate_winter_kw",
    ]
    for col in match_columns:
        if col not in feeder_df.columns:
            feeder_df[col] = np.nan

    # --- Precompute max demand ---
    nrel_df["max_demand_kw"] = nrel_df[[summer_demand_tag, winter_demand_tag]].max(axis=1)
    valid_df["max_demand_kw"] = valid_df[[summer_demand_tag, winter_demand_tag]].max(axis=1)

    # --- County-to-climate mapping ---
    climate_gdf["GEOID_NEW"] = climate_gdf["GEOID"].astype(str).apply(transform_geoid)
    county_to_climate = dict(zip(climate_gdf["GEOID_NEW"], climate_gdf["BA21"]))

    # ============================================================
    # COUNTY MATCH
    # ============================================================
    county_groups = {c: g for c, g in nrel_df.groupby(county_tag)}

    county_matches = []
    for county_code, feeder_kw in zip(feeder_df["county_code"], feeder_df["kw"]):
        if pd.isna(county_code) or pd.isna(feeder_kw):
            county_matches.append((None, None, None))
            continue

        county_subset = county_groups.get(county_code)
        if county_subset is None or county_subset.empty:
            county_matches.append((None, None, None))
            continue

        diffs = np.abs(county_subset["max_demand_kw"].values - feeder_kw)
        best_idx = diffs.argmin()
        best_row = county_subset.iloc[best_idx]
        county_matches.append(
            (best_row["bldg_id"], best_row[summer_demand_tag], best_row[winter_demand_tag])
        )

    county_matches = np.array(county_matches, dtype=object)
    feeder_df["nrel_county_match"] = county_matches[:, 0]
    feeder_df["nrel_county_summer_kw"] = county_matches[:, 1]
    feeder_df["nrel_county_winter_kw"] = county_matches[:, 2]

    # ============================================================
    # CLIMATE MATCH (with required rules)
    # ============================================================

    kdtrees = build_trees(valid_df, climate_tag, summer_demand_tag, winter_demand_tag)

    climate_match_idx, climate_summer, climate_winter = [], [], []

    for county_match, summer_kw, winter_kw, county_code in zip(
        feeder_df["nrel_county_match"],
        feeder_df["nrel_county_summer_kw"],
        feeder_df["nrel_county_winter_kw"],
        feeder_df["county_code"],
    ):
        if pd.isna(county_match):
            climate_match_idx.append(None)
            climate_summer.append(None)
            climate_winter.append(None)
            continue

        summer_dominant = summer_kw >= winter_kw
        climate_zone = county_to_climate.get(county_code)

        candidates_tuple = kdtrees.get((climate_zone, summer_dominant))

        # ============================================================
        # RULE 2: No climate match → assign county match + RESET valid_df
        # ============================================================
        if candidates_tuple is None or candidates_tuple[0].empty:

            print(
                f"⚠️ No climate matches for county {county_code}. "
                f"Using county match and resetting valid_df / usage counts."
            )

            # Assign county match for climate fields
            climate_match_idx.append(county_match)
            climate_summer.append(summer_kw)
            climate_winter.append(winter_kw)

            # Reset valid_df and zero usage counts
            valid_df = nrel_df.copy()
            valid_df["num_climate_matches_zone"] = 0

            # Rebuild KD-trees
            kdtrees = build_trees(valid_df, climate_tag, summer_demand_tag, winter_demand_tag)

            continue

        # ------------------------------------------------------------
        # Normal climate-match lookup via KD-tree
        # ------------------------------------------------------------
        subset, tree = candidates_tuple
        target = np.array([[summer_kw, winter_kw]])
        dist, idx = tree.query(target, k=1)
        best_row = subset.iloc[idx[0]]
        bldg_id = best_row["bldg_id"]

        # Increase usage count — but *never mutate usage_limit*
        valid_df.loc[valid_df["bldg_id"] == bldg_id,
                     "num_climate_matches_zone"] += 1

        match_count = valid_df.loc[
            valid_df["bldg_id"] == bldg_id,
            "num_climate_matches_zone"
        ].iloc[0]

        # If usage limit exceeded, remove building & rebuild KD-trees
        if match_count >= usage_limit:
            print(f"⚠️ Dropping building {bldg_id} (hit usage limit {usage_limit})")

            valid_df = valid_df[valid_df["bldg_id"] != bldg_id]
            kdtrees = build_trees(valid_df, climate_tag, summer_demand_tag, winter_demand_tag)

        climate_match_idx.append(bldg_id)
        climate_summer.append(best_row[summer_demand_tag])
        climate_winter.append(best_row[winter_demand_tag])

    # Write climate matches to feeder dataframe
    feeder_df["nrel_climate_match"] = climate_match_idx
    feeder_df["nrel_climate_summer_kw"] = climate_summer
    feeder_df["nrel_climate_winter_kw"] = climate_winter

    return feeder_df, valid_df, usage_limit


def process_zone(zone_folder, nrel_df, valid_df, climate_gdf, usage_limit):
    """Batch-process all feeders in a zone."""
    feeder_summary_dir = zone_folder / "feeder_summaries"
    if not feeder_summary_dir.exists():
        print(f"⏭️ Skipping {zone_folder.name}: no feeder_summaries folder")
        return valid_df, usage_limit

    feeder_files = [
        f for f in feeder_summary_dir.iterdir()
        if f.is_file() and fnmatch.fnmatch(f.name, "feeder_summary_*.csv")
    ]
    if not feeder_files:
        print(f"⏭️ Skipping {zone_folder.name}: no feeder summary CSVs")
        return valid_df, usage_limit

    print(f"\n🏗️ Processing zone {zone_folder.name} with {len(feeder_files)} feeders")

    # Load and concatenate all feeder summaries
    original_dfs = []
    lengths = []
    for f in feeder_files:
        df = pd.read_csv(f)
        original_dfs.append(df)
        lengths.append(len(df))

    combined_df = pd.concat(original_dfs, ignore_index=True)

    # Match
    enriched_df, valid_df, usage_limit = find_closest_demand_match(
        nrel_df, valid_df, combined_df, climate_gdf, usage_limit
    )

    # Split to original file shapes
    cursor = 0
    for feeder_file, length in zip(feeder_files, lengths):
        sub = enriched_df.iloc[cursor:cursor + length]
        cursor += length
        sub.to_csv(feeder_file, index=False)
        print(f"✅ Saved: {feeder_file.name}")

    return valid_df, usage_limit


# =======================
# Main script
# =======================
if __name__ == "__main__":
    start_time = time.time()
    usage_limit = 1000
    overwrite_existing = True

    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]

    nrel_parquet_path = parent_dir / "in" / "TX_upgrade0.parquet"
    climate_shp_path = parent_dir / "in" / "ClimateZoneDataFiles" / "ClimateZones.shp"

    nrel_df = pd.read_parquet(nrel_parquet_path, engine="fastparquet")
    county_tag = "in.county"
    climate_tag = "in.building_america_climate_zone"
    summer_demand_tag = "out.qoi.electricity.maximum_daily_peak_summer..kw"
    winter_demand_tag = "out.qoi.electricity.maximum_daily_peak_winter..kw"

    climate_gdf = gpd.read_file(climate_shp_path).to_crs(epsg=4326)

    # Load valid bldg_ids from the CSV file
    valid_bldg_ids_df = pd.read_csv(parent_dir / "in" / "valid_bldg_ids_tx.csv")
    # Extract the bldg_id column (you might want to handle any potential missing or malformed data here)
    valid_bldg_ids = valid_bldg_ids_df["bldg_id"].dropna().unique()

    # Filter the nrel_df to include only rows where bldg_id is in the valid_bldg_ids
    valid_df = nrel_df[nrel_df["bldg_id"].isin(valid_bldg_ids)].copy()
    valid_df["num_climate_matches_zone"] = 0
    if "bldg_id" not in valid_df.columns:
        valid_df = valid_df.reset_index().rename(columns={"index": "bldg_id"})

    out_dir = parent_dir / "out"
    zones = [
        d for d in out_dir.iterdir() if d.is_dir() and re.match(r"P\d+[RU]$", d.name)
    ]

    total_zones = len(zones)
    for i, zone_folder in enumerate(zones, start=1):
        valid_df, usage_limit = process_zone(
            zone_folder, nrel_df, valid_df, climate_gdf, usage_limit
        )
        elapsed = time.time() - start_time
        print(f"⏳ Zone {i}/{total_zones} done | {total_zones - i} remaining | Time: {elapsed:.1f}s")

    valid_df_output_path = parent_dir / "out" / "valid_df_output.csv"
    valid_df.to_csv(valid_df_output_path, index=False)
    print(f"\n✅ Final valid_df saved: {valid_df_output_path}")
