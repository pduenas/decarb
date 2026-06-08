import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree

def transform_geoid(geoid: str) -> str:
    return geoid[:3] + '0' + geoid[3:] + '0'


def find_closest_demand_match(nrel_df, valid_df, feeder_summary_path, climate_gdf, usage_limit=500):
    """
    Updated version:
    - valid_df is passed in (persistent across entire zone)
    - num_climate_matches_zone increments with each match
    - buildings removed from valid_df once limit is reached
    """

    feeder_df = pd.read_csv(feeder_summary_path)
    required_cols = ['county_code', 'kw']
    for col in required_cols:
        if col not in feeder_df.columns:
            raise ValueError(f"Feeder summary missing required column: {col}")

    # Initialize output columns if missing
    match_columns = [
        "nrel_county_match",
        "nrel_county_summer_kw",
        "nrel_county_winter_kw",
        "nrel_climate_match",
        "nrel_climate_summer_kw",
        "nrel_climate_winter_kw"
    ]
    for col in match_columns:
        if col not in feeder_df.columns:
            feeder_df[col] = np.nan

    # Precompute max demand
    nrel_df["max_demand_kw"] = nrel_df[[summer_demand_tag, winter_demand_tag]].max(axis=1)
    valid_df["max_demand_kw"] = valid_df[[summer_demand_tag, winter_demand_tag]].max(axis=1)

    # Map county_code to climate zone
    climate_gdf['GEOID_NEW'] = climate_gdf['GEOID'].astype(str).apply(transform_geoid)
    county_to_climate = dict(zip(climate_gdf["GEOID_NEW"], climate_gdf["BA21"]))

    # --- County matches ---
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
        county_matches.append((best_row["bldg_id"], best_row[summer_demand_tag], best_row[winter_demand_tag]))

    county_matches = np.array(county_matches, dtype=object)
    feeder_df["nrel_county_match"] = county_matches[:, 0]
    feeder_df["nrel_county_summer_kw"] = county_matches[:, 1]
    feeder_df["nrel_county_winter_kw"] = county_matches[:, 2]

    # --- Climate matches with KDTree + usage limits ---
    def build_trees(df):
        trees = {}
        for zone in df[climate_tag].unique():
            for summer_dom in [True, False]:
                subset = df[(df[climate_tag] == zone) &
                            ((df[summer_demand_tag] >= df[winter_demand_tag]) == summer_dom)]
                if not subset.empty:
                    vectors = subset[[summer_demand_tag, winter_demand_tag]].to_numpy(dtype=float)
                    trees[(zone, summer_dom)] = (subset.copy(), cKDTree(vectors))
        return trees

    kdtrees = build_trees(valid_df)

    climate_match_idx = []
    climate_summer = []
    climate_winter = []

    for county_match, summer_kw, winter_kw, county_code in zip(
        feeder_df["nrel_county_match"],
        feeder_df["nrel_county_summer_kw"],
        feeder_df["nrel_county_winter_kw"],
        feeder_df["county_code"]
    ):
        if pd.isna(county_match):
            climate_match_idx.append(None)
            climate_summer.append(None)
            climate_winter.append(None)
            continue

        summer_dominant = summer_kw >= winter_kw
        climate_zone = county_to_climate.get(county_code)
        candidates_tuple = kdtrees.get((climate_zone, summer_dominant))

        if candidates_tuple is None or candidates_tuple[0].empty:
            climate_match_idx.append(None)
            climate_summer.append(None)
            climate_winter.append(None)
            continue

        subset, tree = candidates_tuple
        target = np.array([[summer_kw, winter_kw]])
        dist, idx = tree.query(target, k=1)
        best_row = subset.iloc[idx[0]]

        # Update usage count
        bldg_id = best_row["bldg_id"]
        valid_df.loc[valid_df["bldg_id"] == bldg_id, "num_climate_matches_zone"] += 1
        match_count = valid_df.loc[valid_df["bldg_id"] == bldg_id, "num_climate_matches_zone"].iloc[0]

        # If limit reached, remove from valid_df
        if match_count >= usage_limit:
            valid_df = valid_df[valid_df["bldg_id"] != bldg_id]
            print(f"⚠️ Dropping building {bldg_id} from valid_df (reached {usage_limit} matches)")

            # Update KDTree for this subset
            remaining_subset = subset[subset["bldg_id"] != bldg_id]
            if remaining_subset.empty:
                kdtrees.pop((climate_zone, summer_dominant))
            else:
                vectors = remaining_subset[[summer_demand_tag, winter_demand_tag]].to_numpy(dtype=float)
                kdtrees[(climate_zone, summer_dominant)] = (remaining_subset, cKDTree(vectors))

        climate_match_idx.append(bldg_id)
        climate_summer.append(best_row[summer_demand_tag])
        climate_winter.append(best_row[winter_demand_tag])

    feeder_df["nrel_climate_match"] = climate_match_idx
    feeder_df["nrel_climate_summer_kw"] = climate_summer
    feeder_df["nrel_climate_winter_kw"] = climate_winter

    return feeder_df, valid_df


# =======================
# Main script block
# =======================
if __name__ == "__main__":
    zone = "P1R"
    overwrite_existing = True
    save_mode = "zone"
    usage_limit = 500

    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]
    zone_path = parent_dir / zone / "scenarios" / "base_timeseries" / "opendss"
    zone_output_dir = parent_dir / 'out' / zone / "feeder_summaries"
    nrel_parquet_path = parent_dir / 'in' / 'TX_upgrade0.parquet'
    climate_shp_path = parent_dir / 'in' / 'ClimateZoneDataFiles' / 'ClimateZones.shp'

    nrel_df = pd.read_parquet(nrel_parquet_path, engine="fastparquet")
    county_tag = 'in.county'
    climate_tag = 'in.building_america_climate_zone'
    summer_demand_tag = 'out.qoi.electricity.maximum_daily_peak_summer..kw'
    winter_demand_tag = 'out.qoi.electricity.maximum_daily_peak_winter..kw'

    climate_gdf = gpd.read_file(climate_shp_path).to_crs(epsg=4326)

    substation_dirs = [d for d in zone_path.iterdir() if d.is_dir()]

    # Persistent valid_df across the entire zone
    valid_df = nrel_df.copy()
    valid_df["num_climate_matches_zone"] = 0
    if "bldg_id" not in valid_df.columns:
        valid_df = valid_df.reset_index().rename(columns={"index": "bldg_id"})

    print(f"\n🏗️ Found {len(substation_dirs)} substations in zone {zone}...\n")

    for substation_dir in substation_dirs:
        feeder_summary_dir = substation_dir / "feeder_summaries"
        if not feeder_summary_dir.exists():
            print(f"⏭️ Skipping substation {substation_dir.name} (no feeder summaries)\n")
            continue

        feeder_files = list(feeder_summary_dir.glob("feeder_summary_*.csv"))
        print(f"\n🏗️ Substation: {substation_dir.name} | Found {len(feeder_files)} feeder summary files")

        for feeder_file in feeder_files:
            print(f"🔄 Processing: {feeder_file.name}")
            existing_df = pd.read_csv(feeder_file, nrows=1)
            has_nrel_columns = {"nrel_county_match", "nrel_climate_match"}.issubset(existing_df.columns)

            if has_nrel_columns and not overwrite_existing:
                print(f"⏭️ Skipping (already processed): {feeder_file.name}\n")
                continue

            enriched_df, valid_df = find_closest_demand_match(
                nrel_df, valid_df, feeder_file, climate_gdf, usage_limit
            )

            if save_mode == "zone":
                zone_output_dir.mkdir(parents=True, exist_ok=True)
                out_path = zone_output_dir / feeder_file.name
            else:
                out_path = feeder_file

            enriched_df.to_csv(out_path, index=False)
            print(f"✅ Saved: {out_path}\n")

    # Save valid_df at the end of zone processing
    valid_df_output_path = zone_output_dir / 'valid_df_output.csv'
    valid_df.to_csv(valid_df_output_path, index=False)
    print(f"✅ Final valid_df saved: {valid_df_output_path}")
