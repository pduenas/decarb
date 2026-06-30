import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

# -----------------------------
# Parameters
# -----------------------------
PARENT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PARENT_DIR / "out"
MISSING_CSV = OUT_DIR / "missing_matches.csv"

COUNTY_TAG = "in.county"
CLIMATE_TAG = "in.building_america_climate_zone"
SUMMER_TAG = "out.qoi.electricity.maximum_daily_peak_summer..kw"
WINTER_TAG = "out.qoi.electricity.maximum_daily_peak_winter..kw"
USAGE_LIMIT = 1000  # for reference; still tracked

# -----------------------------
# Helpers
# -----------------------------
def transform_geoid(geoid: str) -> str:
    return geoid[:3] + '0' + geoid[3:] + '0'

def max_demand(row):
    return max(row[SUMMER_TAG], row[WINTER_TAG])

# -----------------------------
# Load inputs
# -----------------------------
nrel_df = pd.read_parquet(PARENT_DIR / "in" / "TX_upgrade0.parquet", engine="fastparquet")
climate_gdf = gpd.read_file(PARENT_DIR / "in" / "ClimateZoneDataFiles" / "ClimateZones.shp").to_crs(epsg=4326)
climate_gdf["GEOID_NEW"] = climate_gdf["GEOID"].astype(str).apply(transform_geoid)
county_to_climate = dict(zip(climate_gdf["GEOID_NEW"], climate_gdf["BA21"]))

# Precompute max demand
nrel_df["max_demand_kw"] = nrel_df[[SUMMER_TAG, WINTER_TAG]].max(axis=1)
if "bldg_id" not in nrel_df.columns:
    nrel_df = nrel_df.reset_index().rename(columns={"index": "bldg_id"})
nrel_df["num_climate_matches_zone"] = 0

# Load missing matches
missing_df = pd.read_csv(MISSING_CSV)

# -----------------------------
# Process feeders
# -----------------------------
for idx, row in missing_df.iterrows():
    zone = row["zone"]
    feeder_file = row["feeder_file"]
    feeder_path = OUT_DIR / zone / "feeder_summaries" / feeder_file

    if not feeder_path.exists():
        print(f"⚠️ Missing file: {feeder_path}")
        continue

    feeder_df = pd.read_csv(feeder_path)

    # Identify rows with missing county or climate matches
    missing_mask = feeder_df["nrel_county_match"].isna() | feeder_df["nrel_climate_match"].isna()
    if missing_mask.sum() == 0:
        print(f"✓ {feeder_file}: no missing rows")
        continue

    # Process each missing row
    for i in feeder_df[missing_mask].index:
        feeder_kw = feeder_df.at[i, "kw"]
        feeder_county = feeder_df.at[i, "county_code"]

        # -----------------------------
        # COUNTY MATCH
        # -----------------------------
        county_match_row = None
        if pd.isna(feeder_df.at[i, "nrel_county_match"]):
            if pd.notna(feeder_county):
                subset = nrel_df[nrel_df[COUNTY_TAG] == feeder_county]
                if not subset.empty:
                    diffs = np.abs(subset["max_demand_kw"] - feeder_kw)
                    county_match_row = subset.iloc[diffs.argmin()]
            if county_match_row is None:
                # fallback to global max-demand match
                diffs = np.abs(nrel_df["max_demand_kw"] - feeder_kw)
                county_match_row = nrel_df.iloc[diffs.argmin()]

            # Write county match
            feeder_df.at[i, "nrel_county_match"] = county_match_row["bldg_id"]
            feeder_df.at[i, "nrel_county_summer_kw"] = county_match_row[SUMMER_TAG]
            feeder_df.at[i, "nrel_county_winter_kw"] = county_match_row[WINTER_TAG]

        else:
            # If county already exists, just fetch its info
            subset = nrel_df[nrel_df["bldg_id"] == feeder_df.at[i, "nrel_county_match"]]
            if not subset.empty:
                county_match_row = subset.iloc[0]

        # -----------------------------
        # CLIMATE MATCH
        # -----------------------------
        if pd.isna(feeder_df.at[i, "nrel_climate_match"]):
            summer_dominant = feeder_df.at[i, "nrel_county_summer_kw"] >= feeder_df.at[i, "nrel_county_winter_kw"]

            climate_zone = county_to_climate.get(feeder_county)
            # Filter by zone and summer dominance
            candidates = nrel_df.copy()
            if climate_zone:
                candidates = candidates[candidates[CLIMATE_TAG] == climate_zone]
            if summer_dominant:
                candidates = candidates[candidates[SUMMER_TAG] >= candidates[WINTER_TAG]]
            else:
                candidates = candidates[candidates[SUMMER_TAG] < candidates[WINTER_TAG]]

            if candidates.empty:
                # fallback to county match values
                feeder_df.at[i, "nrel_climate_match"] = feeder_df.at[i, "nrel_county_match"]
                feeder_df.at[i, "nrel_climate_summer_kw"] = feeder_df.at[i, "nrel_county_summer_kw"]
                feeder_df.at[i, "nrel_climate_winter_kw"] = feeder_df.at[i, "nrel_county_winter_kw"]
            else:
                diffs = np.abs(candidates["max_demand_kw"] - feeder_kw)
                best_row = candidates.iloc[diffs.argmin()]
                feeder_df.at[i, "nrel_climate_match"] = best_row["bldg_id"]
                feeder_df.at[i, "nrel_climate_summer_kw"] = best_row[SUMMER_TAG]
                feeder_df.at[i, "nrel_climate_winter_kw"] = best_row[WINTER_TAG]

            # Update usage count
            nrel_df.loc[nrel_df["bldg_id"] == feeder_df.at[i, "nrel_climate_match"], "num_climate_matches_zone"] += 1

    # -----------------------------
    # Save feeder file
    # -----------------------------
    feeder_df.to_csv(feeder_path, index=False)
    print(f"✅ Saved: {feeder_path}")

print("\n🎉 All missing feeders updated successfully.")
