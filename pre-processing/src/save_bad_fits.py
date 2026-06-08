import pandas as pd
from pathlib import Path
import re

def collect_low_r_squared_buildings(
    output_dir,
    feature_set="baseline",
    r_squared_threshold=0.95,
    method="ols"  # "ols", "regularized", or "rescue"
):
    """
    Collect all buildings with R² < threshold from all zones into a single CSV.
    
    Args:
        output_dir: Output directory containing zone folders
        feature_set: Feature set name
        r_squared_threshold: Threshold for filtering buildings
        method: "ols", "regularized", or "rescue"
    """
    
    zone_pattern = re.compile(r"^P\d+[UR]$")
    zone_folders = [f for f in output_dir.iterdir() if f.is_dir()]
    zones = [z.name for z in zone_folders if zone_pattern.match(z.name)]
    
    all_low_r_squared = []
    
    for zone in zones:
        # Determine coefficient file based on method
        if method == "ols":
            coeff_file = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}.csv"
        elif method == "regularized":
            # Use first lambda value as default, or modify to check all
            coeff_file = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_regularized_{feature_set}_lambda_10.0.csv"
        elif method == "rescue":
            coeff_file = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}_rescue.csv"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if not coeff_file.exists():
            print(f"⚠ Skipping {zone}: coefficient file not found")
            continue
        
        # Load coefficients
        df = pd.read_csv(coeff_file)
        
        # Filter by R²
        low_r_squared = df[df["r_squared"] < r_squared_threshold].copy()
        
        if len(low_r_squared) > 0:
            # Add zone column
            low_r_squared.insert(0, "zone", zone)
            all_low_r_squared.append(low_r_squared)
            print(f"✓ {zone}: Found {len(low_r_squared)} buildings with R² < {r_squared_threshold}")
    
    if not all_low_r_squared:
        print(f"\n✅ No buildings found with R² < {r_squared_threshold}")
        return
    
    # Combine all zones
    combined_df = pd.concat(all_low_r_squared, ignore_index=True)
    
    # Sort by R² (worst first)
    combined_df = combined_df.sort_values("r_squared")
    
    # Save to CSV
    out_file = output_dir / f"all_zones_low_r_squared_{feature_set}_{method}_threshold_{r_squared_threshold}.csv"
    combined_df.to_csv(out_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Low R² Buildings (R² < {r_squared_threshold})")
    print(f"{'='*60}")
    print(f"Total buildings: {len(combined_df)}")
    print(f"Zones affected: {combined_df['zone'].nunique()}")
    print(f"Worst R²: {combined_df['r_squared'].min():.6f}")
    print(f"Best (among low): {combined_df['r_squared'].max():.6f}")
    print(f"Mean R²: {combined_df['r_squared'].mean():.6f}")
    print(f"\n💾 Saved to: {out_file}")
    print(f"{'='*60}\n")
    
    return combined_df


if __name__ == "__main__":
    
    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]
    output_dir = parent_dir / "out"
    
    # Collect low R² buildings from OLS with positive constraints
    df_low = collect_low_r_squared_buildings(
        output_dir=output_dir,
        feature_set="baseline",
        r_squared_threshold=0.95,
        method="ols"  # Change to "regularized" or "rescue" as needed
    )
    
    # Optional: show the 10 worst buildings
    if df_low is not None and len(df_low) > 0:
        print("\n🔻 10 WORST BUILDINGS:")
        print(df_low[["zone", "bldg_id", "r_squared", "mse", "k1", "k2", "k3"]].head(10).to_string(index=False))