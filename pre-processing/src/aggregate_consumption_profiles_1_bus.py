import pandas as pd
import os
from pathlib import Path

def load_building_counts(mapping_file_path):
    """
    Load the building mapping file and create a dictionary of bldg_id -> total count.
    
    Args:
        mapping_file_path: Path to ercot_substation_nrel_map.csv
        
    Returns:
        Dictionary with bldg_id as key and sum of counts as value
    """
    print(f"Loading building counts from {mapping_file_path}...")
    df = pd.read_parquet(mapping_file_path, engine="fastparquet")
    
    # Group by bldg_id and sum the counts
    building_counts = df.groupby('bldg_id')['count'].sum().to_dict()
    
    print(f"Loaded counts for {len(building_counts)} unique buildings")
    print(f"Sample building counts: {dict(list(building_counts.items())[:5])}")
    
    return building_counts

def extract_building_id(filename):
    """
    Extract building ID from filename (e.g., '18-0.parquet' -> 18).
    
    Args:
        filename: Name of the parquet file
        
    Returns:
        Building ID as integer, or None if cannot extract
    """
    try:
        # Remove .parquet extension and get the part before '-'
        name_without_ext = filename.replace('.parquet', '')
        bldg_id = int(name_without_ext.split('-')[0])
        return bldg_id
    except (ValueError, IndexError):
        return None

def aggregate_parquet_files(folder_path, mapping_file_path, output_path):
    """
    Aggregates timeseries data from all parquet files in a folder,
    scaling by building counts from the mapping file.
    
    Args:
        folder_path: Path to folder containing .parquet files
        mapping_file_path: Path to ercot_substation_nrel_map.csv
    """
    # Column mapping: original -> new names
    column_mapping = {
        'out.electricity.cooling.energy_consumption..kwh': 'cooling_energy_kwh',
        'out.electricity.heating.energy_consumption..kwh': 'heating_energy_kwh',
        'out.electricity.ev_charging.energy_consumption..kwh': 'ev_charging_energy_kwh',
        'out.electricity.net.energy_consumption..kwh': 'net_energy_kwh',
        'out.electricity.total.energy_consumption..kwh': 'total_energy_kwh'
    }
    
    # Load building counts
    building_counts = load_building_counts(mapping_file_path)
    
    # Get all parquet files
    parquet_files = list(Path(folder_path).glob('*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {folder_path}")
        return
    
    print(f"\nFound {len(parquet_files)} parquet files")
    
    # List to store dataframes
    dfs = []
    skipped_files = []
    
    # Read and process each parquet file
    for file in parquet_files:
        print(f"\nProcessing: {file.name}")
        
        # Extract building ID from filename
        bldg_id = extract_building_id(file.name)
        
        if bldg_id is None:
            print(f"  Warning: Could not extract building ID from {file.name}, skipping...")
            skipped_files.append(file.name)
            continue
        
        # Get scaling factor for this building
        scaling_factor = building_counts.get(bldg_id)
        
        if scaling_factor is None:
            print(f"  Warning: Building ID {bldg_id} not found in mapping file, skipping...")
            skipped_files.append(file.name)
            continue
        
        print(f"  Building ID: {bldg_id}, Scaling factor: {scaling_factor}")
        
        # Read parquet file
        df = pd.read_parquet(file)
        
        # Select timestamp and required columns
        cols_to_keep = ['timestamp'] + list(column_mapping.keys())
        
        # Filter to only existing columns
        available_cols = [col for col in cols_to_keep if col in df.columns]
        
        if 'timestamp' not in available_cols:
            print(f"  Warning: 'timestamp' column not found in {file.name}, skipping...")
            skipped_files.append(file.name)
            continue
        
        df_subset = df[available_cols].copy()
        
        # Scale the energy columns by the building count
        energy_cols = [col for col in column_mapping.keys() if col in df_subset.columns]
        for col in energy_cols:
            df_subset[col] = df_subset[col] * scaling_factor
        
        dfs.append(df_subset)
    
    if not dfs:
        print("\nNo valid data found to aggregate")
        return
    
    # Concatenate all dataframes
    print(f"\n{'='*60}")
    print(f"Concatenating {len(dfs)} dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Group by timestamp and aggregate (sum)
    print("Aggregating by timestamp...")
    aggregated_df = combined_df.groupby('timestamp', as_index=False).sum()
    
    # Rename columns
    aggregated_df.rename(columns=column_mapping, inplace=True)
    
    # Sort by timestamp
    aggregated_df.sort_values('timestamp', inplace=True)
    
    # Save to CSV
    csv_output = output_path / 'ercot_consumption_kwh.csv'
    print(f"Saving to {csv_output}...")
    aggregated_df.to_csv(csv_output, index=False)
    
    # Save to Parquet
    parquet_output = output_path / 'ercot_consumption_kwh.parquet'
    print(f"Saving to {parquet_output}...")
    aggregated_df.to_parquet(parquet_output, index=False)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"Processed: {len(dfs)} files")
    print(f"Skipped: {len(skipped_files)} files")
    if skipped_files:
        print(f"Skipped files: {skipped_files[:10]}{'...' if len(skipped_files) > 10 else ''}")
    print(f"Total rows in output: {len(aggregated_df)}")
    print(f"\nOutput columns: {list(aggregated_df.columns)}")
    print(f"\nFirst few rows:")
    print(aggregated_df.head())
    print(f"\nSummary statistics:")
    print(aggregated_df.describe())

if __name__ == "__main__":
    folder_path = Path(__file__).resolve().parents[1]  # Parent directory of script location
    consumption_path = folder_path / "out" / "consumption_files"
    mapping_file_path = folder_path / "out" / "ercot_substation_nrel_map.parquet"
    output_path = folder_path / 'out'
    
    aggregate_parquet_files(consumption_path, mapping_file_path, output_path)