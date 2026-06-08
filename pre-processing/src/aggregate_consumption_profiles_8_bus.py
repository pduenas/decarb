import pandas as pd
import os
from pathlib import Path

def load_bus_mapping(bus_mapping_file_path):
    """
    Load the bus mapping file to create a dictionary of Number -> Area Number.
    
    Args:
        bus_mapping_file_path: Path to ERCOT_bus_mapping_8bus.csv
        
    Returns:
        Dictionary with Number as key and Area Number as value
    """
    print(f"Loading bus mapping from {bus_mapping_file_path}...")
    df = pd.read_csv(bus_mapping_file_path)
    
    # Create mapping from Number to Area Number
    bus_mapping = dict(zip(df['Number'], df['Area Num']))
    
    print(f"Loaded mapping for {len(bus_mapping)} buses")
    print(f"Unique buses: {sorted(set(bus_mapping.values()))}")
    
    return bus_mapping

def load_building_to_bus_mapping(mapping_file_path, bus_mapping):
    """
    Load the building mapping file and create dictionaries for bldg_id -> count and bldg_id -> bus.
    
    Args:
        mapping_file_path: Path to ercot_substation_nrel_map.parquet
        bus_mapping: Dictionary of network_bus_id (Number) -> Area Number
        
    Returns:
        Tuple of (building_counts dict, building_to_bus dict)
    """
    print(f"\nLoading building counts and bus assignments from {mapping_file_path}...")
    df = pd.read_parquet(mapping_file_path, engine="fastparquet")
    
    # Map network_bus_id to Area Number
    df['bus'] = df['network_bus_id'].map(bus_mapping)
    
    # Check for unmapped buildings
    unmapped = df[df['bus'].isna()]
    if len(unmapped) > 0:
        print(f"  Warning: {len(unmapped)} rows could not be mapped to a bus")
    
    # Remove unmapped rows
    df = df[df['bus'].notna()]
    
    # Group by bldg_id and get total count and bus assignment
    # Note: Assuming each bldg_id maps to only one bus
    building_counts = df.groupby('bldg_id')['count'].sum().to_dict()
    building_to_bus = df.groupby('bldg_id')['bus'].first().to_dict()
    
    print(f"Loaded counts for {len(building_counts)} unique buildings")
    print(f"Buildings mapped to buses: {len(building_to_bus)}")
    
    return building_counts, building_to_bus

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

def aggregate_parquet_files_by_bus(folder_path, mapping_file_path, bus_mapping_file_path, 
                                    output_path, single_file=True):
    """
    Aggregates timeseries data from all parquet files in a folder,
    scaling by building counts and grouping by bus.
    
    Args:
        folder_path: Path to folder containing .parquet files
        mapping_file_path: Path to ercot_substation_nrel_map.parquet
        bus_mapping_file_path: Path to ERCOT_bus_mapping_8bus.csv
        output_path: Path to save output files
        single_file: If True, save all buses in one file with separate columns.
                    If False, save separate files for each bus.
    """
    # Column mapping: original -> new names
    column_mapping = {
        'out.electricity.cooling.energy_consumption..kwh': 'cooling_energy_kwh',
        'out.electricity.heating.energy_consumption..kwh': 'heating_energy_kwh',
        'out.electricity.ev_charging.energy_consumption..kwh': 'ev_charging_energy_kwh',
        'out.electricity.net.energy_consumption..kwh': 'net_energy_kwh',
        'out.electricity.total.energy_consumption..kwh': 'total_energy_kwh'
    }
    
    # Load bus mapping
    bus_mapping = load_bus_mapping(bus_mapping_file_path)
    
    # Load building counts and bus assignments
    building_counts, building_to_bus = load_building_to_bus_mapping(mapping_file_path, bus_mapping)
    
    # Get all parquet files
    parquet_files = list(Path(folder_path).glob('*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {folder_path}")
        return
    
    print(f"\nFound {len(parquet_files)} parquet files")
    
    # Dictionary to store dataframes by bus
    bus_dfs = {}
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
        
        # Get scaling factor and bus for this building
        scaling_factor = building_counts.get(bldg_id)
        bus = building_to_bus.get(bldg_id)
        
        if scaling_factor is None or bus is None:
            print(f"  Warning: Building ID {bldg_id} not found in mapping file, skipping...")
            skipped_files.append(file.name)
            continue
        
        print(f"  Building ID: {bldg_id}, Bus: {bus}, Scaling factor: {scaling_factor}")
        
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
        
        # Add to the appropriate bus list
        if bus not in bus_dfs:
            bus_dfs[bus] = []
        bus_dfs[bus].append(df_subset)
    
    if not bus_dfs:
        print("\nNo valid data found to aggregate")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {len(bus_dfs)} buses...")
    
    # Aggregate data for each bus
    bus_aggregated = {}
    for bus, dfs in bus_dfs.items():
        print(f"\nAggregating bus {bus} ({len(dfs)} files)...")
        
        # Concatenate all dataframes for this bus
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Group by timestamp and aggregate (sum)
        aggregated_df = combined_df.groupby('timestamp', as_index=False).sum()
        
        # Rename columns
        aggregated_df.rename(columns=column_mapping, inplace=True)
        
        # Sort by timestamp
        aggregated_df.sort_values('timestamp', inplace=True)
        
        bus_aggregated[bus] = aggregated_df
        print(f"  Bus {bus}: {len(aggregated_df)} timestamps")
    
    # Save outputs
    if single_file:
        # Combine all buses into one dataframe with prefixed columns
        print(f"\n{'='*60}")
        print("Combining all buses into single file...")
        
        # Start with timestamp from first bus
        first_bus = list(bus_aggregated.keys())[0]
        result_df = bus_aggregated[first_bus][['timestamp']].copy()
        
        # Add columns for each bus
        for bus in sorted(bus_aggregated.keys()):
            bus_df = bus_aggregated[bus]
            for col in ['cooling_energy_kwh', 'heating_energy_kwh', 'ev_charging_energy_kwh', 
                       'net_energy_kwh', 'total_energy_kwh']:
                if col in bus_df.columns:
                    result_df[f'bus{bus}_{col}'] = bus_df[col]
        
        # Save to CSV
        csv_output = output_path / 'ercot_consumption_by_bus.csv'
        print(f"Saving to {csv_output}...")
        result_df.to_csv(csv_output, index=False)
        
        # Save to Parquet
        parquet_output = output_path / 'ercot_consumption_by_bus.parquet'
        print(f"Saving to {parquet_output}...")
        result_df.to_parquet(parquet_output, index=False)
        
        print(f"\nOutput columns: {list(result_df.columns)}")
        print(f"Total rows: {len(result_df)}")
        print(f"\nFirst few rows:")
        print(result_df.head())
        
    else:
        # Save separate file for each bus
        print(f"\n{'='*60}")
        print("Saving separate files for each bus...")
        
        for bus in sorted(bus_aggregated.keys()):
            bus_df = bus_aggregated[bus]
            
            # Save to CSV
            csv_output = output_path / f'ercot_bus{bus}_consumption.csv'
            print(f"Saving bus {bus} to {csv_output}...")
            bus_df.to_csv(csv_output, index=False)
            
            # Save to Parquet
            parquet_output = output_path / f'ercot_bus{bus}_consumption.parquet'
            print(f"Saving bus {bus} to {parquet_output}...")
            bus_df.to_parquet(parquet_output, index=False)
            
            print(f"  Bus {bus}: {len(bus_df)} rows")
    
    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"Processed files for {len(bus_dfs)} buses")
    print(f"Skipped: {len(skipped_files)} files")
    if skipped_files:
        print(f"Skipped files: {skipped_files[:10]}{'...' if len(skipped_files) > 10 else ''}")

if __name__ == "__main__":
    # Configuration
    SINGLE_FILE = True  # Set to True for one file with all buses, False for separate files
    
    folder_path = Path(__file__).resolve().parents[1]  # Parent directory of script location
    consumption_path = folder_path / "out" / "consumption_files"
    mapping_file_path = folder_path / "out" / "ercot_substation_nrel_map.parquet"
    bus_mapping_file_path = folder_path / "in" / "ERCOT_bus_mapping_8bus.csv"
    output_path = folder_path / 'out'
    
    aggregate_parquet_files_by_bus(consumption_path, mapping_file_path, 
                                   bus_mapping_file_path, output_path, 
                                   single_file=SINGLE_FILE)