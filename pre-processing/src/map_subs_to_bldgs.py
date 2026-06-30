import pandas as pd
import re
from pathlib import Path


def extract_substation_name(feeder_name):
    """
    Extract substation name from feeder.
    Example: 'p1rhs0_25--p1rdt6513' -> 'p1rhs0_25'
    """
    substation = feeder_name.split('--')[0]
    return substation


def load_network_mapping(parent_dir):
    """Load the network ID mapping from LoadGenMap CSV"""
    load_map_path = parent_dir / 'in' / 'Texas7k_LoadGenMap.csv'
    
    if not load_map_path.exists():
        print(f"Warning: Load map file not found at {load_map_path}")
        return None
    
    try:
        print(f"Loading network mapping from: {load_map_path}")
        load_map = pd.read_csv(load_map_path)
        
        if 'NREL_Load_ID' not in load_map.columns or 'BusNum' not in load_map.columns:
            print("Warning: Load map missing required columns ('NREL_Load_ID', 'BusNum')")
            return None
        
        # Create dictionary: substation -> network_bus_id
        network_map = dict(zip(load_map['NREL_Load_ID'], load_map['BusNum']))
        print(f"Loaded {len(network_map)} network ID mappings\n")
        return network_map
        
    except Exception as e:
        print(f"Error loading network mapping: {e}")
        return None


def process_zone_folder(zone_path, zone_name):
    """Process a single zone folder and return substation-building mapping"""
    bus_file = zone_path / 'bldg_ids' / f'{zone_name}_bldg_id_mapping.csv'
    
    if not bus_file.exists():
        print(f"Warning: Bus mapping file not found at {bus_file}")
        return None
    
    # print(f"Processing: {bus_file}")
    
    try:
        df = pd.read_csv(bus_file)
        
        if 'feeder' not in df.columns or 'bldg_id' not in df.columns:
            print(f"Warning: Required columns not found in {bus_file}")
            print(f"  Expected: 'feeder' and 'bldg_id'")
            print(f"  Found: {df.columns.tolist()}")
            return None
        
        # Ensure the bldg_id is read as an integer
        df['bldg_id'] = df['bldg_id'].astype(float).astype(int)
        
        # Extract substation names from feeders
        df['substation'] = df['feeder'].apply(extract_substation_name)
        
        # Count occurrences: how many times each building appears at each feeder
        feeder_bldg_counts = (
            df.groupby(['feeder', 'substation', 'bldg_id'])
              .size()
              .reset_index(name='count')
        )
        
        return feeder_bldg_counts
        
    except Exception as e:
        print(f"Error processing {bus_file}: {e}")
        return None


def process_all_zones(zone_folders, network_map):
    """Process all zone folders and return combined dataframe"""
    all_data = []
    
    # Process each zone
    for zone_folder in zone_folders:
        zone_name = zone_folder.name
        print(f"Processing zone: {zone_name}")
        
        zone_data = process_zone_folder(zone_folder, zone_name)
        if zone_data is not None and not zone_data.empty:
            all_data.append(zone_data)
    
    if not all_data:
        return None
    
    # Combine all zones into one dataframe
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Aggregate across all zones (sum counts for same feeder-building pairs)
    final_df = (
        combined_df
        .groupby(['feeder', 'substation', 'bldg_id'])['count']
        .sum()
        .reset_index()
    )
    
    # Add network_bus_id using network mapping
    if network_map is not None:
        final_df['network_bus_id'] = final_df['substation'].map(network_map)
        
        # Report matching stats
        matched = final_df['network_bus_id'].notna().sum()
        total = len(final_df)
        print(f"\nNetwork ID matching: {matched}/{total} ({matched/total*100:.1f}%)")
        
        if matched < total:
            unmatched = final_df[final_df['network_bus_id'].isna()]['substation'].unique()
            print(f"  Unmatched substations (sample): {list(unmatched[:10])}")
    
    # Sort by feeder and count
    final_df = final_df.sort_values(['feeder', 'count'], ascending=[True, False])
    
    return final_df


def print_summary(final_df):
    """Print summary statistics"""
    print(f"\n{'=' * 60}")
    print(f"Total unique feeder-building pairs: {len(final_df)}")
    print(f"Total unique feeders: {final_df['feeder'].nunique()}")
    print(f"Total unique substations: {final_df['substation'].nunique()}")
    print(f"Total unique buildings: {final_df['bldg_id'].nunique()}")
    print(f"\nColumns in output: {final_df.columns.tolist()}")
    print(f"\nSample of results:")
    print(final_df.reset_index(drop=True).head(10))


def save_results(final_df, out_dir):
    """Save dataframe to CSV and Parquet"""
    csv_filename = out_dir / 'ercot_substation_nrel_map.csv'
    final_df.to_csv(csv_filename, index=False)
    print(f"\n✓ Saved to {csv_filename}")
    
    parquet_filename = out_dir / 'ercot_substation_nrel_map.parquet'
    final_df.to_parquet(parquet_filename, index=False)
    print(f"✓ Saved to {parquet_filename}")


if __name__ == "__main__":
    # File locations and setup
    parent_dir = Path(__file__).parents[1]
    out_dir = parent_dir / 'out'
    
    if not out_dir.exists():
        print(f"Error: 'out' directory not found at {out_dir}")
        exit()
    
    # Load network mapping
    network_map = load_network_mapping(parent_dir)
    
    # Find zone folders matching pattern P<number>U or P<number>R
    zone_pattern = re.compile(r'^P\d+[UR]$', re.IGNORECASE)
    zone_folders = [
        d for d in out_dir.iterdir()
        if d.is_dir() and zone_pattern.match(d.name)
    ]
    
    print(f"Found {len(zone_folders)} zone folders: {[z.name for z in zone_folders]}\n")
    
    # Process all zones
    final_df = process_all_zones(zone_folders, network_map)
    
    if final_df is not None:
        print_summary(final_df)
        save_results(final_df, out_dir)
    else:
        print("\nNo data found to process!")