import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

def create_demand_file(input_file, output_file, mapping_file):
    # Read the mapping file
    print(f"Reading mapping file: {mapping_file}")
    mapping_df = pd.read_excel(mapping_file)
    
    # Create a dictionary mapping NREL_Load_ID to BusNum
    load_to_bus = dict(zip(mapping_df['NREL_Load_ID'], mapping_df['BusNum']))
    
    # Read the input CSV file
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    
    # Get column names
    columns = df.columns.tolist()
    new_columns = []
    
    # Process each column name
    for col in columns:
        if col.startswith('Load_MW_'):
            # Extract the load ID (everything after 'Load_MW_')
            load_id = col.replace('Load_MW_', '')
            
            # Look up the BusNum
            if load_id not in load_to_bus:
                raise ValueError(f"Load ID '{load_id}' not found in mapping file!")
            
            bus_num = load_to_bus[load_id]
            new_col_name = f"Load_MW_{bus_num}"
            new_columns.append(new_col_name)
            print(f"Renamed: {col} -> {new_col_name}")
        else:
            # Keep the first 8 columns unchanged
            new_columns.append(col)
    
    # Rename the columns
    df.columns = new_columns
    
    # Save to output file
    print(f"Saving output file: {output_file}")
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, sep='\t')
    
    print("Done!")

def create_network_file(input_file, output_file, mapping_file, network_data_file):
    # Read the network CSV file
    print(f"Reading network file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Read the mapping file
    print(f"Reading mapping file: {mapping_file}")
    mapping_df = pd.read_excel(mapping_file)
    
    # Read the network data file for bus locations
    print(f"Reading network data file: {network_data_file}")
    network_df = pd.read_csv(network_data_file)
    
    # Create a dictionary mapping BusNum to NREL_Load_ID
    bus_to_load = dict(zip(mapping_df['BusNum'], mapping_df['NREL_Load_ID']))
    
    # Create a dictionary mapping bus number to (lat, lon)
    bus_to_coords = {}
    for _, row in network_df.iterrows():
        if pd.notna(row.get('bus_lat')) and pd.notna(row.get('bus_lon')):
            bus_num = row.get('bus number')
            if pd.notna(bus_num):
                bus_to_coords[int(bus_num)] = (row['bus_lat'], row['bus_lon'])
    
    # Function to calculate distance between two points using Haversine formula
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        
        a = sin(delta_lat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        distance = R * c
        return distance
    
    # Get unique zone numbers from both From and To columns
    from_zones = df['From'].dropna().unique()
    to_zones = df['To'].dropna().unique()
    all_zones = sorted(set(list(from_zones) + list(to_zones)))
    
    # Create zone names (z1, z2, z3, etc.)
    zone_names = [f"z{int(zone)}" for zone in all_zones]
    
    # Create network lines data with distances
    lines_data = []
    line_num = 1
    for idx, row in df.iterrows():
        if pd.notna(row['From']) and pd.notna(row['To']):
            from_bus = int(row['From'])
            to_bus = int(row['To'])
            
            # Calculate distance if coordinates are available
            distance = None
            if from_bus in bus_to_coords and to_bus in bus_to_coords:
                lat1, lon1 = bus_to_coords[from_bus]
                lat2, lon2 = bus_to_coords[to_bus]
                distance_km = haversine_distance(lat1, lon1, lat2, lon2)
                # Convert km to miles
                distance = distance_km * 0.621371
                # If distance is 0 or very small, set it to 10 miles
                if distance < 0.01:
                    distance = 10
            
            lines_data.append({
                'Network_Lines': line_num,
                'Start_Zone': from_bus,
                'End_Zone': to_bus,
                'distance_miles': distance
            })
            line_num += 1
    
    # Determine the maximum number of rows needed
    max_rows = max(len(zone_names), len(lines_data))
    
    # Create the combined data with zones and lines side by side
    network_data = []
    for i in range(max_rows):
        # Get substation name for this zone
        substation_name = None
        if i < len(zone_names):
            zone_name = zone_names[i]
            # Extract the number after 'z' or 'z_'
            zone_num = int(zone_name.replace('z_', '').replace('z', ''))
            # Look up in mapping
            if zone_num in bus_to_load:
                substation_name = bus_to_load[zone_num]
        
        row_data = {
            '': substation_name,
            'Network_zones': zone_names[i] if i < len(zone_names) else None,
            'Network_Lines': lines_data[i]['Network_Lines'] if i < len(lines_data) else None,
            'Start_Zone': lines_data[i]['Start_Zone'] if i < len(lines_data) else None,
            'End_Zone': lines_data[i]['End_Zone'] if i < len(lines_data) else None,
            'distance_miles': lines_data[i]['distance_miles'] if i < len(lines_data) else None
        }
        network_data.append(row_data)
    
    # Create output dataframe
    output_df = pd.DataFrame(network_data)
    
    # Save to output file
    print(f"Saving network file: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    print("Network file created!")

if __name__ == "__main__":
    # Define base paths using pathlib
    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]
    
    # Define input and output directories
    input_path = parent_dir / "in"
    output_path = parent_dir / "out"
    
    # Define files for create_demand_file
    demand_input_file = output_path / "residential_profiles" / "Load_data_6716_bus_combined.csv"
    demand_output_file = output_path / "genx_inputs" / "Demand_data.csv"
    mapping_file = input_path / "Texas7k_LoadGenMap.xlsx"
    
    # Call the demand file processing function
    create_demand_file(demand_input_file, demand_output_file, mapping_file)
    
    # Define files for create_network_file
    network_input_file = input_path / "Texas7kNetwork.csv"
    network_output_file = output_path / "genx_inputs" / "Network.csv"
    network_data_file = input_path / "texas_7k_bus_data.csv"
    
    # Call the network file processing function
    create_network_file(network_input_file, network_output_file, mapping_file, network_data_file)