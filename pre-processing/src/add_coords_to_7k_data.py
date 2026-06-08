import pandas as pd
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from pathlib import Path

# Set up file paths
current_path = Path(__file__).resolve()
parent_dir = current_path.parents[1]
input_file = parent_dir / "in" / "texas7k_data.xlsx"
output_file = parent_dir / "in" / "texas_7k_bus_data.csv"

# Read the Excel file
df = pd.read_excel(input_file, sheet_name="bus")

def clean_bus_name(bus_name):
    """
    Remove leading/trailing apostrophes, numbers, and trailing spaces
    from bus_name to extract Texas location names
    """
    if pd.isna(bus_name):
        return ""
    
    # Convert to string
    cleaned = str(bus_name)
    
    # Remove leading and trailing apostrophes
    cleaned = cleaned.strip("'")
    
    # Remove all numbers
    cleaned = re.sub(r'\d+', '', cleaned)
    
    # Remove trailing spaces
    cleaned = cleaned.rstrip()
    
    return cleaned

def get_coordinates(location_name, state="Texas", country="USA"):
    """
    Get latitude and longitude for a given location name
    Returns tuple of (latitude, longitude) or (None, None) if not found
    """
    if not location_name or location_name.strip() == "":
        return None, None
    
    # Initialize geocoder
    geolocator = Nominatim(user_agent="texas_bus_geocoder")
    
    try:
        # Search with state and country for better accuracy
        full_query = f"{location_name}, {state}, {country}"
        location = geolocator.geocode(full_query, timeout=10)
        
        if location:
            return location.latitude, location.longitude
        else:
            # Try without country if first attempt fails
            location = geolocator.geocode(f"{location_name}, {state}", timeout=10)
            if location:
                return location.latitude, location.longitude
            return None, None
            
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Error geocoding {location_name}: {e}")
        return None, None

# Apply the cleaning function to create new column
print("Cleaning bus names...")
df['bus_loc'] = df['bus_name'].apply(clean_bus_name)

# Get unique locations (excluding empty strings)
unique_locations = df['bus_loc'][df['bus_loc'] != ""].unique()
print(f"\nFound {len(unique_locations)} unique locations out of {len(df)} total rows")
print(f"Estimated time: {len(unique_locations) * 1.2 / 60:.1f} minutes")

# Create a cache dictionary for coordinates
location_cache = {}

print("\nGeocoding unique locations and mapping to buses...")
for idx, location in enumerate(unique_locations):
    lat, lon = get_coordinates(location)
    location_cache[location] = {'lat': lat, 'lon': lon}
    
    # Print status for this location
    status = f"Location {idx+1}/{len(unique_locations)}: '{location}' located at {lat}, {lon}"
    if lat is None:
        status = f"Location {idx+1}/{len(unique_locations)}: '{location}' - NOT FOUND"
    print(status)
    
    # Print all buses that use this location
    matching_buses = df[df['bus_loc'] == location]
    for _, row in matching_buses.iterrows():
        bus_num = row.get('bus_num', row.name)  # Use bus_num if it exists, otherwise use index
        bus_name = row['bus_name']
        if pd.notna(lat):
            print(f"  → {bus_num}={bus_name} is in {location} and located at {lat} and {lon}")
        else:
            print(f"  → {bus_num}={bus_name} is in {location} - coordinates NOT FOUND")
    
    print()  # Blank line for readability
    
    # Respect rate limit: 1 request per second
    time.sleep(1)

# Map the cached coordinates back to the dataframe
print("Finalizing dataframe...")
df['bus_lat'] = df['bus_loc'].map(lambda x: location_cache.get(x, {}).get('lat'))
df['bus_lon'] = df['bus_loc'].map(lambda x: location_cache.get(x, {}).get('lon'))

# Save to CSV
df.to_csv(output_file, index=False)

print(f"\nProcessing complete! File saved as '{output_file}'")
print(f"\nFirst few rows of the result:")
print(df[['bus_name', 'bus_loc', 'bus_lat', 'bus_lon']].head(10))
print(f"\nTotal rows processed: {len(df)}")
print(f"Unique locations: {len(unique_locations)}")
print(f"Successfully geocoded: {df['bus_lat'].notna().sum()}/{len(df)} locations")
print(f"Failed to geocode: {df['bus_lat'].isna().sum()} locations")