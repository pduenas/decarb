import boto3
from botocore.config import Config
from botocore import UNSIGNED
import logging
import pandas as pd
import re
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)

# S3 parameters to change dynamically
year = '2025'  # Change the year here
release_version = 'resstock_amy2018_release_1'  # Change the release version here
state = 'TX'
limit = 10000  # Set a large value 

# Construct the prefix dynamically based on the parameters
prefix = f'nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/{year}/{release_version}/timeseries_individual_buildings/by_state/upgrade=0/state={state}/'

# Initialize boto3 S3 client for unsigned (anonymous) access
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# S3 Bucket name
bucket_name = 'oedi-data-lake'

# Directory to save the CSV file
save_path = Path(__file__).parents[1] / 'in'

# Function to list all objects in the S3 bucket under the given prefix with pagination
def list_s3_objects(bucket, prefix):
    object_keys = []
    try:
        continuation_token = None
        
        while True:
            # List objects with a large MaxKeys value (but pagination will handle more than 1000)
            list_params = {'Bucket': bucket, 'Prefix': prefix, 'MaxKeys': limit}
            if continuation_token:
                list_params['ContinuationToken'] = continuation_token
            
            # List objects from S3
            response = s3_client.list_objects_v2(**list_params)
            
            # Append the object keys to the list
            if 'Contents' in response:
                object_keys.extend([obj['Key'] for obj in response['Contents']])
            
            # Check if there's another page of results
            continuation_token = response.get('NextContinuationToken')
            if not continuation_token:
                break  # Exit loop if no more pages
        
        logging.info(f"Total objects found: {len(object_keys)}")
        return object_keys
        
    except Exception as e:
        logging.error(f"Error occurred while listing S3 objects: {e}")
        return []

# Extract building IDs from the S3 keys using regular expression
def extract_building_ids(object_keys):
    building_ids = []
    pattern = re.compile(r'(\d+)-\d+\.parquet')  # Regex to capture building ID (e.g., 1234-5678.parquet)
    
    for key in object_keys:
        match = pattern.search(key)
        if match:
            building_ids.append(match.group(1))  # Extract the building ID (e.g., '1234')
    
    return building_ids

# Main function to execute the logic
if __name__ == '__main__':
    # List S3 objects from the specified bucket and prefix (fetch all objects)
    object_keys = list_s3_objects(bucket_name, prefix)

    # If object keys are found, extract the building IDs
    if object_keys:
        building_ids = extract_building_ids(object_keys)

        if building_ids:
            # Ensure the output directory exists before saving the CSV
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save the building IDs into a CSV file
            df = pd.DataFrame(building_ids, columns=['bldg_id'])
            output_file = save_path / 'valid_bldg_ids_tx.csv'
            df.to_csv(output_file, index=False)
            logging.info(f"CSV saved at: {output_file}")
        else:
            logging.info("No building IDs extracted from the objects.")
    else:
        logging.info("No object keys found in the specified S3 prefix.")
