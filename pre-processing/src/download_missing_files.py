import os
import csv
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

# Configuration
current_path = Path(__file__).resolve()
parent_dir = current_path.parents[1]
output_root = parent_dir / "out" / "consumption_files"

CSV_FILE = output_root / "missing_files.csv"  # CSV with a column of missing building IDs
OUTPUT_DIR = output_root        # Folder to save parquet files
BUCKET_NAME = "oedi-data-lake"         # Public S3 bucket
STATE = "TX"                            # State folder in S3
S3_PREFIX = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1/timeseries_individual_buildings/by_state/upgrade=0"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize anonymous S3 client
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

downloaded = 0
missing = []

# Read CSV of building IDs
all_ids = []
with open(CSV_FILE, newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # Skip the first line
    for row in reader:
        if row:
            try:
                # Convert to int to avoid floats like 1234.0
                bldg_id = int(float(row[0].strip()))
                all_ids.append(bldg_id)
            except ValueError:
                print(f"Skipping invalid ID: {row[0]}")

# Download each file
for bldg_id in all_ids:
    key = f"{S3_PREFIX}/state={STATE}/{bldg_id}-0.parquet"
    save_path = os.path.join(OUTPUT_DIR, f"{bldg_id}-0.parquet")

    try:
        s3.download_file(BUCKET_NAME, key, save_path)
        print(f"Downloaded: {bldg_id}-0.parquet")
        downloaded += 1
    except Exception as e:
        print(f"Missing or failed: {bldg_id}-0.parquet ({e})")
        missing.append(bldg_id)

print(f"\nDownload complete. Successfully downloaded: {downloaded}")
if missing:
    print(f"Missing files: {len(missing)}")
    # Optionally save missing IDs to a CSV
    with open("still_missing.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for mid in missing:
            writer.writero
