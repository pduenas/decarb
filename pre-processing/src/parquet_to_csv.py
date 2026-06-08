import pandas as pd

def parquet_to_csv(parquet_path, csv_path):
    # Read the parquet file
    df = pd.read_parquet(parquet_path, engine="fastparquet")

    # Save to CSV
    df.to_csv(csv_path, index=False)

    print(f"Converted '{parquet_path}' → '{csv_path}'")

if __name__ == "__main__":
    # Example usage
    parquet_file = r"Z:\ercot_project\out\consumption_files\109086-0.parquet"
    csv_file = r"Z:\ercot_project\109086-0.csv"

    parquet_to_csv(parquet_file, csv_file)
