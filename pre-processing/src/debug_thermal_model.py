import json
import csv
import os

R2_THRESHOLD = 0.9
COEFF_THRESHOLD = 1e-8

input_path = r"D:\shared\ercot_project\out\thermal_model\baseline_internal_gains\regression_coeff_baseline_internal_gains.json"
output_path = r"D:\shared\ercot_project\out\thermal_model\bad_resstock_bldgs_ihg.csv"

with open(input_path, "r") as f:
    data = json.load(f)

low_r2_count = 0
small_coeff_count = 0
both_count = 0
total_problematic = 0

# Dynamic column names
r2_col_name = f"r_2 < {R2_THRESHOLD}"
coeff_col_name = f"k_i < {COEFF_THRESHOLD}"

rows = []

for key, values in data.items():
    r2 = values.get("r_squared")
    coeffs = [values.get("k1"), values.get("k2"), values.get("k3")]

    has_low_r2 = (r2 is None) or (r2 < R2_THRESHOLD)
    has_small_coeff = any(c is not None and abs(c) < COEFF_THRESHOLD for c in coeffs)

    # Counts
    if has_low_r2:
        low_r2_count += 1
    if has_small_coeff:
        small_coeff_count += 1
    if has_low_r2 and has_small_coeff:
        both_count += 1

    # Only store problematic ones
    if has_low_r2 or has_small_coeff:
        total_problematic += 1

        rows.append([
            key,
            int(has_low_r2),
            int(has_small_coeff),
            int(has_low_r2 and has_small_coeff)
        ])

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Write CSV
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["bldg_id", r2_col_name, coeff_col_name, "both"])
    writer.writerows(rows)

# Output summary
print("==== RESULTS ====")
print(f"Total entries: {len(data)}")
print(f"r_squared < {R2_THRESHOLD}: {low_r2_count}")
print(f"Any coefficient < {COEFF_THRESHOLD}: {small_coeff_count}")
print(f"Both conditions: {both_count}")
print(f"Total problematic (either condition): {total_problematic}")
print(f"\nCSV saved to: {output_path}")