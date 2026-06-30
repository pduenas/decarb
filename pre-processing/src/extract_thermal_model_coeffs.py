"""
aggregate_regression_coeffs.py

Scans a root directory for electrical grid zone folders (P<number>U or P<number>R),
reads regression coefficient CSVs from each zone's input_timeseries subfolder,
and produces one JSON file per file type (baseline, lambda10, lambda100).

Each JSON maps bldg_id -> {k1, k2, k3, mse, r_squared}.

If the same bldg_id appears in multiple zones with differing k1/k2/k3 values,
a warning is printed and the entry with the lower MSE is kept.

The script is placed inside the root zone folder. Output JSONs are written
one level up (Path(__file__).parents[1]).
"""

import re
import json
import csv
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

ZONE_PATTERN = re.compile(r'^P\d+[UR]$')

FILE_TYPES = {
    "baseline":  "{zone}_regression_coeff_baseline.csv",
    "lambda10":  "{zone}_regression_coeff_regularized_baseline_lambda_10.0.csv",
    "lambda100": "{zone}_regression_coeff_regularized_baseline_lambda_100.0.csv",
}

REQUIRED_COLS = {"bldg_id", "k1", "k2", "k3", "mse", "r_squared"}

INPUT_SUBDIR = "input_timeseries"


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_zone_folders(root_dir: Path) -> list[Path]:
    """Return sorted list of zone folder paths matching P<number>U or P<number>R."""
    return sorted(
        p for p in root_dir.iterdir()
        if p.is_dir() and ZONE_PATTERN.match(p.name)
    )


def read_csv_columns(filepath: Path) -> list[dict]:
    """
    Read a CSV file and return only the REQUIRED_COLS columns as a list of dicts.
    Values for k1/k2/k3/mse/r_squared are cast to float; bldg_id is cast to str.
    Rows missing any required column are skipped with a warning.
    """
    rows = []
    text = filepath.read_text(encoding="utf-8")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","

    with filepath.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        headers = reader.fieldnames or []

        missing = REQUIRED_COLS - set(headers)
        if missing:
            print(f"  [WARNING] File '{filepath}' is missing columns: {missing}. Skipping file.")
            return rows

        for lineno, row in enumerate(reader, start=2):
            try:
                rows.append({
                    "bldg_id":   str(row["bldg_id"]).strip(),
                    "k1":        float(row["k1"]),
                    "k2":        float(row["k2"]),
                    "k3":        float(row["k3"]),
                    "mse":       float(row["mse"]),
                    "r_squared": float(row["r_squared"]),
                })
            except (ValueError, KeyError) as exc:
                print(f"  [WARNING] Skipping line {lineno} in '{filepath}': {exc}")

    return rows


def values_differ(a: float, b: float, rel_tol: float = 0.1) -> bool:
    """
    Return True if two floats differ by more than rel_tol (10% by default).
    Compares relative to the smaller absolute value.
    """
    if a == b:
        return False
    smaller = min(abs(a), abs(b))
    if smaller == 0.0:
        # If one is zero and the other isn't, they differ
        return True
    return abs(a - b) / smaller > rel_tol


# ── Core logic ────────────────────────────────────────────────────────────────

def build_mapping_for_file_type(
    zone_folders: list[Path],
    file_type_key: str,
    filename_template: str,
) -> dict:
    """
    Iterate over all zone folders, read the CSV for this file type, and merge
    into one dict keyed by bldg_id.  Conflicts are resolved by lower MSE.
    """
    merged: dict[str, dict] = {}   # bldg_id -> {k1, k2, k3, mse, r_squared, _zone}

    for zone_path in zone_folders:
        zone_name = zone_path.name
        csv_path  = zone_path / INPUT_SUBDIR / filename_template.format(zone=zone_name)

        if not csv_path.is_file():
            print(f"  [INFO] File not found, skipping: '{csv_path}'")
            continue

        rows = read_csv_columns(csv_path)

        for row in rows:
            bid   = row["bldg_id"]
            entry = {
                "k1":        row["k1"],
                "k2":        row["k2"],
                "k3":        row["k3"],
                "mse":       row["mse"],
                "r_squared": row["r_squared"],
            }

            if bid not in merged:
                merged[bid] = {**entry, "_zone": zone_name}
                continue

            existing = merged[bid]

            # Only flag a conflict if the MSEs are meaningfully different
            if values_differ(entry["mse"], existing["mse"]):
                coeff_conflict = any(
                    values_differ(entry[c], existing[c])
                    for c in ("k1", "k2", "k3")
                )
                if coeff_conflict:
                    print(
                        f"  [WARNING] [{file_type_key}] bldg_id={bid} has different "
                        f"k1/k2/k3 in zone '{zone_name}' vs zone '{existing['_zone']}' "
                        f"(>10% deviation). Keeping entry with lower MSE."
                    )
                    print(
                        f"           Zone '{existing['_zone']}': "
                        f"k1={existing['k1']:.6e}, k2={existing['k2']:.6e}, "
                        f"k3={existing['k3']:.6e}, mse={existing['mse']:.6e}"
                    )
                    print(
                        f"           Zone '{zone_name}': "
                        f"k1={entry['k1']:.6e}, k2={entry['k2']:.6e}, "
                        f"k3={entry['k3']:.6e}, mse={entry['mse']:.6e}"
                    )

                # Keep the entry with the lower MSE
                if entry["mse"] < existing["mse"]:
                    merged[bid] = {**entry, "_zone": zone_name}

    # Strip the internal bookkeeping field before returning
    return {
        bid: {k: v for k, v in entry.items() if k != "_zone"}
        for bid, entry in merged.items()
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_metric_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of floats."""
    vals   = sorted(values)
    n      = len(vals)
    mean   = statistics.mean(vals)
    median = statistics.median(vals)
    std    = statistics.stdev(vals) if n > 1 else 0.0
    return {
        "n":                  n,
        "mean":               mean,
        "median":             median,
        "std_dev":            std,
        "min":                vals[0],
        "max":                vals[-1],
        "mean_minus_1std":    mean - std,
        "mean_plus_1std":     mean + std,
        "mean_minus_2std":    mean - 2 * std,
        "mean_plus_2std":     mean + 2 * std,
        "median_minus_1std":  median - std,
        "median_plus_1std":   median + std,
        "median_minus_2std":  median - 2 * std,
        "median_plus_2std":   median + 2 * std,
    }


def print_and_collect_stats(label: str, mapping: dict) -> list[dict] | None:
    """
    Compute k1, k2, k3, MSE, and R² statistics, print summary tables, and return
    a list of five flat dicts (one per metric) ready to write to CSV.
    """
    if not mapping:
        print(f"  No data for {label}.")
        return None

    k1_values  = [entry["k1"]        for entry in mapping.values()]
    k2_values  = [entry["k2"]        for entry in mapping.values()]
    k3_values  = [entry["k3"]        for entry in mapping.values()]
    mse_values = [entry["mse"]       for entry in mapping.values()]
    r2_values  = [entry["r_squared"] for entry in mapping.values()]

    k1_stats  = compute_metric_stats(k1_values)
    k2_stats  = compute_metric_stats(k2_values)
    k3_stats  = compute_metric_stats(k3_values)
    mse_stats = compute_metric_stats(mse_values)
    r2_stats  = compute_metric_stats(r2_values)

    col = 14
    for metric_label, stats in [
        ("k1", k1_stats),
        ("k2", k2_stats),
        ("k3", k3_stats),
        ("MSE", mse_stats),
        ("R²", r2_stats),
    ]:
        n = stats["n"]
        print(f"\n  {metric_label} statistics — {label} ({n} buildings)")
        print(f"  {'─' * 42}")
        print(f"  {'Mean':<14} {stats['mean']:{col}.6e}")
        print(f"  {'Median':<14} {stats['median']:{col}.6e}")
        print(f"  {'Std dev':<14} {stats['std_dev']:{col}.6e}")
        print(f"  {'Min':<14} {stats['min']:{col}.6e}")
        print(f"  {'Max':<14} {stats['max']:{col}.6e}")
        print(f"  {'μ - 1σ':<14} {stats['mean_minus_1std']:{col}.6e}")
        print(f"  {'μ + 1σ':<14} {stats['mean_plus_1std']:{col}.6e}")
        print(f"  {'μ - 2σ':<14} {stats['mean_minus_2std']:{col}.6e}")
        print(f"  {'μ + 2σ':<14} {stats['mean_plus_2std']:{col}.6e}")
        print(f"  {'median - 1σ':<14} {stats['median_minus_1std']:{col}.6e}")
        print(f"  {'median + 1σ':<14} {stats['median_plus_1std']:{col}.6e}")
        print(f"  {'median - 2σ':<14} {stats['median_minus_2std']:{col}.6e}")
        print(f"  {'median + 2σ':<14} {stats['median_plus_2std']:{col}.6e}")
        print(f"  {'─' * 42}")

    return [
        {"method": label, "metric": "k1",        **k1_stats},
        {"method": label, "metric": "k2",        **k2_stats},
        {"method": label, "metric": "k3",        **k3_stats},
        {"method": label, "metric": "mse",       **mse_stats},
        {"method": label, "metric": "r_squared", **r2_stats},
    ]




def plot_histograms(all_mappings: dict, output_dir: Path) -> None:
    """
    Create histogram plots for k1, k2, k3, mse, and r_squared.
    Each metric gets its own figure with 3 subplots (baseline, lambda10, lambda100).
    
    Args:
        all_mappings: dict with keys 'baseline', 'lambda10', 'lambda100', 
                     each containing the building mapping dict
        output_dir: directory to save the histogram PNG files
    """
    metrics = ["k1", "k2", "k3", "mse", "r_squared"]
    methods = ["baseline", "lambda10", "lambda100"]
    method_labels = {
        "baseline": "Baseline",
        "lambda10": r"Regularized - $\lambda = 10$",
        "lambda100": r"Regularized - $\lambda = 100$",
    }
    
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Better title formatting
        if metric == "mse":
            title = "MSE Distribution"
        elif metric == "r_squared":
            title = r"$R^2$ Distribution"
        else:
            title = f"{metric.upper()} Distribution"
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        for idx, method in enumerate(methods):
            ax = axes[idx]
            mapping = all_mappings.get(method, {})
            
            if not mapping:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(method_labels[method])
                continue
            
            values = np.array([entry[metric] for entry in mapping.values()])
            
            # Use percentile-based limits to exclude extreme outliers
            # but keep reasonable range for r_squared
            if metric == "r_squared":
                # Clip to [0, 1] or use 1st-99th percentile if that's tighter
                p1, p99 = np.percentile(values, [1, 99])
                xlim = (max(0, p1 - 0.05), min(1, p99 + 0.05))
            else:
                # For other metrics, use 1st-99th percentile with 5% padding
                p1, p99 = np.percentile(values, [1, 99])
                range_width = p99 - p1
                xlim = (p1 - 0.05 * range_width, p99 + 0.05 * range_width)
            
            # Filter values to plot range for cleaner histograms
            plot_values = values[(values >= xlim[0]) & (values <= xlim[1])]
            
            # Create histogram
            ax.hist(plot_values, bins=50, edgecolor='black', alpha=0.7)
            ax.set_title(method_labels[method], fontsize=12)
            ax.set_xlabel(metric if metric != "r_squared" else r"$R^2$")
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xlim(xlim)
            
            # Add stats text (compute on ALL values, not just plotted ones)
            mean_val = np.mean(values)
            median_val = np.median(values)
            
            # Only draw lines if they're in the visible range
            if xlim[0] <= mean_val <= xlim[1]:
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                          label=f'Mean: {mean_val:.2e}')
            if xlim[0] <= median_val <= xlim[1]:
                ax.axvline(median_val, color='blue', linestyle='--', linewidth=1.5, 
                          label=f'Median: {median_val:.2e}')
            
            ax.legend(fontsize=8)
            
            # Force scientific notation for k1, k2, k3 x-axis
            if metric in ["k1", "k2", "k3"]:
                ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            # Print outliers info to terminal
            n_excluded = len(values) - len(plot_values)
            if n_excluded > 0:
                print(f"    [{method_labels[method]}] {metric}: excluded {n_excluded} outliers")
        
        plt.tight_layout()
        output_path = output_dir / f'histogram_{metric}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved histogram → '{output_path}'")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root_dir   = Path(__file__).parents[1] / "out"
    output_dir = root_dir / "thermal_model"
    output_dir.mkdir(exist_ok=True)

    zone_folders = find_zone_folders(root_dir)
    if not zone_folders:
        raise SystemExit(f"ERROR: No zone folders (P<n>U / P<n>R) found in '{root_dir}'")

    print(f"Found {len(zone_folders)} zone folder(s) in '{root_dir}'.")
    print(f"Output directory: '{output_dir}'.")

    all_stats = []
    all_mappings = {}

    for file_type_key, filename_template in FILE_TYPES.items():
        print(f"\n── Processing file type: {file_type_key} ──")

        mapping = build_mapping_for_file_type(zone_folders, file_type_key, filename_template)
        all_mappings[file_type_key] = mapping

        output_path = output_dir / f"regression_coeff_{file_type_key}.json"
        output_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

        print(f"  Wrote {len(mapping)} building(s) → '{output_path}'")

        rows = print_and_collect_stats(file_type_key, mapping)
        if rows:
            all_stats.extend(rows)

    # Write stats CSV — one row per (method, metric) combination
    if all_stats:
        csv_path   = output_dir / "regression_coeff_stats.csv"
        fieldnames = ["method", "metric", "n", "mean", "median", "std_dev",
                      "min", "max", "mean_minus_1std", "mean_plus_1std",
                      "mean_minus_2std", "mean_plus_2std",
                      "median_minus_1std", "median_plus_1std",
                      "median_minus_2std", "median_plus_2std"]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"\n  Statistics saved → '{csv_path}'")

    # Generate histograms
    print("\n── Generating histograms ──")
    plot_histograms(all_mappings, output_dir)

    print("\nDone.")