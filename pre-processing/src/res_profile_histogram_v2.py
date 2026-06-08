import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

JSON_PATH      = Path(__file__).parents[1] / "out" / "residential_profiles" / "sub_bldg_lookup.json"
PARQUET_PATH   = Path(__file__).parents[1] / "in" / "TX_upgrade0.parquet"
OUT_ROOT       = JSON_PATH.parent          # .../out/residential_profiles/
ALL_OUT        = OUT_ROOT / "all_ercot_stats"
SUBSET_OUT     = OUT_ROOT / "model_subset_stats"

ALL_OUT.mkdir(parents=True, exist_ok=True)
SUBSET_OUT.mkdir(parents=True, exist_ok=True)

# ── Column config ─────────────────────────────────────────────────────────────

COLUMNS = [
    "in.representative_income",
    "in.area_median_income",
    "in.ashrae_iecc_climate_zone_2004",
    "in.geometry_building_type_acs",
    "in.geometry_floor_area",
    "in.geometry_stories",
    "in.heating_fuel",
    "in.hvac_heating_type_and_fuel",
]

NUMERIC_COLUMNS = {
    "in.representative_income",
}

# Numeric columns whose bin edge labels should be displayed as integers
INTEGER_LABEL_COLUMNS = {
    "in.representative_income",
}

# Custom bins: {col: [(label, low, high), ...]} where ranges are [low, high)
# Use float('inf') for open-ended upper bound
CUSTOM_BIN_COLUMNS = {
    "in.geometry_stories": [
        ("1",   1, 2),
        ("2",   2, 3),
        ("3",   3, 4),
        ("4-8", 4, 8),
        ("8+",  8, float("inf")),
    ],
}

NUM_BINS = 5

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✅ Saved: {path}")

def load_parquet(path, columns):
    """Load parquet using bldg_id as the index."""
    import fastparquet
    pf = fastparquet.ParquetFile(str(path))
    available = pf.columns
    print(f"  Parquet columns available ({len(available)}): {available[:10]}{'...' if len(available) > 10 else ''}")
    valid = [c for c in columns if c in available]
    missing = [c for c in columns if c not in available]
    if missing:
        print(f"  [WARN] Columns not found, skipped:\n    " + "\n    ".join(missing))
    print(f"  Loading {len(valid)} matched columns: {valid}")
    df = pd.read_parquet(path, engine="fastparquet", columns=["bldg_id"] + valid)
    df = df.set_index("bldg_id")
    return df

# ── Lookup aggregation ────────────────────────────────────────────────────────

def aggregate_counts(data):
    """Returns {bldg_id: total_count_across_all_substations}."""
    aggregated = {}
    for sub_data in data.values():
        for id_, count in sub_data.items():
            aggregated[id_] = aggregated.get(id_, 0) + int(count)
    return aggregated

def aggregate_per_substation(data):
    """Returns {substation: total_count}."""
    return {
        sub: sum(int(c) for c in sub_data.values())
        for sub, sub_data in data.items()
    }

def get_all_building_ids(data):
    """Union of all building IDs across every substation."""
    ids = set()
    for sub_data in data.values():
        ids.update(sub_data.keys())
    return ids

# ── Statistics ────────────────────────────────────────────────────────────────

def compute_statistics(aggregated_counts, top_n=10, percentiles=(50, 75, 90, 95, 99)):
    values = list(aggregated_counts.values())
    arr = np.array(values)
    sorted_items = sorted(aggregated_counts.items(), key=lambda x: x[1], reverse=True)
    total = int(arr.sum())
    return {
        "total_buildings": total,
        "unique_ids": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "percentiles": {str(p): float(np.percentile(arr, p)) for p in percentiles},
        "top_n": sorted_items[:top_n],
        "bottom_n": sorted_items[-top_n:],
        # Percentage of total buildings represented by each top/bottom ID
        "top_n_pct": [(id_, count, round(100 * count / total, 4)) for id_, count in sorted_items[:top_n]],
        "bottom_n_pct": [(id_, count, round(100 * count / total, 4)) for id_, count in sorted_items[-top_n:]],
    }

def print_statistics(stats, top_n=10):
    total = stats["total_buildings"]
    print("  ===== Building Count Report =====")
    print(f"  Total buildings  : {total:,}")
    print(f"  Unique IDs       : {stats['unique_ids']:,}")
    print(f"  Mean             : {stats['mean']:.2f}")
    print(f"  Median           : {stats['median']:.2f}")
    print(f"  Std dev          : {stats['std']:.2f}")
    print(f"  Min              : {stats['min']}")
    print(f"  Max              : {stats['max']}")
    print("\n  Percentiles:")
    for p, v in stats["percentiles"].items():
        print(f"    p{p:<3}: {v:.1f}")
    print(f"\n  Top {top_n} IDs by count:")
    for id_, count, pct in stats["top_n_pct"]:
        print(f"    {id_:<20} {count:>8,}  ({pct:.2f}%)")
    print(f"\n  Bottom {top_n} IDs by count:")
    for id_, count, pct in stats["bottom_n_pct"]:
        print(f"    {id_:<20} {count:>8,}  ({pct:.2f}%)")
    print("  =================================")

# ── Bucketing ─────────────────────────────────────────────────────────────────

def bucketize_counts(aggregated_counts, bucket_size=200, max_value=2000):
    bucket_counts = {
        f"{s}-{s + bucket_size}": 0
        for s in range(0, max_value, bucket_size)
    }
    bucket_counts[f"{max_value}+"] = 0

    for total in aggregated_counts.values():
        if total > max_value:
            bucket_counts[f"{max_value}+"] += 1
            continue
        bucket_start = (total // bucket_size) * bucket_size
        if total % bucket_size == 0 and total != 0:
            bucket_start -= bucket_size
        bucket_counts[f"{bucket_start}-{bucket_start + bucket_size}"] += 1

    return bucket_counts

# ── Histogram builders ────────────────────────────────────────────────────────

def _add_percentages(bins, total=None):
    """Add a 'pct' key to each bin dict. Uses sum of counts if total not given."""
    if total is None:
        total = sum(b["count"] for b in bins)
    for b in bins:
        b["pct"] = round(100 * b["count"] / total, 4) if total > 0 else 0.0
    return bins

def histogram_numeric(series, n_bins=NUM_BINS, integer_labels=False):
    clean = series.dropna()
    counts, edges = np.histogram(clean, bins=n_bins)
    bins = [
        {
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "label": f"{int(edges[i])}–{int(edges[i + 1])}" if integer_labels else f"{edges[i]:.4g}–{edges[i + 1]:.4g}",
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]
    bins = _add_percentages(bins)
    return {"type": "numeric", "n_bins": n_bins, "bins": bins}

def histogram_custom_bins(series, bin_defs):
    """Bin a numeric series into explicitly defined ranges [(label, low, high), ...].
    Bounds are inclusive: [low, high]. Use float('inf') for open-ended upper bound.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    bins = []
    for label, low, high in bin_defs:
        if high == float("inf"):
            count = int((clean >= low).sum())
        else:
            count = int(((clean >= low) & (clean <= high)).sum())
        bins.append({
            "bin_start": float(low),
            "bin_end": None if high == float("inf") else float(high),
            "label": label,
            "count": count,
        })
    bins = _add_percentages(bins)
    return {"type": "custom_numeric", "n_bins": len(bins), "bins": bins}

def histogram_categorical(series):
    counts = series.dropna().astype(str).value_counts()
    bins = [{"label": label, "count": int(count)} for label, count in counts.items()]
    bins = _add_percentages(bins)
    return {"type": "categorical", "n_bins": len(bins), "bins": bins}

def build_histograms(df, n_bins=NUM_BINS):
    histograms = {}
    for col in df.columns:
        series = df[col]
        if col in CUSTOM_BIN_COLUMNS:
            histograms[col] = histogram_custom_bins(series, CUSTOM_BIN_COLUMNS[col])
        elif col in NUMERIC_COLUMNS:
            histograms[col] = histogram_numeric(series, n_bins, integer_labels=col in INTEGER_LABEL_COLUMNS)
        else:
            histograms[col] = histogram_categorical(series)
        print(f"    ✓ {col}  [{histograms[col]['type']}, {histograms[col]['n_bins']} bins]")
    return histograms

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_bucketed_histogram(bucket_counts, out_dir, prefix=""):
    labels, values = zip(*bucket_counts.items())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, values, edgecolor="black")
    ax.set_xlabel("Total Count per ID (Buckets)")
    ax.set_ylabel("Number of IDs")
    ax.set_title("Distribution of Aggregated ID Counts")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}bucketed_histogram.png", dpi=150)
    plt.close(fig)

def plot_cdf(aggregated_counts, out_dir, prefix=""):
    values = np.sort(list(aggregated_counts.values()))
    cdf = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(values, cdf, linewidth=1.5)
    ax.set_xlabel("Total Count per ID")
    ax.set_ylabel("Cumulative Fraction of IDs")
    ax.set_title("CDF of Aggregated ID Counts")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}cdf.png", dpi=150)
    plt.close(fig)


def plot_metadata_histograms(histograms, out_dir, prefix="", max_cat_labels=30):
    for col, info in histograms.items():
        bins = info["bins"]
        counts = [b["count"] for b in bins]
        pcts = [b["pct"] for b in bins]
        labels = [b["label"] for b in bins]
        n = len(bins)

        if info["type"] == "numeric":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            ax1.bar(range(n), counts, edgecolor="black")
            ax1.set_xticks(range(n))
            ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax1.set_ylabel("Count")
            ax1.set_title(f"{col} — Count")

            ax2.bar(range(n), pcts, edgecolor="black", color="steelblue")
            ax2.set_xticks(range(n))
            ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax2.set_ylabel("Percentage (%)")
            ax2.set_title(f"{col} — %")
        else:
            if n > max_cat_labels:
                bins = bins[:max_cat_labels]
                labels = [b["label"] for b in bins]
                counts = [b["count"] for b in bins]
                pcts = [b["pct"] for b in bins]
                title_suffix = f" (top {max_cat_labels} of {n})"
            else:
                title_suffix = ""

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(labels) * 0.3)))
            ax1.barh(labels[::-1], counts[::-1], edgecolor="black")
            ax1.set_xlabel("Count")
            ax1.set_title(col + title_suffix + " — Count")

            ax2.barh(labels[::-1], pcts[::-1], edgecolor="black", color="steelblue")
            ax2.set_xlabel("Percentage (%)")
            ax2.set_title(col + title_suffix + " — %")

        fig.tight_layout()
        safe_col = col.replace(".", "_").replace("/", "_")
        fig.savefig(out_dir / f"{prefix}{safe_col}.png", dpi=150)
        plt.close(fig)

# ── Analysis runners ──────────────────────────────────────────────────────────

def run_lookup_analysis(data, aggregated_counts, out_dir, prefix=""):
    """Bucketed histogram + CDF + top-N + substation plots from the lookup JSON."""
    print("\n── Lookup analysis ──────────────────────────────────────")
    stats = compute_statistics(aggregated_counts)
    print_statistics(stats)
    save_json(stats, out_dir / f"{prefix}lookup_stats.json")

    # Save scalar stats + percentiles as CSV
    csv_rows = {
        "total_buildings": stats["total_buildings"],
        "unique_ids": stats["unique_ids"],
        "mean": stats["mean"],
        "median": stats["median"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        **{f"p{p}": v for p, v in stats["percentiles"].items()},
    }
    pd.DataFrame([csv_rows]).to_csv(out_dir / f"{prefix}lookup_stats.csv", index=False)
    print(f"  ✅ Saved: {out_dir / f'{prefix}lookup_stats.csv'}")

    # Save top/bottom N with percentages as CSV
    top_rows = [{"rank": i+1, "id": id_, "count": count, "pct": pct}
                for i, (id_, count, pct) in enumerate(stats["top_n_pct"])]
    bottom_rows = [{"rank": i+1, "id": id_, "count": count, "pct": pct}
                   for i, (id_, count, pct) in enumerate(stats["bottom_n_pct"])]
    pd.DataFrame(top_rows).to_csv(out_dir / f"{prefix}top_ids.csv", index=False)
    pd.DataFrame(bottom_rows).to_csv(out_dir / f"{prefix}bottom_ids.csv", index=False)
    print(f"  ✅ Saved: {out_dir / f'{prefix}top_ids.csv'}, {out_dir / f'{prefix}bottom_ids.csv'}")

    bucket_counts = bucketize_counts(aggregated_counts)
    save_json(bucket_counts, out_dir / f"{prefix}bucket_counts.json")

    plot_bucketed_histogram(bucket_counts, out_dir, prefix=prefix)
    plot_cdf(aggregated_counts, out_dir, prefix=prefix)

def run_metadata_analysis(df, out_dir, prefix=""):
    """Metadata histogram plots + JSON + CSV for a given dataframe slice."""
    print("\n── Metadata histogram analysis ──────────────────────────")
    print(f"  Rows in slice: {len(df):,}")
    histograms = build_histograms(df)

    # Save JSON
    save_json(histograms, out_dir / f"{prefix}histogram_bins.json")

    # Save CSV — one row per (column, bin), now including pct
    csv_rows = []
    for col, info in histograms.items():
        for bin_ in info["bins"]:
            csv_rows.append({
                "column": col,
                "type": info["type"],
                "label": bin_["label"],
                "bin_start": bin_.get("bin_start", ""),
                "bin_end": bin_.get("bin_end", ""),
                "count": bin_["count"],
                "pct": bin_["pct"],
            })
    pd.DataFrame(csv_rows).to_csv(out_dir / f"{prefix}histogram_bins.csv", index=False)
    print(f"  ✅ Saved: {out_dir / f'{prefix}histogram_bins.csv'}")

    plot_metadata_histograms(histograms, out_dir, prefix=prefix)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load lookup JSON
    print(f"Loading lookup JSON: {JSON_PATH}")
    data = load_json(JSON_PATH)
    aggregated_counts = aggregate_counts(data)
    all_bldg_ids = get_all_building_ids(data)
    print(f"  {len(all_bldg_ids):,} unique building IDs in lookup.")

    # 2. Load parquet (full)
    print(f"\nLoading parquet: {PARQUET_PATH}")
    df_all = load_parquet(PARQUET_PATH, COLUMNS)
    print(f"  {len(df_all):,} rows loaded.")

    # 3. Filter to model subset
    # bldg_id index values are integers in parquet; lookup keys may be strings
    lookup_ids_as_index = pd.Index(all_bldg_ids).astype(df_all.index.dtype)
    df_subset = df_all[df_all.index.isin(lookup_ids_as_index)]
    print(f"  {len(df_subset):,} rows match lookup IDs (model subset).")

    # ── all_ercot_stats: metadata histograms for all parquet rows ─────────────
    print(f"\n{'='*55}")
    print("ALL ERCOT STATS")
    print(f"{'='*55}")
    run_metadata_analysis(df_all, ALL_OUT, prefix="all_ercot_")

    # ── model_subset_stats: lookup analysis + metadata histograms for subset ──
    print(f"\n{'='*55}")
    print("MODEL SUBSET STATS")
    print(f"{'='*55}")
    run_lookup_analysis(data, aggregated_counts, SUBSET_OUT, prefix="model_")
    run_metadata_analysis(df_subset, SUBSET_OUT, prefix="model_")


if __name__ == "__main__":
    main()