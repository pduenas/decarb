"""
ResStock Building Insulation Clustering
========================================
Groups buildings by insulative properties using K-Means on ordinally-encoded
categorical features. Only clusters buildings whose thermal-model regression
R² exceeds R2_THRESHOLD. Outputs cluster assignments and scatter plots using
both PCA and UMAP projections for comparison.

Requirements:
    pip install pandas pyarrow scikit-learn matplotlib seaborn umap-learn

Usage:
    python resstock_insulation_clustering.py
"""

import warnings
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  (all relative to the directory containing this script)
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parents[1]  # script lives in src/, project root is one level up
METADATA_PATH = BASE_DIR / "in" / "TX_upgrade0.parquet"
REGRESSION_PATH = (
    BASE_DIR / "out" / "thermal_model" / "baseline"
    / "regression_coeff_baseline.parquet"
)
OUT_DIR       = BASE_DIR / "out" / "building_clusters"

# ---------------------------------------------------------------------------
# Clustering settings
# ---------------------------------------------------------------------------
N_CLUSTERS      = 2      # 2 = good insulation vs bad insulation split
RUN_UMAP        = True   # set False to skip UMAP (faster, no umap-learn needed)
R2_THRESHOLD    = 0.85   # only cluster buildings with R² above this value

# UMAP performance settings
UMAP_N_JOBS      = -1     # CPU cores for UMAP: -1 = all cores, 1 = single-threaded
UMAP_N_NEIGHBORS = 10     # lower = faster; 10 is fine for visualization (default 15)
UMAP_SUBSAMPLE   = None   # fit UMAP on this many points, project the rest
                           # set to None to fit on all points (slower but exact)

# Feature selection — only used when N_CLUSTERS > 2.
# Keeps the top N features by eta² from an initial K-Means pass.
# Set to None to use all features.
TOP_N_FEATURES = 5

# Composite insulation score weights — used when N_CLUSTERS=2 for a clean
# good/bad split. Each standardised feature is multiplied by its weight and
# summed into a single score. Equal weights (1.0) = all features contribute
# equally. Set a feature to 0.0 to exclude it.
INSULATION_SCORE_WEIGHTS = {
    "r_ceiling":      1.0,
    "r_floor":        1.0,
    "r_roof":         1.0,
    "r_slab":         1.0,
    "r_wall":         1.0,
    "attic_type":     1.0,
    "air_tightness":  1.0,
    "window_quality": 1.0,
}

# ---------------------------------------------------------------------------
# Feature columns to use for clustering
# ---------------------------------------------------------------------------
INSULATION_FEATURES = [
    "in.insulation_ceiling",
    "in.insulation_floor",
    # "in.insulation_foundation_wall",  # excluded: eta² ~50, hijacks clustering
    # "in.insulation_rim_joist",        # excluded: eta² ~50, hijacks clustering
    "in.insulation_roof",
    "in.insulation_slab",
    "in.insulation_wall",
    "in.windows",
    "in.infiltration",              # air leakage – tightly related to envelope
    "in.geometry_foundation_type",  # structural context
    "in.geometry_attic_type",       # structural context
]

# ---------------------------------------------------------------------------
# R-value extraction helpers (to derive a numeric insulation quality score)
# ---------------------------------------------------------------------------
R_VALUE_PATTERN = re.compile(r"R-?(\d+(?:\.\d+)?)", re.IGNORECASE)

def extract_r_value(text: str) -> float:
    """Pull the first R-value number from a string, else return 0."""
    if pd.isna(text) or str(text).strip().lower() in ("none", "uninsulated", ""):
        return 0.0
    match = R_VALUE_PATTERN.search(str(text))
    return float(match.group(1)) if match else 0.0

R_VALUE_COLS = [
    "in.insulation_ceiling",
    "in.insulation_floor",
    # "in.insulation_foundation_wall",  # excluded: see INSULATION_FEATURES
    # "in.insulation_rim_joist",        # excluded: see INSULATION_FEATURES
    "in.insulation_roof",
    "in.insulation_slab",
    "in.insulation_wall",
]

# ACH50 extraction for infiltration
ACH50_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*ACH50", re.IGNORECASE)

def extract_ach50(text: str) -> float:
    """Extract ACH50 numeric value. Higher = leakier (worse)."""
    if pd.isna(text):
        return np.nan
    match = ACH50_PATTERN.search(str(text))
    return float(match.group(1)) if match else np.nan

# Window quality mapping (ordinal – roughly increasing insulation quality)
WINDOW_QUALITY = {
    "single": 1,
    "double, clear, metal": 2,
    "double, clear, non-metal": 3,
    "double, low-e, metal": 4,
    "double, low-e, non-metal": 5,
    "double, low-e, non-metal, air, m-gain": 5,
    "double, clear, metal, air": 3,
    "double, clear, metal, air, exterior clear storm": 3,
    "triple": 6,
    "triple, low-e": 7,
}

def map_window_quality(text: str) -> float:
    if pd.isna(text):
        return np.nan
    key = str(text).strip().lower()
    # Try exact match first
    if key in WINDOW_QUALITY:
        return float(WINDOW_QUALITY[key])
    # Partial match
    for pattern, val in WINDOW_QUALITY.items():
        if pattern in key:
            return float(val)
    return 2.0  # default: basic double-pane

# Foundation & attic type ordinal encoding
FOUNDATION_ORDER = {
    "slab": 1,
    "heated basement": 4,
    "conditioned basement": 4,
    "unheated basement": 3,
    "unconditioned basement": 3,
    "vented crawlspace": 2,
    "unvented crawlspace": 2,
    "ambient": 1,
    "none": 1,
}

ATTIC_ORDER = {
    "none": 1,
    "flat roof": 1,
    "cathedral ceiling": 2,
    "unvented attic": 3,
    "vented attic": 2,
    "conditioned attic": 4,
}

def map_ordinal(text: str, mapping: dict, default: float = 2.0) -> float:
    if pd.isna(text):
        return default
    key = str(text).strip().lower()
    for pattern, val in mapping.items():
        if pattern in key:
            return float(val)
    return default

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {path.name}")
    return df


def filter_by_r2(
    df: pd.DataFrame, regression_path: Path, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only rows whose bldg_id appears in the regression file with R² > threshold.

    Returns
    -------
    filtered : pd.DataFrame
        Metadata rows for buildings that passed the R² filter.
    reg_filtered : pd.DataFrame
        Full regression table (all columns) for those same buildings,
        used later to compute per-cluster coefficient statistics.
    """
    reg = pd.read_parquet(regression_path)
    print(f"Regression file: {len(reg):,} rows from {regression_path.name}")

    # Identify the R² column (handles r2, r_squared, r2_score, etc.)
    r2_col = next(
        (c for c in reg.columns if re.search(r"r.?2|r_squared", c, re.IGNORECASE)),
        None,
    )
    if r2_col is None:
        raise ValueError(
            f"Could not find an R² column in {regression_path.name}. "
            f"Available columns: {list(reg.columns)}"
        )
    print(f"Using R² column: '{r2_col}'")

    reg_filtered = reg[reg[r2_col] > threshold].copy()
    good_ids = reg_filtered["bldg_id"]
    print(f"Buildings with R² > {threshold}: {len(good_ids):,}")

    filtered = df[df["bldg_id"].isin(good_ids)].copy()
    print(f"Metadata rows after filter: {len(filtered):,}")
    return filtered, reg_filtered


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw categorical insulation columns into numeric feature matrix."""
    feat = pd.DataFrame(index=df.index)

    # R-values for each insulation surface
    for col in R_VALUE_COLS:
        if col in df.columns:
            short = col.replace("in.insulation_", "r_")
            feat[short] = df[col].apply(extract_r_value)
        else:
            print(f"  Warning: column '{col}' not found – skipping")

    # Air leakage (ACH50) – invert so higher = better (less leaky)
    if "in.infiltration" in df.columns:
        ach = df["in.infiltration"].apply(extract_ach50)
        max_ach = ach.quantile(0.95)  # clip outliers
        feat["air_tightness"] = (max_ach - ach.clip(upper=max_ach)).fillna(0)

    # Window quality
    if "in.windows" in df.columns:
        feat["window_quality"] = df["in.windows"].apply(map_window_quality)

    # Foundation and attic type
    if "in.geometry_foundation_type" in df.columns:
        feat["foundation_type"] = df["in.geometry_foundation_type"].apply(
            lambda x: map_ordinal(x, FOUNDATION_ORDER)
        )
    if "in.geometry_attic_type" in df.columns:
        feat["attic_type"] = df["in.geometry_attic_type"].apply(
            lambda x: map_ordinal(x, ATTIC_ORDER)
        )

    feat = feat.fillna(feat.median())
    return feat


def select_features(feat: pd.DataFrame, labels: np.ndarray,
                    top_n: int | None) -> list[str]:
    """Return the top_n feature names ranked by between-cluster eta² ratio.

    For each feature, computes:
        eta² = between-cluster variance / total variance
    A ratio near 0 means the feature looks the same in every cluster (useless).
    A ratio near 1 means the feature almost perfectly separates clusters.
    Set top_n=None to keep all features.
    """
    grand_mean = feat.mean()
    total_var  = feat.var()

    ratios = {}
    for col in feat.columns:
        between_var = (
            feat.groupby(labels)[col]
            .mean()
            .apply(lambda cm: (cm - grand_mean[col]) ** 2)
            .mean()
        )
        ratios[col] = between_var / total_var[col] if total_var[col] > 0 else 0.0

    ranked = sorted(ratios.items(), key=lambda x: -x[1])

    print("\nBetween-cluster variance ratios (eta²):")
    for col, r in ranked:
        print(f"  {col:<30s}  {r:.3f}")

    if top_n is None or top_n >= len(ranked):
        kept = [col for col, _ in ranked]
        print(f"\nKeeping all {len(kept)} features.")
    else:
        kept    = [col for col, _ in ranked[:top_n]]
        dropped = [col for col, _ in ranked[top_n:]]
        print(f"\nTop {top_n} features kept:    {kept}")
        print(f"Dropped ({len(dropped)}):  {dropped}")

    return kept


def select_k(X_scaled: np.ndarray, k_range: range) -> int:
    """Use silhouette score to suggest best k."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    best_k = max(scores, key=scores.get)
    print("\nSilhouette scores:")
    for k, s in scores.items():
        marker = " ← best" if k == best_k else ""
        print(f"  k={k}: {s:.4f}{marker}")
    return best_k


def compute_umap(
    X_scaled: np.ndarray,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    n_jobs: int = -1,
    subsample: int | None = 5_000,
) -> np.ndarray:
    """Fit UMAP and return 2D embedding for all points.

    If subsample is set and the dataset is larger than that threshold, UMAP is
    fit on a random subsample and the remaining points are projected with
    transform() — much faster with negligible quality loss for visualization.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is not installed. Run:  pip install umap-learn\n"
            "Or set RUN_UMAP = False to skip the UMAP figure."
        )

    n_total = X_scaled.shape[0]
    use_subsample = subsample is not None and n_total > subsample

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        metric="euclidean",
        n_jobs=n_jobs,
    )

    if use_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, size=subsample, replace=False)
        mask = np.zeros(n_total, dtype=bool)
        mask[idx] = True

        print(f"  Fitting UMAP on {subsample:,}-point subsample "
              f"(out of {n_total:,}) using {n_jobs if n_jobs != -1 else 'all'} core(s)...")
        reducer.fit(X_scaled[mask])

        coords = np.empty((n_total, 2), dtype=np.float32)
        coords[mask] = reducer.embedding_
        rest = ~mask
        print(f"  Projecting remaining {rest.sum():,} points...")
        coords[rest] = reducer.transform(X_scaled[rest])
    else:
        print(f"  Fitting UMAP on all {n_total:,} points "
              f"using {n_jobs if n_jobs != -1 else 'all'} core(s)...")
        coords = reducer.fit_transform(X_scaled)

    return coords


def compute_insulation_score(feat: pd.DataFrame,
                             weights: dict) -> np.ndarray:
    """Compute a single composite insulation quality score per building.

    Weighting strategy: std-based. Each feature is first standardised
    (zero mean, unit variance), then multiplied by its own standard deviation
    so that features which vary more across buildings contribute more to the
    final score. Features that are nearly constant everywhere barely move the
    needle. Any manual override weights in INSULATION_SCORE_WEIGHTS are
    multiplied on top (set to 1.0 to leave std-weighting untouched, 0.0 to
    exclude a feature entirely).
    Higher score = better insulated overall.
    """
    cols = [c for c in weights if c in feat.columns and weights[c] != 0.0]
    missing = [c for c in weights if c not in feat.columns and weights[c] != 0.0]
    if missing:
        print(f"  Warning: score features not in data, skipping: {missing}")

    sub = feat[cols].copy()
    stds = sub.std().replace(0, 1)

    # Standardise, then re-weight by std so high-variance features dominate
    sub_z = (sub - sub.mean()) / stds
    w = stds.values * np.array([weights[c] for c in cols])

    score = sub_z.values @ w

    print("  Feature std-weights used in composite score:")
    for col, sw in sorted(zip(cols, w), key=lambda x: -x[1]):
        print(f"    {col:<25s}  std-weight = {sw:.3f}")
    print(f"  Insulation score: min={score.min():.2f}  mean={score.mean():.2f}  max={score.max():.2f}")
    return score


def cluster_buildings(df: pd.DataFrame, n_clusters: int,
                      run_umap: bool = True):
    feat = engineer_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat)

    if n_clusters == 0:
        n_clusters = select_k(X_scaled, range(2, 11))
        print(f"\nAuto-selected k = {n_clusters}")

    if n_clusters == 2:
        # ── Good vs bad mode: cluster on a single composite score ────────
        print("\nN_CLUSTERS=2 → using composite insulation score for good/bad split.")
        score = compute_insulation_score(feat, INSULATION_SCORE_WEIGHTS)
        X_for_km = score.reshape(-1, 1)
        feat = pd.DataFrame({"insulation_score": score}, index=feat.index)
    else:
        # ── Multi-cluster mode: optional top-N feature selection ─────────
        km0 = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels0 = km0.fit_predict(X_scaled)
        if TOP_N_FEATURES is None:
            print("\nTOP_N_FEATURES = None — using all features.")
        else:
            keep_cols = select_features(feat, labels0, TOP_N_FEATURES)
            feat = feat[keep_cols]
        X_for_km = StandardScaler().fit_transform(feat)
        score = None

    # ── Final fit ────────────────────────────────────────────────────────
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_for_km)

    # Ensure cluster 0 = good (higher score), cluster 1 = bad
    if n_clusters == 2 and score is not None:
        if score[labels == 1].mean() > score[labels == 0].mean():
            labels = 1 - labels
            print("  Swapped labels: cluster 0 = good insulation, cluster 1 = bad.")
        print(f"  Cluster 0 (good) mean score: {score[labels == 0].mean():.2f}")
        print(f"  Cluster 1 (bad)  mean score: {score[labels == 1].mean():.2f}")

    sil = silhouette_score(X_for_km, labels)
    print(f"\nFinal model: k={n_clusters}, silhouette={sil:.4f}")

    # Add results back to original df
    result = df[["bldg_id"] + [c for c in INSULATION_FEATURES if c in df.columns]].copy()
    result["cluster"] = labels

    # ── PCA projection — always on the full original feature matrix ──────
    # feat_full preserves all engineered features for interpretable loadings,
    # regardless of whether clustering used a composite score or a subset.
    feat_full = engineer_features(df)
    X_pca = StandardScaler().fit_transform(feat_full)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_pca)
    result["pca_x"] = pca_coords[:, 0]
    result["pca_y"] = pca_coords[:, 1]
    var = pca.explained_variance_ratio_
    print(f"PCA: PC1 explains {var[0]*100:.1f}%, PC2 explains {var[1]*100:.1f}% of variance")

    # ── UMAP projection ──────────────────────────────────────────────────
    if run_umap:
        print("Running UMAP...")
        umap_coords = compute_umap(
            X_scaled,
            n_neighbors=UMAP_N_NEIGHBORS,
            n_jobs=UMAP_N_JOBS,
            subsample=UMAP_SUBSAMPLE,
        )
        result["umap_x"] = umap_coords[:, 0]
        result["umap_y"] = umap_coords[:, 1]
        print("UMAP complete.")
    else:
        result["umap_x"] = np.nan
        result["umap_y"] = np.nan

    return result, feat, feat_full, km, n_clusters, pca


def _scatter_ax(ax, x, y, labels, n_clusters, palette, title, xlabel, ylabel):
    """Helper: draw a coloured scatter on a given Axes."""
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(
            x[mask], y[mask],
            c=[palette[k]], label=f"Cluster {k}",
            alpha=0.75, s=60, edgecolors="white", linewidths=0.4,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Cluster", loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_pca(result: pd.DataFrame, feat: pd.DataFrame, n_clusters: int,
             pca, output_prefix: str) -> str:
    """Save PCA scatter + centroid heatmap side-by-side."""
    palette = sns.color_palette("tab10", n_clusters)
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("ResStock Insulation Clusters — PCA Projection", fontsize=14, fontweight="bold")

    _scatter_ax(
        axes[0],
        result["pca_x"].values, result["pca_y"].values,
        result["cluster"].values, n_clusters, palette,
        title="PCA: global structure",
        xlabel=f"PC 1 ({var[0]*100:.1f}% var.)",
        ylabel=f"PC 2 ({var[1]*100:.1f}% var.)",
    )

    # Loadings table — index directly from feat columns (the actual feature names)
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        index=feat.columns,
        columns=["PC 1", "PC 2"],
    )
    sns.heatmap(
        loadings, ax=axes[1], cmap="coolwarm", center=0,
        annot=True, fmt=".2f", linewidths=0.5,
        cbar_kws={"label": "Loading"},
    )
    axes[1].set_title("PCA loadings — which features drive each PC", fontsize=11)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    path = f"{output_prefix}_pca.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_umap(result: pd.DataFrame, n_clusters: int, output_prefix: str) -> str:
    """Save UMAP scatter (coloured by cluster + by top features)."""
    if result["umap_x"].isna().all():
        print("  UMAP coords not available — skipping UMAP figure.")
        return ""

    palette = sns.color_palette("tab10", n_clusters)
    feat_cols = [c for c in result.columns
                 if c not in ("bldg_id", "cluster", "pca_x", "pca_y", "umap_x", "umap_y")
                 and not c.startswith("in.")]

    # How many feature sub-plots fit alongside the main cluster plot?
    n_feat = min(len(feat_cols), 3)
    ncols = 1 + n_feat
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)
    axes = axes[0]  # flatten to 1-D array — safe regardless of ncols
    fig.suptitle("ResStock Insulation Clusters — UMAP Projection", fontsize=14, fontweight="bold")

    _scatter_ax(
        axes[0],
        result["umap_x"].values, result["umap_y"].values,
        result["cluster"].values, n_clusters, palette,
        title="UMAP: local neighbourhood structure",
        xlabel="UMAP 1", ylabel="UMAP 2",
    )

    # Side panels: colour by individual feature values
    for i, col in enumerate(feat_cols[:n_feat]):
        ax = axes[i + 1]
        sc = ax.scatter(
            result["umap_x"], result["umap_y"],
            c=result[col], cmap="YlOrRd",
            alpha=0.8, s=55, edgecolors="white", linewidths=0.3,
        )
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(col.replace("_", " "), fontsize=10)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = f"{output_prefix}_umap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_centroid_heatmap(feat: pd.DataFrame, result: pd.DataFrame,
                          km, n_clusters: int, output_prefix: str) -> str:
    """Save cluster-centroid heatmap (shared between PCA & UMAP runs).

    Computes centroids directly from feat grouped by cluster label, so the
    heatmap always reflects the full feature set regardless of what space
    K-Means was actually fit in (e.g. a 1D composite score).
    """
    palette = sns.color_palette("tab10", n_clusters)
    feat_labelled = feat.copy()
    feat_labelled["cluster"] = result["cluster"].values
    centroids_raw = (
        feat_labelled.groupby("cluster")[feat.columns.tolist()].mean()
    )
    centroids_raw.index = [f"Cluster {k}" for k in centroids_raw.index]
    fig, ax = plt.subplots(figsize=(12, max(4, n_clusters * 1.1)))
    sns.heatmap(
        centroids_raw, ax=ax, cmap="YlOrRd",
        annot=True, fmt=".1f", linewidths=0.5,
        cbar_kws={"label": "Standardised value"},
    )
    ax.set_title("Cluster centroids (standardised feature values)", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    plt.tight_layout()
    path = f"{output_prefix}_centroids.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_r_values(feat: pd.DataFrame, result: pd.DataFrame,
                  n_clusters: int, output_prefix: str) -> str:
    """Save per-cluster mean R-value bar chart."""
    r_cols = [c for c in feat.columns if c.startswith("r_")]
    cm = feat.copy()
    cm["cluster"] = result["cluster"].values
    means = cm.groupby("cluster")[r_cols].mean()

    means.T.plot(
        kind="bar", figsize=(12, 5),
        colormap="tab10", edgecolor="black", linewidth=0.5,
    )
    plt.title("Mean R-values by surface & cluster", fontsize=13, fontweight="bold")
    plt.xlabel("Insulation surface")
    plt.ylabel("Mean R-value")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = f"{output_prefix}_r_values.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_side_by_side(result: pd.DataFrame, n_clusters: int,
                      pca, output_prefix: str) -> str:
    """Save a single figure with PCA and UMAP side-by-side for easy comparison."""
    if result["umap_x"].isna().all():
        return ""
    palette = sns.color_palette("tab10", n_clusters)
    var = pca.explained_variance_ratio_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ResStock Insulation Clusters — PCA vs UMAP", fontsize=14, fontweight="bold")

    _scatter_ax(
        ax1,
        result["pca_x"].values, result["pca_y"].values,
        result["cluster"].values, n_clusters, palette,
        title="PCA  (global structure, axes interpretable)",
        xlabel=f"PC 1 ({var[0]*100:.1f}% var.)",
        ylabel=f"PC 2 ({var[1]*100:.1f}% var.)",
    )
    _scatter_ax(
        ax2,
        result["umap_x"].values, result["umap_y"].values,
        result["cluster"].values, n_clusters, palette,
        title="UMAP  (local neighbourhood, tighter clusters)",
        xlabel="UMAP 1", ylabel="UMAP 2",
    )

    plt.tight_layout()
    path = f"{output_prefix}_pca_vs_umap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    return path


def summarise_clusters(result: pd.DataFrame, feat: pd.DataFrame, n_clusters: int):
    feat_c = feat.copy()
    feat_c["cluster"] = result["cluster"].values

    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    for k in range(n_clusters):
        mask = feat_c["cluster"] == k
        sub = feat_c[mask].drop(columns="cluster")
        print(f"\n── Cluster {k}  ({mask.sum()} buildings) ──")
        print(sub.mean().round(2).to_string())
    print("=" * 60)


def cluster_coeff_stats(
    result: pd.DataFrame,
    reg: pd.DataFrame,
    n_clusters: int,
    output_prefix: str,
) -> str:
    """For each cluster, compute summary statistics on every regression coefficient.

    Detects coefficient columns automatically — any numeric column that is not
    'bldg_id' or an R² column is treated as a coefficient (k1, k2, k3, etc.).

    Outputs a CSV with a MultiIndex: (cluster, statistic) x coefficient.
    """
    # Identify k1, k2, k3 columns only
    coeff_cols = sorted([c for c in reg.columns if re.match(r"^k\d+$", c, re.IGNORECASE)])

    if not coeff_cols:
        print("  Warning: no k1/k2/k3 columns found in regression parquet — skipping stats.")
        print(f"  Available columns: {list(reg.columns)}")
        return ""

    print(f"\nComputing per-cluster stats for {len(coeff_cols)} coefficient(s): {coeff_cols}")

    # Merge cluster labels into regression table
    reg_labelled = reg[["bldg_id"] + coeff_cols].merge(
        result[["bldg_id", "cluster"]], on="bldg_id", how="inner"
    )

    stats = ["mean", "median", "std", "min", "max",
             lambda x: x.quantile(0.05),
             lambda x: x.quantile(0.95)]
    stat_names = ["mean", "median", "std", "min", "max", "p5", "p95"]

    # Build per-cluster agg dataframes: rows=coeff, cols=statistic
    frames = {}
    for cluster_id in range(n_clusters):
        sub = reg_labelled[reg_labelled["cluster"] == cluster_id][coeff_cols]
        agg = {}
        for fn, name in zip(stats, stat_names):
            agg[name] = sub.agg(fn)
        frames[f"cluster_{cluster_id}"] = pd.DataFrame(agg)  # shape: coeff x statistic

    # Reorder columns so identical statistics are side by side:
    # mean_cluster_0, mean_cluster_1, ... median_cluster_0, median_cluster_1, ...
    col_tuples = [
        (stat, f"cluster_{cluster_id}")
        for stat in stat_names
        for cluster_id in range(n_clusters)
    ]
    stats_df = pd.DataFrame(
        {(stat, clust): frames[clust][stat]
         for stat, clust in col_tuples},
    )
    stats_df.columns = pd.MultiIndex.from_tuples(col_tuples, names=["statistic", "cluster"])
    stats_df.index.name = "coefficient"

    csv_path = f"{output_prefix}_coeff_stats.csv"
    stats_df.to_csv(csv_path)
    print(f"Saved: {csv_path}")
    return csv_path


if __name__ == "__main__":
    # ── 1. Load metadata ────────────────────────────────────────────────
    df_full = load_parquet(METADATA_PATH)

    # ── 2. Filter to well-fitted buildings (R² > threshold) ─────────────
    df, reg = filter_by_r2(df_full, REGRESSION_PATH, R2_THRESHOLD)

    if len(df) == 0:
        raise RuntimeError(
            f"No buildings survived the R² > {R2_THRESHOLD} filter. "
            "Lower R2_THRESHOLD or check the regression parquet."
        )

    # ── 3. Cluster ───────────────────────────────────────────────────────
    result, feat, feat_full, km, n_clusters, pca = cluster_buildings(
        df, N_CLUSTERS, run_umap=RUN_UMAP
    )

    summarise_clusters(result, feat_full, n_clusters)

    # ── 4. Build output paths with technique + n_neighbors in filenames ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nn = f"nn{UMAP_N_NEIGHBORS}"   # e.g. nn10
    # Each plot function appends its own suffix (_pca.png, _umap.png, etc.)
    # so we just pass a base prefix that encodes the run parameters.
    pca_prefix  = str(OUT_DIR / f"k{n_clusters}")
    umap_prefix = str(OUT_DIR / f"k{n_clusters}_{nn}")
    shared_prefix = str(OUT_DIR / f"k{n_clusters}_shared")

    print("\nSaving figures...")
    plot_pca(result, feat_full, n_clusters, pca, output_prefix=pca_prefix)

    if RUN_UMAP:
        plot_umap(result, n_clusters, output_prefix=umap_prefix)
        plot_side_by_side(result, n_clusters, pca, output_prefix=umap_prefix)

    plot_centroid_heatmap(feat_full, result, km, n_clusters, output_prefix=shared_prefix)
    plot_r_values(feat_full, result, n_clusters, output_prefix=shared_prefix)

    # ── 5. Save cluster assignments CSV ─────────────────────────────────
    csv_name = f"k{n_clusters}_assignments.csv"
    out_csv = str(OUT_DIR / csv_name)
    result[["bldg_id", "cluster", "pca_x", "pca_y", "umap_x", "umap_y"]].to_csv(
        out_csv, index=False
    )

    # ── 6. Save per-cluster coefficient statistics CSV ───────────────────
    cluster_coeff_stats(result, reg, n_clusters, output_prefix=shared_prefix)

    # ── 7. Save cluster membership JSON ─────────────────────────────────
    import json
    membership = {
        f"cluster_{k}": result.loc[result["cluster"] == k, "bldg_id"].astype(int).tolist()
        for k in range(n_clusters)
    }
    json_name = f"k{n_clusters}_cluster_membership.json"
    json_path = OUT_DIR / json_name
    json_path.write_text(json.dumps(membership, indent=2))
    print(f"Saved: {json_path}")

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print(f"  k{n_clusters}_pca.png")
    if RUN_UMAP:
        print(f"  k{n_clusters}_{nn}_umap.png")
        print(f"  k{n_clusters}_{nn}_pca_vs_umap.png")
    print(f"  k{n_clusters}_shared_centroids.png")
    print(f"  k{n_clusters}_shared_r_values.png")
    print(f"  k{n_clusters}_shared_coeff_stats.csv")
    print(f"  {json_name}")
    print(f"  {csv_name}")