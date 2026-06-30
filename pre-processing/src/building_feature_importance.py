"""
Insulation Feature Importance
==============================
Explains how much each insulation property accounts for the variation in the
thermal-model regression coefficients k1, k2, k3 using a Random Forest
regressor. Outputs a feature importance bar chart for each coefficient.

Requirements:
    pip install pandas pyarrow scikit-learn matplotlib seaborn

Usage:
    python insulation_feature_importance.py
"""

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  (script lives in src/, project root is one level up)
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).resolve().parents[1]
METADATA_PATH   = BASE_DIR / "in" / "TX_upgrade0.parquet"
REGRESSION_PATH = (
    BASE_DIR / "out" / "thermal_model" / "baseline"
    / "regression_coeff_baseline.parquet"
)
OUT_DIR         = BASE_DIR / "out" / "building_clusters"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
R2_THRESHOLD = 0.85   # only include buildings with R² above this value
N_TREES      = 300    # random forest trees — more = more stable, slower
N_CV_FOLDS   = 5      # cross-validation folds for R² scoring
N_JOBS       = -1     # CPU cores: -1 = all

# ---------------------------------------------------------------------------
# Feature engineering  (identical to building_clusters.py)
# ---------------------------------------------------------------------------
R_VALUE_PATTERN = re.compile(r"R-?(\d+(?:\.\d+)?)", re.IGNORECASE)
ACH50_PATTERN   = re.compile(r"(\d+(?:\.\d+)?)\s*ACH50", re.IGNORECASE)

R_VALUE_COLS = [
    "in.insulation_ceiling",
    "in.insulation_floor",
    # "in.insulation_foundation_wall",  # excluded: binary split, hijacks model
    # "in.insulation_rim_joist",        # excluded: binary split, hijacks model
    "in.insulation_roof",
    "in.insulation_slab",
    "in.insulation_wall",
]

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

FOUNDATION_ORDER = {
    "slab": 1, "heated basement": 4, "conditioned basement": 4,
    "unheated basement": 3, "unconditioned basement": 3,
    "vented crawlspace": 2, "unvented crawlspace": 2,
    "ambient": 1, "none": 1,
}

ATTIC_ORDER = {
    "none": 1, "flat roof": 1, "cathedral ceiling": 2,
    "unvented attic": 3, "vented attic": 2, "conditioned attic": 4,
}


def extract_r_value(text: str) -> float:
    if pd.isna(text) or str(text).strip().lower() in ("none", "uninsulated", ""):
        return 0.0
    m = R_VALUE_PATTERN.search(str(text))
    return float(m.group(1)) if m else 0.0


def extract_ach50(text: str) -> float:
    if pd.isna(text):
        return np.nan
    m = ACH50_PATTERN.search(str(text))
    return float(m.group(1)) if m else np.nan


def map_window_quality(text: str) -> float:
    if pd.isna(text):
        return np.nan
    key = str(text).strip().lower()
    if key in WINDOW_QUALITY:
        return float(WINDOW_QUALITY[key])
    for pattern, val in WINDOW_QUALITY.items():
        if pattern in key:
            return float(val)
    return 2.0


def map_ordinal(text: str, mapping: dict, default: float = 2.0) -> float:
    if pd.isna(text):
        return default
    key = str(text).strip().lower()
    for pattern, val in mapping.items():
        if pattern in key:
            return float(val)
    return default


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    for col in R_VALUE_COLS:
        if col in df.columns:
            feat[col.replace("in.insulation_", "r_")] = df[col].apply(extract_r_value)
    if "in.infiltration" in df.columns:
        ach = df["in.infiltration"].apply(extract_ach50)
        max_ach = ach.quantile(0.95)
        feat["air_tightness"] = (max_ach - ach.clip(upper=max_ach)).fillna(0)
    if "in.windows" in df.columns:
        feat["window_quality"] = df["in.windows"].apply(map_window_quality)
    if "in.geometry_foundation_type" in df.columns:
        feat["foundation_type"] = df["in.geometry_foundation_type"].apply(
            lambda x: map_ordinal(x, FOUNDATION_ORDER)
        )
    if "in.geometry_attic_type" in df.columns:
        feat["attic_type"] = df["in.geometry_attic_type"].apply(
            lambda x: map_ordinal(x, ATTIC_ORDER)
        )
    return feat.fillna(feat.median())


# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
print("Loading data...")
meta = pd.read_parquet(METADATA_PATH)
print(f"  Metadata: {len(meta):,} rows")

reg = pd.read_parquet(REGRESSION_PATH)
print(f"  Regression: {len(reg):,} rows")

r2_col = next(
    (c for c in reg.columns if re.search(r"r.?2|r_squared", c, re.IGNORECASE)), None
)
if r2_col is None:
    raise ValueError(f"No R² column found. Columns: {list(reg.columns)}")
print(f"  R² column: '{r2_col}'")

reg = reg[reg[r2_col] > R2_THRESHOLD].copy()
print(f"  Buildings with R² > {R2_THRESHOLD}: {len(reg):,}")

# Identify k columns
k_cols = sorted([c for c in reg.columns if re.match(r"^k\d+$", c, re.IGNORECASE)])
if not k_cols:
    raise ValueError(f"No k1/k2/k3 columns found. Columns: {list(reg.columns)}")
print(f"  Coefficient columns: {k_cols}")

# Merge metadata with regression on bldg_id
merged = meta.merge(reg[["bldg_id"] + k_cols], on="bldg_id", how="inner")
print(f"  Buildings after merge: {len(merged):,}")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
print("\nEngineering insulation features...")
feat = engineer_features(merged)
print(f"  Features: {list(feat.columns)}")

# ---------------------------------------------------------------------------
# Random Forest regression + feature importance
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_LABELS = {
    "r_ceiling":      "Ceiling R-value",
    "r_floor":        "Floor R-value",
    "r_roof":         "Roof R-value",
    "r_slab":         "Slab R-value",
    "r_wall":         "Wall R-value",
    "air_tightness":  "Air tightness",
    "window_quality": "Window quality",
    "foundation_type":"Foundation type",
    "attic_type":     "Attic type",
}

X = feat.values
feature_names = [FEATURE_LABELS.get(c, c) for c in feat.columns]

results = {}
for k_col in k_cols:
    y = merged[k_col].values
    print(f"\nFitting Random Forest for {k_col}...")

    rf = RandomForestRegressor(
        n_estimators=N_TREES,
        max_features="sqrt",
        random_state=42,
        n_jobs=N_JOBS,
    )
    rf.fit(X, y)

    cv_r2 = cross_val_score(rf, X, y, cv=N_CV_FOLDS, scoring="r2", n_jobs=N_JOBS)
    print(f"  CV R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

    results[k_col] = {
        "importances": rf.feature_importances_,
        "cv_r2_mean": cv_r2.mean(),
        "cv_r2_std":  cv_r2.std(),
    }

# ---------------------------------------------------------------------------
# Plot: one subplot per k, bars sorted by importance
# ---------------------------------------------------------------------------
n_k = len(k_cols)
palette = sns.color_palette("tab10", n_k)

fig, axes = plt.subplots(1, n_k, figsize=(6 * n_k, 6), sharey=False)
if n_k == 1:
    axes = [axes]

fig.suptitle(
    "Insulation feature importance for thermal model coefficients",
    fontsize=14, fontweight="bold",
)

for ax, (k_col, res), color in zip(axes, results.items(), palette):
    imp   = res["importances"]
    order = np.argsort(imp)[::-1]
    names_sorted = [feature_names[i] for i in order]
    imp_sorted   = imp[order]

    bars = ax.barh(
        names_sorted[::-1], imp_sorted[::-1],
        color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    ax.set_title(
        f"{k_col}\n(CV R² = {res['cv_r2_mean']:.2f} ± {res['cv_r2_std']:.2f})",
        fontsize=11,
    )
    ax.set_xlabel("Feature importance (mean decrease in impurity)")
    ax.set_xlim(0, max(imp) * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels on bars
    for bar, val in zip(bars, imp_sorted[::-1]):
        ax.text(
            val + max(imp) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=8,
        )

plt.tight_layout()
out_path = OUT_DIR / "insulation_feature_importance.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")