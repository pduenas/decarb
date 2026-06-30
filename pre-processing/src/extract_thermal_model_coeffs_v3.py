"""
aggregate_regression_coeffs.py

Reads a single regression coefficient CSV that contains all buildings,
and produces:
  - regression_coeff.json          — bldg_id -> {k1, k2, k3, f_pval, r_squared}
  - regression_coeff_stats.csv     — summary statistics per metric
  - histogram_coefficients.png     — k1, k2, k3 distributions side by side
  - histogram_fit.png              — F-test p-value and R² distributions side by side
  - sample_high_r2.png             — 3 sample buildings with R² > 0.9
  - sample_low_r2.png              — 3 sample buildings with 0 < R² < 0.5

Empty or unparseable float values are stored as null in the JSON and
excluded from statistics and plots.

NOTE: k2 and k3 are scaled by 1000 after reading from the CSV.
"""

import csv
import json
import random
import statistics
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_COLS = {"bldg_id", "k1", "k2", "k3", "f_pval", "r_squared"}


# ── I/O ───────────────────────────────────────────────────────────────────────

def parse_float(val: str) -> float | None:
    """Return float, or None if the value is blank or unparseable."""
    s = val.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_csv(filepath: Path) -> list[dict]:
    """
    Read the CSV and return a list of dicts with REQUIRED_COLS.
    Float fields are cast to float; empty or unparseable values become None.
    k2 and k3 are scaled by 1000 after parsing.
    """
    text = filepath.read_text(encoding="utf-8")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","

    rows = []
    with filepath.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        headers = set(reader.fieldnames or [])

        missing = REQUIRED_COLS - headers
        if missing:
            raise SystemExit(f"ERROR: '{filepath}' is missing required columns: {missing}")

        for lineno, row in enumerate(reader, start=2):
            try:
                bldg_id = str(row["bldg_id"]).strip()
            except KeyError:
                print(f"  [WARNING] Skipping line {lineno}: missing bldg_id.")
                continue

            k2_raw = parse_float(row["k2"])
            k3_raw = parse_float(row["k3"])

            rows.append({
                "bldg_id":   bldg_id,
                "k1":        parse_float(row["k1"]),
                "k2":        k2_raw * 1000 if k2_raw is not None else None,
                "k3":        k3_raw * 1000 if k3_raw is not None else None,
                "f_pval":    parse_float(row["f_pval"]),
                "r_squared": parse_float(row["r_squared"]),
            })

    return rows


def build_mapping(csv_path: Path) -> dict:
    """
    Read CSV and return bldg_id -> {k1, k2, k3, f_pval, r_squared}.
    k2 and k3 are already scaled by 1000 from read_csv.
    Duplicate bldg_ids are resolved by keeping the entry with the lower f_pval
    (None f_pval is treated as worse than any real value).
    """
    rows = read_csv(csv_path)
    mapping: dict[str, dict] = {}

    for row in rows:
        bid   = row["bldg_id"]
        entry = {k: row[k] for k in ("k1", "k2", "k3", "f_pval", "r_squared")}

        if bid not in mapping:
            mapping[bid] = entry
            continue

        existing_fpval = mapping[bid]["f_pval"]
        new_fpval      = entry["f_pval"]
        if new_fpval is not None and (existing_fpval is None or new_fpval < existing_fpval):
            print(f"  [WARNING] Duplicate bldg_id={bid}. Replacing with lower-f_pval entry.")
            mapping[bid] = entry

    return mapping


def load_metadata(metadata_path: Path) -> list[str]:
    """
    Load the regression metadata JSON and return the ordered list of feature names.
    Falls back to the baseline default if the file is missing.
    """
    default_features = ["temp_diff_lag", "radiation", "heat_minus_cool_w"]
    if not metadata_path.exists():
        print(f"  [WARNING] Metadata not found at '{metadata_path}'. Using default features.")
        return default_features
    with metadata_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)
    return meta.get("features", default_features)


def load_building_parquet(bid: str, consumption_dir: Path, ts_config: dict) -> pd.DataFrame | None:
    """
    Load a building's raw parquet file, rename columns using ts_config to match
    the names expected by the thermal model (indoor_temp_c, outdoor_temp_c,
    solar_radiation_w, heating_load_w, cooling_load_w), and set a DatetimeIndex.

    Replicates the preprocessing done in create_building_timeseries:
      - Renames raw column keys from ts_config to standardised names
      - Converts kBtu loads to Watts (× 1172.284)
      - Fills missing load columns with 0
      - Rounds indoor temperature to nearest 0.5 °C (matches regression fitting)

    Returns None if the file is missing or lacks the indoor temperature column.
    """
    parquet_path = consumption_dir / f"{bid}-0.parquet"
    if not parquet_path.exists():
        print(f"  [WARNING] Parquet not found: '{parquet_path}' — skipping.")
        return None

    df = pd.read_parquet(parquet_path, engine="fastparquet")

    # Set datetime index from the timestamp column
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Check for the indoor temperature column (mandatory)
    indoor_key = ts_config["indoor_temp_key"]
    if indoor_key not in df.columns:
        print(f"  [WARNING] Building {bid} missing indoor temp column '{indoor_key}'.")
        print(f"             Available columns: {list(df.columns)}")
        return None

    # Rename indoor temp
    df = df.rename(columns={indoor_key: "indoor_temp_c"})

    # Round indoor temperature to nearest 0.5 °C — must match regression preprocessing
    df["indoor_temp_c"] = (df["indoor_temp_c"] * 2).round() / 2

    # Heating and cooling loads (kBtu -> W); fill missing with 0
    heating_key = ts_config.get("heating_load_key")
    cooling_key = ts_config.get("cooling_load_key")

    if heating_key and heating_key in df.columns:
        df["heating_load_w"] = df[heating_key] * 1172.284
    else:
        df["heating_load_w"] = 0.0

    if cooling_key and cooling_key in df.columns:
        df["cooling_load_w"] = df[cooling_key] * 1172.284
    else:
        df["cooling_load_w"] = 0.0

    df["heating_load_w"] = df["heating_load_w"].fillna(0)
    df["cooling_load_w"] = df["cooling_load_w"].fillna(0)

    return df


# ── Thermal model ─────────────────────────────────────────────────────────────

def reconstruct_temperature(
    df: pd.DataFrame,
    coefficients: list[float],
    feature_names: list[str],
    outdoor_temp: pd.Series | None,
    solar_radiation: pd.Series | None,
) -> pd.Series:
    """
    Reconstruct indoor temperature using the thermal model, matching exactly
    how compute_regression_coefficients builds X in the pipeline script.

    Model form (baseline):
      T_in(t) = T_in(t-1) + k1*(T_out(t-1) - T_in(t-1))
                           + k2*radiation(t)
                           + k3*(heat - cool)(t-1)

    Feature names and their order are read from the metadata JSON, so this
    works for any feature set defined in FEATURE_SETS.

    Args:
        df:               Building DataFrame with indoor_temp_c, heating_load_w,
                          cooling_load_w (already preprocessed by load_building_parquet)
        coefficients:     [k1, k2, ...] in the same order as feature_names
        feature_names:    Ordered list from metadata JSON
        outdoor_temp:     Outdoor temperature Series aligned to df's index (or None)
        solar_radiation:  Solar radiation Series aligned to df's index (or None)
    """
    lag = df.shift(1)

    X_dict = {}
    for feature_name in feature_names:
        if feature_name == "temp_diff_lag":
            if outdoor_temp is None:
                X_dict[feature_name] = pd.Series(np.nan, index=df.index)
            else:
                X_dict[feature_name] = outdoor_temp.shift(1) - lag["indoor_temp_c"]
        elif feature_name == "radiation":
            X_dict[feature_name] = solar_radiation if solar_radiation is not None else pd.Series(0.0, index=df.index)
        elif "heat_minus_cool" in feature_name:
            X_dict[feature_name] = lag["heating_load_w"] - lag["cooling_load_w"]
        elif "lag" in feature_name:
            base_col = feature_name.replace("_lag", "")
            X_dict[feature_name] = lag[base_col] if base_col in df.columns else pd.Series(np.nan, index=df.index)
        else:
            X_dict[feature_name] = df[feature_name] if feature_name in df.columns else pd.Series(np.nan, index=df.index)

    X = pd.DataFrame(X_dict, index=df.index)

    delta_T = sum(
        coefficients[i] * X[fname]
        for i, fname in enumerate(feature_names)
        if i < len(coefficients)
    )
    return lag["indoor_temp_c"] + delta_T


def load_weather_for_building(
    bid: str,
    consumption_dir: Path,
    ts_config: dict,
    input_dir: Path,
) -> tuple[pd.Series | None, pd.Series | None, float]:
    """
    Look up the county code for a building from the ResStock parquet, then load
    the corresponding weather CSV to extract outdoor_temp_c and solar_radiation_w.

    Returns (outdoor_temp, solar_radiation, sqm) — any may be None/0 if unavailable.
    This mirrors the weather lookup in create_building_timeseries.
    """
    outdoor_temp    = None
    solar_radiation = None
    sqm             = 1.0  # fallback so radiation is not zero-scaled

    # Load ResStock metadata for sqm and county
    resstock_file = input_dir / "TX_upgrade0.parquet"
    if not resstock_file.exists():
        print(f"  [WARNING] ResStock file not found: '{resstock_file}'")
        return outdoor_temp, solar_radiation, sqm

    try:
        resstock_df = pd.read_parquet(resstock_file, engine="fastparquet")
        resstock_df["bldg_id"] = pd.to_numeric(resstock_df["bldg_id"], errors="coerce")
        row = resstock_df[resstock_df["bldg_id"] == int(bid)]
        if row.empty:
            return outdoor_temp, solar_radiation, sqm

        sqft = row["in.sqft..ft2"].values[0]
        sqm  = float(sqft) * 0.092903 if not pd.isna(sqft) else 1.0
        county_code = row["in.county"].values[0]
    except Exception as e:
        print(f"  [WARNING] Could not load ResStock for building {bid}: {e}")
        return outdoor_temp, solar_radiation, sqm

    if pd.isna(county_code):
        return outdoor_temp, solar_radiation, sqm

    weather_path = input_dir / "TX_weather_data" / f"{county_code}_2018.csv"
    if not weather_path.exists():
        return outdoor_temp, solar_radiation, sqm

    try:
        df_weather = pd.read_csv(weather_path)

        outdoor_col = next(
            (c for c in df_weather.columns
             if ts_config["outdoor_temp_key"].lower().replace(" ", "")
             in c.lower().replace(" ", "")),
            None,
        )
        dnr_col = next(
            (c for c in df_weather.columns
             if ts_config["dnr_radiation_key"].lower().replace(" ", "")
             in c.lower().replace(" ", "")),
            None,
        )

        if outdoor_col is None or dnr_col is None:
            return outdoor_temp, solar_radiation, sqm

        df_weather["datetime"] = pd.to_datetime(df_weather["date_time"])
        df_weather = df_weather.set_index("datetime")
        outdoor_temp    = df_weather[outdoor_col].rename("outdoor_temp_c")
        solar_radiation = (df_weather[dnr_col] * sqm).rename("solar_radiation_w")

    except Exception as e:
        print(f"  [WARNING] Could not load weather for building {bid}: {e}")

    return outdoor_temp, solar_radiation, sqm


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of floats."""
    vals   = sorted(values)
    n      = len(vals)
    mean   = statistics.mean(vals)
    median = statistics.median(vals)
    std    = statistics.stdev(vals) if n > 1 else 0.0
    return {
        "n":                 n,
        "mean":              mean,
        "median":            median,
        "std_dev":           std,
        "min":               vals[0],
        "max":               vals[-1],
        "mean_minus_1std":   mean - std,
        "mean_plus_1std":    mean + std,
        "mean_minus_2std":   mean - 2 * std,
        "mean_plus_2std":    mean + 2 * std,
        "median_minus_1std": median - std,
        "median_plus_1std":  median + std,
        "median_minus_2std": median - 2 * std,
        "median_plus_2std":  median + 2 * std,
    }


def print_and_collect_stats(mapping: dict, label: str) -> list[dict]:
    """Print per-metric summary tables and return rows ready for CSV."""
    metrics = {
        "k1":        [e["k1"]        for e in mapping.values() if e["k1"]        is not None],
        "k2":        [e["k2"]        for e in mapping.values() if e["k2"]        is not None],
        "k3":        [e["k3"]        for e in mapping.values() if e["k3"]        is not None],
        "f_pval":    [e["f_pval"]    for e in mapping.values() if e["f_pval"]    is not None],
        "r_squared": [e["r_squared"] for e in mapping.values() if e["r_squared"] is not None],
    }

    total     = len(mapping)
    stat_rows = []
    col       = 14

    for metric, values in metrics.items():
        n_null = total - len(values)
        if n_null:
            print(f"  [INFO] {metric}: {n_null} building(s) have null values — excluded from stats.")
        if not values:
            print(f"  [INFO] {metric}: no valid values, skipping.")
            continue

        stats   = compute_stats(values)
        display = "R²" if metric == "r_squared" else "F p-val" if metric == "f_pval" else metric

        print(f"\n  {display} statistics — {label} ({stats['n']} buildings)")
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

        stat_rows.append({"metric": metric, **stats})

    return stat_rows


# ── Histogram plots ───────────────────────────────────────────────────────────

def _draw_histogram(ax: plt.Axes, values: np.ndarray, metric: str) -> None:
    """Draw one histogram panel onto ax."""
    if len(values) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    p1, p99 = np.percentile(values, [1, 99])
    if metric == "r_squared":
        xlim = (max(0.0, p1 - 0.05), min(1.0, p99 + 0.05))
    elif metric == "f_pval":
        # Fixed range 0–0.05 with an overflow bin for values >= 0.05
        cutoff   = 0.05
        n_bins   = 50
        bin_width = cutoff / n_bins
        in_range  = values[values < cutoff]
        overflow  = values[values >= cutoff]

        ax.hist(in_range, bins=np.linspace(0, cutoff, n_bins + 1),
                edgecolor="black", alpha=0.7)
        if len(overflow) > 0:
            ax.bar(cutoff + bin_width / 2, len(overflow), width=bin_width,
                   color="#1f77b4", edgecolor="black", alpha=0.7)
            ax.text(cutoff - bin_width / 2, len(overflow),
                    f"≥{cutoff} (n={len(overflow)})",
                    ha="right", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xlim(0.0, cutoff + bin_width * 1.8)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)

        mean_val   = np.mean(values)
        median_val = np.median(values)
        if mean_val <= cutoff:
            ax.axvline(mean_val,   color="red",  linestyle="--", linewidth=1.5,
                       label=f"Mean: {mean_val:.2e}")
        if median_val <= cutoff:
            ax.axvline(median_val, color="blue", linestyle="--", linewidth=1.5,
                       label=f"Median: {median_val:.2e}")
        ax.legend(fontsize=8)
        return
    else:
        span = p99 - p1 or abs(p99) * 0.1 or 1.0
        xlim = (p1 - 0.05 * span, p99 + 0.05 * span)

    plot_values = values[(values >= xlim[0]) & (values <= xlim[1])]
    n_excluded  = len(values) - len(plot_values)

    ax.hist(plot_values, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlim(xlim)
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.3)

    if metric in ("k1", "k2", "k3"):
        ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

    mean_val   = np.mean(values)
    median_val = np.median(values)
    if xlim[0] <= mean_val <= xlim[1]:
        ax.axvline(mean_val,   color="red",  linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_val:.2e}")
    if xlim[0] <= median_val <= xlim[1]:
        ax.axvline(median_val, color="blue", linestyle="--", linewidth=1.5,
                   label=f"Median: {median_val:.2e}")
    ax.legend(fontsize=8)

    if n_excluded:
        print(f"    {metric}: excluded {n_excluded} outlier(s) from plot.")


def plot_histograms(mapping: dict, output_dir: Path) -> None:
    """
    Produce two PNG files:
      histogram_coefficients.png — k1, k2, k3 side by side
      histogram_fit.png          — F-test p-value and R² side by side
    """
    def get_values(metric: str) -> np.ndarray:
        return np.array(
            [e[metric] for e in mapping.values() if e[metric] is not None],
            dtype=float,
        )

    # k1, k2, k3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Regression Coefficients Distribution", fontsize=14, fontweight="bold")
    titles = {
        "k1": r"Temperature Differential Sensitivity ($\alpha$)",
        "k2": r"Solar Radiation Sensitivity ($\beta$)",
        "k3": r"Internal Heat Gain Sensitivity ($\gamma$)",
    }
    xlabels = {
        "k1": r"$\alpha$ [$^\circ$C / $^\circ$C]",
        "k2": r"$\beta$ [$^\circ$C / kW]",
        "k3": r"$\gamma$ [$^\circ$C / kW]",
    }
    for ax, metric in zip(axes, ("k1", "k2", "k3")):
        ax.set_title(titles[metric], fontsize=12)
        ax.set_xlabel(xlabels[metric])
        _draw_histogram(ax, get_values(metric), metric)
    plt.tight_layout()
    path = output_dir / "histogram_coefficients.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → '{path}'")

    # F-test p-value and R²
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Model Fit Distribution", fontsize=14, fontweight="bold")
    for ax, (metric, title, xlabel) in zip(axes, [
        ("f_pval",    "F-test p-value", "p-value"),
        ("r_squared", r"$R^2$",         r"$R^2$"),
    ]):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel)
        _draw_histogram(ax, get_values(metric), metric)
    plt.tight_layout()
    path = output_dir / "histogram_fit.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → '{path}'")

    # Combined figure: coefficients (top row) + fit (bottom row)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.7,
                          top=0.93, bottom=0.07, left=0.06, right=0.97)

    # Row titles
    fig.text(0.5, 0.96, "Regression Coefficients Distribution",
             ha="center", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.45, "Model Fit Distribution",
             ha="center", fontsize=14, fontweight="bold")

    # Top row: k1, k2, k3 — each spans 2 of 6 columns
    for col, metric in enumerate(("k1", "k2", "k3")):
        ax = fig.add_subplot(gs[0, col * 2 : col * 2 + 2])
        ax.set_title(titles[metric], fontsize=10)
        ax.set_xlabel(xlabels[metric], fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.tick_params(labelsize=9)
        _draw_histogram(ax, get_values(metric), metric)
        ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
        ax.legend(fontsize=7)

    # Bottom row: f_pval and R² — each spans 3 of 6 columns (equal width, square)
    ax_fpval = fig.add_subplot(gs[1, 0:3])
    ax_fpval.set_title("F-test p-value", fontsize=10)
    ax_fpval.set_xlabel("p-value", fontsize=10)
    ax_fpval.set_ylabel("Frequency", fontsize=10)
    ax_fpval.tick_params(labelsize=9)
    _draw_histogram(ax_fpval, get_values("f_pval"), "f_pval")
    ax_fpval.legend(fontsize=7)

    ax_r2 = fig.add_subplot(gs[1, 3:6])
    ax_r2.set_title(r"$R^2$", fontsize=10)
    ax_r2.set_xlabel(r"$R^2$", fontsize=10)
    ax_r2.set_ylabel("Frequency", fontsize=10)
    ax_r2.tick_params(labelsize=9)
    _draw_histogram(ax_r2, get_values("r_squared"), "r_squared")
    ax_r2.legend(fontsize=7)

    path = output_dir / "histogram_combined.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → '{path}'")


# ── Sample building plots ─────────────────────────────────────────────────────

def pick_sample_buildings(
    mapping: dict,
    consumption_dir: Path,
    input_dir: Path,
    ts_config: dict,
    feature_names: list[str],
    r2_min: float,
    r2_max: float,
    rng: random.Random,
    n: int = 3,
) -> list[tuple]:
    """
    Randomly draw n buildings whose R² falls within (r2_min, r2_max), load
    their parquet + weather data, and reconstruct the temperature profile.

    Returns a list of tuples: (bldg_id, r2, f_pval, y_true, y_pred)
    """
    candidates = [
        bid for bid, e in mapping.items()
        if e["r_squared"] is not None and r2_min < e["r_squared"] < r2_max
        and all(e[k] is not None for k in ("k1", "k2", "k3"))
    ]
    print(f"  R² in ({r2_min}, {r2_max}) candidates: {len(candidates)}")

    pool = candidates.copy()
    rng.shuffle(pool)

    sample = []
    for bid in pool:
        if len(sample) == n:
            break

        df = load_building_parquet(bid, consumption_dir, ts_config)
        if df is None:
            continue

        outdoor_temp, solar_radiation, _ = load_weather_for_building(
            bid, consumption_dir, ts_config, input_dir
        )

        # Align weather series to the building's index
        if outdoor_temp is not None:
            outdoor_temp = outdoor_temp.reindex(df.index).interpolate("linear")
        if solar_radiation is not None:
            solar_radiation = solar_radiation.reindex(df.index).interpolate("linear")

        e            = mapping[bid]
        coefficients = [e["k1"], e["k2"], e["k3"]]
        y_pred       = reconstruct_temperature(
            df, coefficients, feature_names, outdoor_temp, solar_radiation
        )
        sample.append((bid, e["r_squared"], e["f_pval"], df["indoor_temp_c"], y_pred))

    if len(sample) < n:
        print(f"  [WARNING] Only found {len(sample)} loadable buildings (wanted {n}).")

    return sample


def make_sample_figure(
    sample: list[tuple],
    title: str,
    output_path: Path,
) -> None:
    """
    Plot a figure with one subplot per building. Each subplot shows:
      - actual indoor temperature (blue)
      - reconstructed temperature (orange)
      - shaded error band between them (light red)
      - signed error on a twin right-hand axis (dotted black)
    """
    n = len(sample)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for ax, (bid, r2, f_pval, y_true, y_pred) in zip(axes, sample):
        valid = y_pred.dropna().index

        # Actual and reconstructed lines
        ax.plot(y_true.index, y_true.values,
                color="steelblue", linewidth=0.9, label="Actual", zorder=2)
        ax.plot(y_pred.loc[valid].index, y_pred.loc[valid].values,
                color="darkorange", linewidth=0.9, alpha=0.85, label="Reconstructed", zorder=1)

        ax.set_ylabel("Indoor Temp (°C)")
        fpval_str = f"{f_pval:.2e}" if f_pval is not None else "N/A"
        ax.set_title(f"Building {bid}  |  R² = {r2:.3f}  |  F p-val = {fpval_str}", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → '{output_path}'")


def plot_sample_buildings(
    mapping: dict,
    consumption_dir: Path,
    input_dir: Path,
    output_dir: Path,
    ts_config: dict,
    metadata_path: Path,
    random_seed: int = 42,
) -> None:
    """
    Produce two PNGs of 3 sample buildings each:
      sample_high_r2.png  — buildings with R² > 0.9
      sample_low_r2.png   — buildings with 0 < R² < 0.5
    """
    feature_names = load_metadata(metadata_path)
    print(f"  Features from metadata: {feature_names}")

    rng = random.Random(random_seed)

    print("\n  Selecting high-R² sample (R² > 0.9)...")
    high_sample = pick_sample_buildings(
        mapping, consumption_dir, input_dir, ts_config, feature_names,
        r2_min=0.9, r2_max=1.0, rng=rng,
    )
    if high_sample:
        make_sample_figure(
            high_sample,
            title=r"Sample Buildings — $R^2 > 0.9$",
            output_path=output_dir / "sample_high_r2.png",
        )

    print("\n  Selecting low-R² sample (0 < R² < 0.5)...")
    low_sample = pick_sample_buildings(
        mapping, consumption_dir, input_dir, ts_config, feature_names,
        r2_min=0.0, r2_max=0.5, rng=rng,
    )
    if low_sample:
        make_sample_figure(
            low_sample,
            title=r"Sample Buildings — $0 < R^2 < 0.5$",
            output_path=output_dir / "sample_low_r2.png",
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Configuration ─────────────────────────────────────────────────────────
    # Path(__file__).parents[1] resolves to ercot_project/ (one level above src/).

    # Single CSV containing all buildings for this case
    LOCATION = (
        Path(__file__).parents[1]
        / "out" / "thermal_model" / "baseline"
        / "regression_coeff_baseline.csv"
    )

    # Folder containing per-building parquet files (<bldg_id>-0.parquet)
    CONSUMPTION_DIR = Path(__file__).parents[1] / "out" / "consumption_files"

    # Input directory (ResStock parquet + weather CSVs)
    INPUT_DIR = Path(__file__).parents[1] / "in"

    # Raw column names in the building parquet files — must match the pipeline
    TS_CONFIG = {
        "indoor_temp_key":      "out.indoor_temperature.conditioned_space..c",
        "cooling_load_key":     "out.load.cooling.energy_delivered..kbtu",
        "heating_load_key":     "out.load.heating.energy_delivered..kbtu",
        "outdoor_temp_key":     "Dry Bulb Temperature [°C]",
        "dnr_radiation_key":    "Direct Normal Radiation [W/m2]",
        "outdoor_humidity_key": "Relative Humidity [%]",
        "wind_speed_key":       "Wind Speed [m/s]",
    }
    # ─────────────────────────────────────────────────────────────────────────

    csv_path      = Path(LOCATION)
    output_dir    = csv_path.parent
    metadata_path = output_dir / f"regression_coeff_{output_dir.name}_metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:    '{csv_path}'")
    print(f"Metadata: '{metadata_path}'")
    print(f"Output:   '{output_dir}'")

    # Read & build mapping
    mapping = build_mapping(csv_path)
    print(f"\n  Loaded {len(mapping)} building(s).")

    # Write JSON
    json_path = output_dir / "regression_coeff.json"
    json_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"  Wrote JSON → '{json_path}'")

    # Statistics
    print("\n── Computing statistics ──")
    stat_rows = print_and_collect_stats(mapping, label=csv_path.stem)
    if stat_rows:
        stats_path = output_dir / "regression_coeff_stats.csv"
        fieldnames = ["metric", "n", "mean", "median", "std_dev",
                      "min", "max", "mean_minus_1std", "mean_plus_1std",
                      "mean_minus_2std", "mean_plus_2std",
                      "median_minus_1std", "median_plus_1std",
                      "median_minus_2std", "median_plus_2std"]
        with stats_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stat_rows)
        print(f"  Wrote stats → '{stats_path}'")

    # Histograms
    print("\n── Generating histograms ──")
    plot_histograms(mapping, output_dir)

    # Sample building plots
    print("\n── Generating sample building plots ──")
    plot_sample_buildings(
        mapping       = mapping,
        consumption_dir = CONSUMPTION_DIR,
        input_dir     = INPUT_DIR,
        output_dir    = output_dir,
        ts_config     = TS_CONFIG,
        metadata_path = metadata_path,
        random_seed   = 42,
    )

    print("\nDone.")