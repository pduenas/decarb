from pathlib import Path
import pandas as pd
import numpy as np
import concurrent.futures


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KW_TO_BTU = 3412.142  # 1 kWh = 3412.142 BTU = 3.412142 kBTU

CONSUMPTION_COLS = [
    "out.load.heating.energy_delivered..kbtu",
    "out.load.cooling.energy_delivered..kbtu",
    "out.electricity.heating.energy_consumption..kwh",
    "out.electricity.heating_hp_bkup.energy_consumption..kwh",
    "out.electricity.heating_fans_pumps.energy_consumption..kwh",
    "out.electricity.cooling.energy_consumption..kwh",
    "out.electricity.cooling_fans_pumps.energy_consumption..kwh",
    "out.indoor_operative_temperature.conditioned_space..c",
    "out.outdoor_air_drybulb_temp..c",
]

METADATA_COLS = [
    "bldg_id",
    "in.hvac_heating_type_and_fuel",
    "in.hvac_heating_type",
    "in.hvac_cooling_type",
    "in.hvac_heating_efficiency",
    "in.hvac_cooling_efficiency",
]

# Configuration
BIN_SIZE_C = 1.0
MIN_ELEC_KWH = 0.01       # minimum per-interval electricity to compute COP
COP_RANGE_HEATING = (0.3, 15.0)
COP_RANGE_COOLING = (0.5, 15.0)
N_WORKERS = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_building_consumption(consumption_dir: Path, building_id: int) -> pd.DataFrame:
    """Load consumption parquet for a single building with only needed columns."""
    filepath = consumption_dir / f"{building_id}-0.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"Consumption file not found: {filepath}")

    # Read only the columns we need; handle missing columns gracefully
    import pyarrow.parquet as pq
    schema_names = pq.read_schema(filepath).names
    cols_to_load = [c for c in CONSUMPTION_COLS if c in schema_names]
    df = pd.read_parquet(filepath, engine="pyarrow", columns=cols_to_load)

    # Fill missing columns with 0
    for c in CONSUMPTION_COLS:
        if c not in df.columns:
            df[c] = 0.0

    return df


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load building metadata and return indexed by bldg_id."""
    df = pd.read_parquet(metadata_path, engine="pyarrow", columns=METADATA_COLS)
    if "bldg_id" in df.columns:
        df = df.set_index("bldg_id")

    # Build granular category labels (strip commas for clean names)
    # Heating: combine type_and_fuel with efficiency, remove the duplicated
    # equipment name that appears in both columns.
    # e.g. "Electricity ASHP" + "ASHP, SEER 10, 6.2 HSPF" → "Electricity ASHP SEER 10 6.2 HSPF"
    type_fuel = df["in.hvac_heating_type_and_fuel"].fillna("")
    eff_raw = df["in.hvac_heating_efficiency"].fillna("")

    def _build_heating_label(tf: str, eff: str) -> str:
        eff_clean = eff.replace(",", "").strip()
        if not tf or not eff_clean:
            return (tf + " " + eff_clean).strip()
        # Extract just the fuel type (e.g. "Electricity", "Natural Gas",
        # "Propane", "Fuel Oil", "Other Fuel") and prepend to efficiency.
        fuel_prefixes = [
            "Natural Gas", "Fuel Oil", "Other Fuel",
            "Electricity", "Propane", "None",
        ]
        fuel = ""
        for fp in fuel_prefixes:
            if tf.startswith(fp):
                fuel = fp
                break
        return (fuel + " " + eff_clean).strip()

    df["heating_category"] = [
        _build_heating_label(t, e) for t, e in zip(type_fuel, eff_raw)
    ]

    # Cooling: "AC, SEER 10" → "AC SEER 10"
    df["cooling_category"] = (
        df["in.hvac_cooling_efficiency"].fillna("").str.replace(",", "").str.strip()
    )

    # For heat pump buildings, use the full ASHP/MSHP category (with SEER & HSPF)
    # instead of the generic "Ducted Heat Pump" / "Non-Ducted Heat Pump" label.
    hp_mask = df["in.hvac_cooling_type"].isin(["Ducted Heat Pump", "Non-Ducted Heat Pump"])
    df.loc[hp_mask, "cooling_category"] = df.loc[hp_mask, "heating_category"]

    return df


# ---------------------------------------------------------------------------
# COP and delta-T computation
# ---------------------------------------------------------------------------

def compute_cop_delta_t(df: pd.DataFrame, is_electric_heating: bool) -> dict:
    """
    Compute COP and delta-T arrays for heating and cooling from a building's
    consumption data.

    Delta-T convention:
      - Heating: delta_T = T_indoor - T_outdoor  (positive; indoor warmer)
      - Cooling: delta_T = T_outdoor - T_indoor  (positive; outdoor warmer)

    Returns dict with keys: heating_cop, heating_delta_t, cooling_cop, cooling_delta_t
    (each a numpy array, possibly empty).
    """
    t_indoor = df["out.indoor_operative_temperature.conditioned_space..c"].values
    t_outdoor = df["out.outdoor_air_drybulb_temp..c"].values

    result = {}

    # --- Heating ---
    if is_electric_heating:
        heat_load = df["out.load.heating.energy_delivered..kbtu"].values
        heat_elec = (
            df["out.electricity.heating.energy_consumption..kwh"].values
            + df["out.electricity.heating_hp_bkup.energy_consumption..kwh"].values
            + df["out.electricity.heating_fans_pumps.energy_consumption..kwh"].values
        )

        # delta_T for heating = T_indoor - T_outdoor (positive)
        delta_t_heat = t_indoor - t_outdoor

        # Valid: nonzero load, sufficient electricity, positive delta_T
        mask = (heat_load > 0) & (heat_elec >= MIN_ELEC_KWH) & (delta_t_heat > 0)
        safe_elec = np.where(heat_elec > 0, heat_elec, 1.0)
        cop_heat = np.where(mask, heat_load / (safe_elec * KW_TO_BTU / 1000), np.nan)

        # Apply COP range filter
        lo, hi = COP_RANGE_HEATING
        valid = mask & (cop_heat >= lo) & (cop_heat <= hi)

        result["heating_cop"] = cop_heat[valid]
        result["heating_delta_t"] = delta_t_heat[valid]
    else:
        result["heating_cop"] = np.array([])
        result["heating_delta_t"] = np.array([])

    # --- Cooling ---
    cool_load = df["out.load.cooling.energy_delivered..kbtu"].values
    cool_elec = (
        df["out.electricity.cooling.energy_consumption..kwh"].values
        + df["out.electricity.cooling_fans_pumps.energy_consumption..kwh"].values
    )

    # delta_T for cooling = T_outdoor - T_indoor (positive)
    delta_t_cool = t_outdoor - t_indoor

    mask = (cool_load > 0) & (cool_elec >= MIN_ELEC_KWH) & (delta_t_cool > 0)
    safe_elec = np.where(cool_elec > 0, cool_elec, 1.0)
    cop_cool = np.where(mask, cool_load / (safe_elec * KW_TO_BTU / 1000), np.nan)

    lo, hi = COP_RANGE_COOLING
    valid = mask & (cop_cool >= lo) & (cop_cool <= hi)

    result["cooling_cop"] = cop_cool[valid]
    result["cooling_delta_t"] = delta_t_cool[valid]

    return result


# ---------------------------------------------------------------------------
# Binning and slope fitting
# ---------------------------------------------------------------------------

def bin_cop_by_delta_t(
    cops: np.ndarray,
    delta_ts: np.ndarray,
    bin_size: float = BIN_SIZE_C,
) -> pd.DataFrame:
    """
    Bin COP values by delta-T and compute statistics per bin.

    Returns DataFrame with columns:
        delta_t_bin_center, cop_mean, cop_median, cop_std,
        cop_25pct, cop_75pct, cop_count
    """
    if len(cops) == 0:
        return pd.DataFrame(columns=[
            "delta_t_bin_center", "cop_mean", "cop_median", "cop_std",
            "cop_25pct", "cop_75pct", "cop_count",
        ])

    # Create bin edges
    bin_min = np.floor(delta_ts.min() / bin_size) * bin_size
    bin_max = np.ceil(delta_ts.max() / bin_size) * bin_size + bin_size
    edges = np.arange(bin_min, bin_max, bin_size)
    bin_indices = np.digitize(delta_ts, edges) - 1

    records = []
    for i in range(len(edges) - 1):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        bin_cops = cops[mask]
        records.append({
            "delta_t_bin_center": edges[i] + bin_size / 2,
            "cop_mean": np.mean(bin_cops),
            "cop_median": np.median(bin_cops),
            "cop_std": np.std(bin_cops),
            "cop_25pct": np.percentile(bin_cops, 25),
            "cop_75pct": np.percentile(bin_cops, 75),
            "cop_count": int(mask.sum()),
        })

    return pd.DataFrame(records)


def fit_cop_slope(cops: np.ndarray, delta_ts: np.ndarray) -> tuple[float, float, float]:
    """
    Fit COP = intercept - slope * delta_T via least squares.

    The slope is returned as a positive magnitude representing COP loss
    per degree C of delta-T (i.e. the raw negative slope is negated).

    Returns (slope_positive, intercept, r_squared).
    """
    if len(cops) < 3:
        return np.nan, np.nan, np.nan

    A = np.column_stack([delta_ts, np.ones(len(delta_ts))])
    result = np.linalg.lstsq(A, cops, rcond=None)
    raw_slope, intercept = result[0]

    # R²
    predicted = raw_slope * delta_ts + intercept
    ss_res = np.sum((cops - predicted) ** 2)
    ss_tot = np.sum((cops - np.mean(cops)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Return slope as positive loss magnitude
    return float(-raw_slope), float(intercept), float(r_squared)


# ---------------------------------------------------------------------------
# Per-building processing
# ---------------------------------------------------------------------------

def process_building(
    building_id: int,
    consumption_dir: Path,
    is_electric_heating: bool,
) -> dict | None:
    """
    Process a single building: compute COP vs delta-T for heating and cooling.

    Returns a dict with per-building summary or None on failure.
    """
    try:
        df = load_building_consumption(consumption_dir, building_id)
        data = compute_cop_delta_t(df, is_electric_heating)

        # Heating binned results and slope
        h_bins = bin_cop_by_delta_t(data["heating_cop"], data["heating_delta_t"])
        h_slope, h_intercept, h_r2 = fit_cop_slope(data["heating_cop"], data["heating_delta_t"])

        # Cooling binned results and slope
        c_bins = bin_cop_by_delta_t(data["cooling_cop"], data["cooling_delta_t"])
        c_slope, c_intercept, c_r2 = fit_cop_slope(data["cooling_cop"], data["cooling_delta_t"])

        return {
            "building_id": building_id,
            "heating_bins": h_bins,
            "cooling_bins": c_bins,
            "heating_slope": h_slope,
            "heating_intercept": h_intercept,
            "heating_r2": h_r2,
            "heating_n_obs": len(data["heating_cop"]),
            "cooling_slope": c_slope,
            "cooling_intercept": c_intercept,
            "cooling_r2": c_r2,
            "cooling_n_obs": len(data["cooling_cop"]),
        }

    except Exception as e:
        print(f"Building {building_id}: ERROR — {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    all_results: list[dict],
    bin_size: float = BIN_SIZE_C,
) -> dict:
    """
    Aggregate per-building binned COP results into overall averages.

    Returns dict with:
        heating_aggregate, cooling_aggregate (DataFrames),
        heating_slope_agg, cooling_slope_agg (tuples of slope, intercept, r²)
    """
    heating_records = []
    cooling_records = []

    for r in all_results:
        bid = r["building_id"]
        for _, row in r["heating_bins"].iterrows():
            heating_records.append({
                "building_id": bid,
                "delta_t_bin_center": row["delta_t_bin_center"],
                "cop_mean": row["cop_mean"],
                "cop_count": row["cop_count"],
            })
        for _, row in r["cooling_bins"].iterrows():
            cooling_records.append({
                "building_id": bid,
                "delta_t_bin_center": row["delta_t_bin_center"],
                "cop_mean": row["cop_mean"],
                "cop_count": row["cop_count"],
            })

    output = {}

    for mode, records in [("heating", heating_records), ("cooling", cooling_records)]:
        if not records:
            output[f"{mode}_aggregate"] = pd.DataFrame()
            output[f"{mode}_slope_agg"] = (np.nan, np.nan, np.nan)
            output[f"{mode}_per_building"] = pd.DataFrame()
            continue

        df_all = pd.DataFrame(records)
        output[f"{mode}_per_building"] = df_all

        # Weighted average per bin (weighted by observation count)
        agg = df_all.groupby("delta_t_bin_center").apply(
            lambda g: pd.Series({
                "cop_mean": np.average(g["cop_mean"], weights=g["cop_count"]),
                "n_observations": g["cop_count"].sum(),
                "n_buildings": g["building_id"].nunique(),
            })
        ).reset_index()

        output[f"{mode}_aggregate"] = agg

        # Fit aggregate slope on the weighted-average bins
        if len(agg) >= 3:
            output[f"{mode}_slope_agg"] = fit_cop_slope(
                agg["cop_mean"].values,
                agg["delta_t_bin_center"].values,
            )
        else:
            output[f"{mode}_slope_agg"] = (np.nan, np.nan, np.nan)

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    BASE_DIR = Path(__file__).parents[1]
    consumption_dir = BASE_DIR / "out" / "consumption_files"
    metadata_path = BASE_DIR / "in" / "TX_upgrade0.parquet"
    ercot_map_path = BASE_DIR / "out" / "ercot_substation_nrel_map.parquet"
    output_dir = BASE_DIR / "out" / "cop_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    debug = False       # If True, only process first 10 buildings
    n_workers = N_WORKERS

    # --- Load ERCOT building list ---
    ercot_map_df = pd.read_parquet(ercot_map_path, engine="pyarrow", columns=["bldg_id"])
    ercot_bldg_ids = set(ercot_map_df["bldg_id"].astype(int).tolist())
    print(f"ERCOT map contains {len(ercot_bldg_ids)} buildings.")

    # --- Load metadata to determine which buildings have electric heating ---
    metadata_df = load_metadata(metadata_path)
    electric_heating_ids = set()
    for bid in ercot_bldg_ids:
        if bid in metadata_df.index:
            fuel = str(metadata_df.loc[bid, "in.hvac_heating_type_and_fuel"])
            if "Electricity" in fuel:
                electric_heating_ids.add(bid)
    print(f"{len(electric_heating_ids)} buildings have electric heating.")

    # --- Find available consumption files ---
    consumption_files = sorted(consumption_dir.glob("*-0.parquet"))
    available_ids = [
        int(f.stem.split("-")[0]) for f in consumption_files
        if int(f.stem.split("-")[0]) in ercot_bldg_ids
    ]
    if debug:
        available_ids = available_ids[:10]
    print(f"Processing {len(available_ids)} buildings with {n_workers} workers.")

    # --- Parallel processing ---
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_building,
                bid,
                consumption_dir,
                bid in electric_heating_ids,
            ): bid
            for bid in available_ids
        }
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                all_results.append(result)
            done_count += 1
            if done_count % 500 == 0:
                print(f"  {done_count}/{len(available_ids)} buildings processed...")

    print(f"Successfully processed {len(all_results)} of {len(available_ids)} buildings.")

    # --- Slope summary per building ---
    slope_rows = []
    for r in all_results:
        slope_rows.append({
            "building_id": r["building_id"],
            "heating_slope": r["heating_slope"],
            "heating_intercept": r["heating_intercept"],
            "heating_r2": r["heating_r2"],
            "heating_n_obs": r["heating_n_obs"],
            "cooling_slope": r["cooling_slope"],
            "cooling_intercept": r["cooling_intercept"],
            "cooling_r2": r["cooling_r2"],
            "cooling_n_obs": r["cooling_n_obs"],
        })

    slope_df = pd.DataFrame(slope_rows).sort_values("building_id").reset_index(drop=True)
    slope_df.to_csv(output_dir / "cop_slope_summary.csv", index=False)
    print(f"Saved per-building slope summary ({len(slope_df)} buildings).")

    # --- Aggregate ---
    agg = aggregate_results(all_results)

    for mode in ["heating", "cooling"]:
        agg_df = agg[f"{mode}_aggregate"]
        if not agg_df.empty:
            agg_df.to_csv(output_dir / f"{mode}_cop_by_delta_t_aggregate.csv", index=False)
            slope, intercept, r2 = agg[f"{mode}_slope_agg"]
            print(f"{mode.capitalize()} aggregate: COP = {intercept:.3f} - {slope:.4f} * delta_T  (R² = {r2:.3f})")

        per_bldg_df = agg[f"{mode}_per_building"]
        if not per_bldg_df.empty:
            per_bldg_df.to_parquet(
                output_dir / f"{mode}_cop_by_delta_t_per_building.parquet",
                engine="pyarrow", index=False,
            )

    print(f"All outputs saved to {output_dir}")

    # --- Plot COP spread by HVAC type ---
    plot_cop_spread(output_dir, metadata_df)


# ---------------------------------------------------------------------------
# Plotting: COP spread by HVAC type
# ---------------------------------------------------------------------------

def plot_cop_spread(output_dir: Path, metadata_df: pd.DataFrame = None):
    """
    Generate plots showing COP and COP-loss spread grouped by HVAC type.

    Reads saved output files from output_dir. If metadata_df is not provided,
    loads it from the default path.

    Produces 6 plots (3 chart types x heating/cooling):
      1. COP vs delta_T lines per HVAC type (with 25th-75th bands)
      2. Box plots of slope and intercept by HVAC type
      3. Scatter: intercept vs slope, colored by HVAC type
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    # --- Load metadata if not passed ---
    if metadata_df is None:
        base_dir = output_dir.parents[1]
        metadata_path = base_dir / "in" / "TX_upgrade0.parquet"
        metadata_df = load_metadata(metadata_path)

    # --- Load saved results ---
    slope_df = pd.read_csv(output_dir / "cop_slope_summary.csv")

    # Merge HVAC categories onto slope summary
    hvac_map = metadata_df[["heating_category", "cooling_category"]].copy()
    hvac_map.index.name = "building_id"
    slope_df = slope_df.merge(hvac_map, left_on="building_id", right_index=True, how="left")

    # --- Fit COP slope per HVAC category and save ---
    category_fit_rows = []
    for mode, type_col, slope_col, intercept_col, n_obs_col, per_bldg_file in [
        ("heating", "heating_category", "heating_slope", "heating_intercept", "heating_n_obs",
         "heating_cop_by_delta_t_per_building.parquet"),
        ("cooling", "cooling_category", "cooling_slope", "cooling_intercept", "cooling_n_obs",
         "cooling_cop_by_delta_t_per_building.parquet"),
    ]:
        df_mode = slope_df[slope_df[n_obs_col] > 0].copy()
        if df_mode.empty:
            continue

        per_bldg_path = output_dir / per_bldg_file
        if not per_bldg_path.exists():
            continue
        per_bldg = pd.read_parquet(per_bldg_path, engine="pyarrow")
        per_bldg = per_bldg.merge(hvac_map, left_on="building_id", right_index=True, how="left")

        for htype in sorted(df_mode[type_col].dropna().unique()):
            sub = per_bldg[per_bldg[type_col] == htype]
            if sub.empty:
                continue
            # Weighted-average COP per bin, then fit slope on the binned curve
            binned = sub.groupby("delta_t_bin_center").apply(
                lambda g: pd.Series({
                    "cop_mean": np.average(g["cop_mean"], weights=g["cop_count"]),
                    "n_observations": g["cop_count"].sum(),
                    "n_buildings": g["building_id"].nunique(),
                }),
                include_groups=False,
            ).reset_index()
            binned = binned[binned["n_buildings"] >= 3]
            if len(binned) < 3:
                continue
            slope, intercept, r2 = fit_cop_slope(
                binned["cop_mean"].values, binned["delta_t_bin_center"].values,
            )
            n_bldg = df_mode[df_mode[type_col] == htype]["building_id"].nunique()
            category_fit_rows.append({
                "mode": mode,
                "hvac_type": htype,
                "cop_intercept": intercept,
                "cop_loss_per_c": slope,
                "r2": r2,
                "n_buildings": n_bldg,
                "n_observations": int(binned["n_observations"].sum()),
            })

    if category_fit_rows:
        cat_df = pd.DataFrame(category_fit_rows)
        cat_df.to_csv(output_dir / "cop_fit_by_hvac_category.csv", index=False)
        print(f"Saved COP fits by HVAC category ({len(cat_df)} rows).")

    # --- Per-mode plotting ---
    mode_configs = [
        {
            "mode": "heating",
            "type_col": "heating_category",
            "slope_col": "heating_slope",
            "intercept_col": "heating_intercept",
            "r2_col": "heating_r2",
            "n_obs_col": "heating_n_obs",
            "per_bldg_file": "heating_cop_by_delta_t_per_building.parquet",
        },
        {
            "mode": "cooling",
            "type_col": "cooling_category",
            "slope_col": "cooling_slope",
            "intercept_col": "cooling_intercept",
            "r2_col": "cooling_r2",
            "n_obs_col": "cooling_n_obs",
            "per_bldg_file": "cooling_cop_by_delta_t_per_building.parquet",
        },
    ]

    for cfg in mode_configs:
        mode = cfg["mode"]
        type_col = cfg["type_col"]

        # Filter to buildings with data for this mode
        df_mode = slope_df[slope_df[cfg["n_obs_col"]] > 0].copy()
        if df_mode.empty:
            print(f"No {mode} data to plot, skipping.")
            continue

        hvac_types = sorted(df_mode[type_col].dropna().unique())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(hvac_types), 1)))

        # ---- Plot 1: COP vs delta_T lines per HVAC type ----
        per_bldg_path = output_dir / cfg["per_bldg_file"]
        if per_bldg_path.exists():
            per_bldg = pd.read_parquet(per_bldg_path, engine="pyarrow")
            per_bldg = per_bldg.merge(hvac_map, left_on="building_id", right_index=True, how="left")

            fig, ax = plt.subplots(figsize=(12, 7))
            for i, htype in enumerate(hvac_types):
                sub = per_bldg[per_bldg[type_col] == htype]
                if sub.empty:
                    continue
                binned = sub.groupby("delta_t_bin_center").agg(
                    cop_mean=("cop_mean", lambda x: np.average(x, weights=sub.loc[x.index, "cop_count"])),
                    cop_25=("cop_mean", lambda x: np.percentile(x, 25)),
                    cop_75=("cop_mean", lambda x: np.percentile(x, 75)),
                    n_bldg=("building_id", "nunique"),
                ).reset_index()
                # Only plot bins with >= 3 buildings
                binned = binned[binned["n_bldg"] >= 3]
                if binned.empty:
                    continue
                ax.plot(binned["delta_t_bin_center"], binned["cop_mean"],
                        label=f"{htype} (n={sub['building_id'].nunique()})",
                        color=colors[i], linewidth=2)
                ax.fill_between(binned["delta_t_bin_center"],
                                binned["cop_25"], binned["cop_75"],
                                alpha=0.15, color=colors[i])

            ax.set_xlabel("Delta T (°C)")
            ax.set_ylabel("COP")
            ax.set_title(f"{mode.capitalize()} COP vs Delta-T by HVAC Type")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / f"{mode}_cop_vs_delta_t_by_type.png", dpi=150)
            plt.close(fig)
            print(f"Saved {mode}_cop_vs_delta_t_by_type.png")

        # ---- Plot 2: Box plots of slope and intercept by HVAC type ----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        box_data_slope = [df_mode.loc[df_mode[type_col] == ht, cfg["slope_col"]].dropna()
                          for ht in hvac_types]
        box_data_intercept = [df_mode.loc[df_mode[type_col] == ht, cfg["intercept_col"]].dropna()
                              for ht in hvac_types]

        # Only include types with data
        valid = [(ht, s, ic) for ht, s, ic in zip(hvac_types, box_data_slope, box_data_intercept)
                 if len(s) >= 2]
        if valid:
            v_types, v_slopes, v_intercepts = zip(*valid)
            labels = [f"{t}\n(n={len(s)})" for t, s in zip(v_types, v_slopes)]

            bp1 = ax1.boxplot(v_slopes, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch, c in zip(bp1["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
            ax1.set_ylabel("COP Loss (COP/°C)")
            ax1.set_title(f"{mode.capitalize()} COP Loss Slope by HVAC Type")
            ax1.tick_params(axis="x", rotation=30, labelsize=8)
            ax1.grid(True, alpha=0.3, axis="y")

            bp2 = ax2.boxplot(v_intercepts, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch, c in zip(bp2["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
            ax2.set_ylabel("COP Intercept")
            ax2.set_title(f"{mode.capitalize()} COP Intercept by HVAC Type")
            ax2.tick_params(axis="x", rotation=30, labelsize=8)
            ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(output_dir / f"{mode}_cop_boxplots_by_type.png", dpi=150)
        plt.close(fig)
        print(f"Saved {mode}_cop_boxplots_by_type.png")

        # ---- Plot 3: Scatter of intercept vs slope, colored by HVAC type ----
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, htype in enumerate(hvac_types):
            sub = df_mode[df_mode[type_col] == htype]
            sub = sub.dropna(subset=[cfg["slope_col"], cfg["intercept_col"]])
            if sub.empty:
                continue
            ax.scatter(sub[cfg["slope_col"]], sub[cfg["intercept_col"]],
                       label=f"{htype} (n={len(sub)})",
                       color=colors[i], alpha=0.4, s=15, edgecolors="none")

        ax.set_xlabel("COP Loss Slope (COP/°C)")
        ax.set_ylabel("COP Intercept")
        ax.set_title(f"{mode.capitalize()} COP: Intercept vs Loss Slope by HVAC Type")
        ax.legend(fontsize=8, loc="best", markerscale=3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{mode}_cop_scatter_by_type.png", dpi=150)
        plt.close(fig)
        print(f"Saved {mode}_cop_scatter_by_type.png")


if __name__ == "__main__":
    main()
