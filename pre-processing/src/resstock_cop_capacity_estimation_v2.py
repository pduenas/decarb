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
PLOT_ONLY = False          # Set to True to skip processing and just regenerate plots


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_building_consumption(consumption_dir: Path, building_id: int) -> pd.DataFrame:
    """Load consumption parquet for a single building with only needed columns."""
    filepath = consumption_dir / f"{building_id}-0.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"Consumption file not found: {filepath}")

    import pyarrow.parquet as pq
    schema_names = pq.read_schema(filepath).names
    cols_to_load = [c for c in CONSUMPTION_COLS if c in schema_names]
    df = pd.read_parquet(filepath, engine="pyarrow", columns=cols_to_load)

    for c in CONSUMPTION_COLS:
        if c not in df.columns:
            df[c] = 0.0

    return df


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load building metadata and return indexed by bldg_id."""
    df = pd.read_parquet(metadata_path, engine="pyarrow", columns=METADATA_COLS)
    if "bldg_id" in df.columns:
        df = df.set_index("bldg_id")

    type_fuel = df["in.hvac_heating_type_and_fuel"].fillna("")
    eff_raw = df["in.hvac_heating_efficiency"].fillna("")

    def _build_heating_label(tf: str, eff: str) -> str:
        eff_clean = eff.replace(",", "").strip()
        if not tf or not eff_clean:
            return (tf + " " + eff_clean).strip()
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

    df["cooling_category"] = (
        df["in.hvac_cooling_efficiency"].fillna("").str.replace(",", "").str.strip()
    )

    hp_mask = df["in.hvac_cooling_type"].isin(["Ducted Heat Pump", "Non-Ducted Heat Pump"])
    df.loc[hp_mask, "cooling_category"] = df.loc[hp_mask, "heating_category"]

    return df


# ---------------------------------------------------------------------------
# COP, capacity, and delta-T computation
# ---------------------------------------------------------------------------

def compute_building_metrics(df: pd.DataFrame, is_electric_heating: bool) -> dict:
    """
    Compute COP, capacity (kW), and delta-T arrays for heating and cooling.

    Delta-T convention:
      - Heating: delta_T = T_indoor - T_outdoor  (positive; indoor warmer)
      - Cooling: delta_T = T_outdoor - T_indoor  (positive; outdoor warmer)

    Capacity is derived from load delivered:
      capacity_kw = load_kbtu * 4 / 3.412142
      (15-min energy in kBtu → hourly rate in kW)

    Returns dict with keys:
        heating_cop, heating_delta_t, cooling_cop, cooling_delta_t,
        heating_capacity_kw, heating_cap_delta_t,
        cooling_capacity_kw, cooling_cap_delta_t
    """
    t_indoor = df["out.indoor_operative_temperature.conditioned_space..c"].values
    t_outdoor = df["out.outdoor_air_drybulb_temp..c"].values

    result = {}

    # --- Heating ---
    heat_load = df["out.load.heating.energy_delivered..kbtu"].values
    delta_t_heat = t_indoor - t_outdoor

    # Capacity: all timesteps with nonzero load and positive delta_T
    cap_mask_heat = (heat_load > 0) & (delta_t_heat > 0)
    heat_cap_kw = heat_load * 4 / (KW_TO_BTU / 1000)  # kBtu/15min → kW
    result["heating_capacity_kw"] = heat_cap_kw[cap_mask_heat]
    result["heating_cap_delta_t"] = delta_t_heat[cap_mask_heat]

    # COP: only for electric heating
    if is_electric_heating:
        heat_elec = (
            df["out.electricity.heating.energy_consumption..kwh"].values
            + df["out.electricity.heating_hp_bkup.energy_consumption..kwh"].values
            + df["out.electricity.heating_fans_pumps.energy_consumption..kwh"].values
        )
        mask = (heat_load > 0) & (heat_elec >= MIN_ELEC_KWH) & (delta_t_heat > 0)
        safe_elec = np.where(heat_elec > 0, heat_elec, 1.0)
        cop_heat = np.where(mask, heat_load / (safe_elec * KW_TO_BTU / 1000), np.nan)

        lo, hi = COP_RANGE_HEATING
        valid = mask & (cop_heat >= lo) & (cop_heat <= hi)
        result["heating_cop"] = cop_heat[valid]
        result["heating_delta_t"] = delta_t_heat[valid]
    else:
        result["heating_cop"] = np.array([])
        result["heating_delta_t"] = np.array([])

    # --- Cooling ---
    cool_load = df["out.load.cooling.energy_delivered..kbtu"].values
    delta_t_cool = t_outdoor - t_indoor

    # Capacity: all timesteps with nonzero load and positive delta_T
    cap_mask_cool = (cool_load > 0) & (delta_t_cool > 0)
    cool_cap_kw = cool_load * 4 / (KW_TO_BTU / 1000)  # kBtu/15min → kW
    result["cooling_capacity_kw"] = cool_cap_kw[cap_mask_cool]
    result["cooling_cap_delta_t"] = delta_t_cool[cap_mask_cool]

    # COP
    cool_elec = (
        df["out.electricity.cooling.energy_consumption..kwh"].values
        + df["out.electricity.cooling_fans_pumps.energy_consumption..kwh"].values
    )
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
    """Bin COP values by delta-T and compute statistics per bin."""
    if len(cops) == 0:
        return pd.DataFrame(columns=[
            "delta_t_bin_center", "cop_mean", "cop_median", "cop_std",
            "cop_25pct", "cop_75pct", "cop_count",
        ])

    bin_min = np.floor(delta_ts.min() / bin_size) * bin_size
    bin_max = np.ceil(delta_ts.max() / bin_size) * bin_size + bin_size
    edges = np.arange(bin_min, bin_max, bin_size)
    bin_indices = np.digitize(delta_ts, edges) - 1

    records = []
    for i in range(len(edges) - 1):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        vals = cops[mask]
        records.append({
            "delta_t_bin_center": edges[i] + bin_size / 2,
            "cop_mean": np.mean(vals),
            "cop_median": np.median(vals),
            "cop_std": np.std(vals),
            "cop_25pct": np.percentile(vals, 25),
            "cop_75pct": np.percentile(vals, 75),
            "cop_count": int(mask.sum()),
        })

    return pd.DataFrame(records)


def bin_capacity_by_delta_t(
    capacities_kw: np.ndarray,
    delta_ts: np.ndarray,
    bin_size: float = BIN_SIZE_C,
) -> pd.DataFrame:
    """
    Bin capacity values by delta-T and compute statistics per bin.

    The 95th percentile per bin approximates the max capacity at that delta-T.
    """
    if len(capacities_kw) == 0:
        return pd.DataFrame(columns=[
            "delta_t_bin_center", "cap_95pct_kw", "cap_max_kw",
            "cap_mean_kw", "cap_median_kw", "cap_count",
        ])

    bin_min = np.floor(delta_ts.min() / bin_size) * bin_size
    bin_max = np.ceil(delta_ts.max() / bin_size) * bin_size + bin_size
    edges = np.arange(bin_min, bin_max, bin_size)
    bin_indices = np.digitize(delta_ts, edges) - 1

    records = []
    for i in range(len(edges) - 1):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        vals = capacities_kw[mask]
        records.append({
            "delta_t_bin_center": edges[i] + bin_size / 2,
            "cap_95pct_kw": np.percentile(vals, 95),
            "cap_max_kw": np.max(vals),
            "cap_mean_kw": np.mean(vals),
            "cap_median_kw": np.median(vals),
            "cap_count": int(mask.sum()),
        })

    return pd.DataFrame(records)


def fit_cop_slope(values: np.ndarray, delta_ts: np.ndarray) -> tuple[float, float, float]:
    """
    Fit COP = intercept - slope * delta_T via least squares.

    Slope returned as positive magnitude (COP loss per °C).

    Returns (slope_positive, intercept, r_squared).
    """
    if len(values) < 3:
        return np.nan, np.nan, np.nan

    A = np.column_stack([delta_ts, np.ones(len(delta_ts))])
    result = np.linalg.lstsq(A, values, rcond=None)
    raw_slope, intercept = result[0]

    predicted = raw_slope * delta_ts + intercept
    ss_res = np.sum((values - predicted) ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(-raw_slope), float(intercept), float(r_squared)


def fit_capacity_from_bins(cap_bins: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Derive a unit's max capacity and its degradation from binned load data.

    The 95th-pct load per delta_T bin rises in the demand-limited region,
    then falls in the capacity-limited region. The peak bin approximates
    rated capacity. We fit the decline from the peak onward:

        capacity = rated_cap - loss * (delta_T - peak_delta_T)

    Returns (rated_cap_kw, loss_kw_per_c, peak_delta_t_c, r_squared).
    loss_kw_per_c is positive when capacity drops with increasing delta_T.
    """
    if len(cap_bins) < 3:
        return np.nan, np.nan, np.nan, np.nan

    caps = cap_bins["cap_95pct_kw"].values
    dts = cap_bins["delta_t_bin_center"].values

    peak_idx = int(np.argmax(caps))
    rated_cap = float(caps[peak_idx])
    peak_dt = float(dts[peak_idx])

    # Fit only the capacity-limited region (from peak onward)
    decline_caps = caps[peak_idx:]
    decline_dts = dts[peak_idx:]

    if len(decline_caps) < 3:
        return rated_cap, np.nan, peak_dt, np.nan

    A = np.column_stack([decline_dts, np.ones(len(decline_dts))])
    result = np.linalg.lstsq(A, decline_caps, rcond=None)
    raw_slope, intercept = result[0]

    predicted = raw_slope * decline_dts + intercept
    ss_res = np.sum((decline_caps - predicted) ** 2)
    ss_tot = np.sum((decline_caps - np.mean(decline_caps)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Negate: raw_slope is negative (capacity drops) → positive loss magnitude
    return rated_cap, float(-raw_slope), peak_dt, float(r2)


# ---------------------------------------------------------------------------
# Per-building processing
# ---------------------------------------------------------------------------

def process_building(
    building_id: int,
    consumption_dir: Path,
    is_electric_heating: bool,
) -> dict | None:
    """
    Process a single building: compute COP and capacity vs delta-T.

    Capacity is in absolute kW.

    Returns a dict with per-building summary or None on failure.
    """
    try:
        df = load_building_consumption(consumption_dir, building_id)
        data = compute_building_metrics(df, is_electric_heating)

        # COP binned results and slope
        h_cop_bins = bin_cop_by_delta_t(data["heating_cop"], data["heating_delta_t"])
        h_cop_slope, h_cop_intercept, h_cop_r2 = fit_cop_slope(data["heating_cop"], data["heating_delta_t"])

        c_cop_bins = bin_cop_by_delta_t(data["cooling_cop"], data["cooling_delta_t"])
        c_cop_slope, c_cop_intercept, c_cop_r2 = fit_cop_slope(data["cooling_cop"], data["cooling_delta_t"])

        # Capacity: bin raw kW, find peak, fit decline
        h_cap_bins = bin_capacity_by_delta_t(data["heating_capacity_kw"], data["heating_cap_delta_t"])
        h_rated, h_cap_loss, h_peak_dt, h_cap_r2 = fit_capacity_from_bins(h_cap_bins)

        c_cap_bins = bin_capacity_by_delta_t(data["cooling_capacity_kw"], data["cooling_cap_delta_t"])
        c_rated, c_cap_loss, c_peak_dt, c_cap_r2 = fit_capacity_from_bins(c_cap_bins)

        return {
            "building_id": building_id,
            # COP
            "heating_cop_bins": h_cop_bins,
            "cooling_cop_bins": c_cop_bins,
            "heating_cop_slope": h_cop_slope,
            "heating_cop_intercept": h_cop_intercept,
            "heating_cop_r2": h_cop_r2,
            "heating_cop_n_obs": len(data["heating_cop"]),
            "cooling_cop_slope": c_cop_slope,
            "cooling_cop_intercept": c_cop_intercept,
            "cooling_cop_r2": c_cop_r2,
            "cooling_cop_n_obs": len(data["cooling_cop"]),
            # Capacity (kW)
            "heating_cap_bins": h_cap_bins,
            "cooling_cap_bins": c_cap_bins,
            "heating_rated_cap_kw": h_rated,
            "heating_cap_loss_kw_per_c": h_cap_loss,
            "heating_cap_peak_delta_t": h_peak_dt,
            "heating_cap_r2": h_cap_r2,
            "heating_cap_n_obs": len(data["heating_capacity_kw"]),
            "cooling_rated_cap_kw": c_rated,
            "cooling_cap_loss_kw_per_c": c_cap_loss,
            "cooling_cap_peak_delta_t": c_peak_dt,
            "cooling_cap_r2": c_cap_r2,
            "cooling_cap_n_obs": len(data["cooling_capacity_kw"]),
        }

    except Exception as e:
        print(f"Building {building_id}: ERROR — {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_cop_results(all_results: list[dict]) -> dict:
    """Aggregate per-building binned COP results into overall averages."""
    heating_records = []
    cooling_records = []

    for r in all_results:
        bid = r["building_id"]
        for _, row in r["heating_cop_bins"].iterrows():
            heating_records.append({
                "building_id": bid,
                "delta_t_bin_center": row["delta_t_bin_center"],
                "cop_mean": row["cop_mean"],
                "cop_count": row["cop_count"],
            })
        for _, row in r["cooling_cop_bins"].iterrows():
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

        agg = df_all.groupby("delta_t_bin_center").apply(
            lambda g: pd.Series({
                "cop_mean": np.average(g["cop_mean"], weights=g["cop_count"]),
                "n_observations": g["cop_count"].sum(),
                "n_buildings": g["building_id"].nunique(),
            }),
            include_groups=False,
        ).reset_index()

        output[f"{mode}_aggregate"] = agg

        if len(agg) >= 3:
            output[f"{mode}_slope_agg"] = fit_cop_slope(
                agg["cop_mean"].values, agg["delta_t_bin_center"].values,
            )
        else:
            output[f"{mode}_slope_agg"] = (np.nan, np.nan, np.nan)

    return output


def aggregate_capacity_results(all_results: list[dict]) -> dict:
    """Aggregate per-building binned capacity results into overall averages."""
    heating_records = []
    cooling_records = []

    for r in all_results:
        bid = r["building_id"]
        for _, row in r["heating_cap_bins"].iterrows():
            heating_records.append({
                "building_id": bid,
                "delta_t_bin_center": row["delta_t_bin_center"],
                "cap_95pct_kw": row["cap_95pct_kw"],
                "cap_count": row["cap_count"],
            })
        for _, row in r["cooling_cap_bins"].iterrows():
            cooling_records.append({
                "building_id": bid,
                "delta_t_bin_center": row["delta_t_bin_center"],
                "cap_95pct_kw": row["cap_95pct_kw"],
                "cap_count": row["cap_count"],
            })

    output = {}
    for mode, records in [("heating", heating_records), ("cooling", cooling_records)]:
        if not records:
            output[f"{mode}_cap_aggregate"] = pd.DataFrame()
            output[f"{mode}_cap_slope_agg"] = (np.nan, np.nan, np.nan)
            output[f"{mode}_cap_per_building"] = pd.DataFrame()
            continue

        df_all = pd.DataFrame(records)
        output[f"{mode}_cap_per_building"] = df_all

        agg = df_all.groupby("delta_t_bin_center").apply(
            lambda g: pd.Series({
                "cap_95pct_kw": np.average(g["cap_95pct_kw"], weights=g["cap_count"]),
                "n_observations": g["cap_count"].sum(),
                "n_buildings": g["building_id"].nunique(),
            }),
            include_groups=False,
        ).reset_index()

        output[f"{mode}_cap_aggregate"] = agg

        if len(agg) >= 3:
            agg_for_fit = agg.rename(columns={"cap_95pct_kw": "cap_95pct_kw"})
            output[f"{mode}_cap_fit_agg"] = fit_capacity_from_bins(agg_for_fit)
        else:
            output[f"{mode}_cap_fit_agg"] = (np.nan, np.nan, np.nan, np.nan)

    return output


# ---------------------------------------------------------------------------
# Plot-only entry point
# ---------------------------------------------------------------------------

def plot_only(output_dir: Path, metadata_path: Path) -> None:
    """Re-generate all plots from existing summary files without reprocessing buildings."""
    output_dir = Path(output_dir)
    metadata_df = load_metadata(metadata_path)
    print("Plot-only mode: loading existing summary files...")
    plot_cop_spread(output_dir, metadata_df)
    plot_capacity_spread(output_dir, metadata_df)
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    BASE_DIR = Path(__file__).parents[1]
    consumption_dir = BASE_DIR / "out" / "consumption_files"
    metadata_path = BASE_DIR / "in" / "TX_upgrade0.parquet"
    ercot_map_path = BASE_DIR / "out" / "ercot_substation_nrel_map.parquet"
    output_dir = BASE_DIR / "out" / "cop_capacity_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    debug = False
    n_workers = N_WORKERS

    # Skip all processing and just regenerate plots from existing files
    if PLOT_ONLY:
        plot_only(output_dir, metadata_path)
        return

    # --- Load ERCOT building list ---
    ercot_map_df = pd.read_parquet(ercot_map_path, engine="pyarrow", columns=["bldg_id"])
    ercot_bldg_ids = set(ercot_map_df["bldg_id"].astype(int).tolist())
    print(f"ERCOT map contains {len(ercot_bldg_ids)} buildings.")

    # --- Load metadata ---
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
        available_ids = available_ids[:100]
    print(f"Processing {len(available_ids)} buildings with {n_workers} workers.")

    # --- Parallel processing ---
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_building, bid, consumption_dir,
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

    # --- Per-building summary (COP + capacity) ---
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "building_id": r["building_id"],
            "heating_cop_slope": r["heating_cop_slope"],
            "heating_cop_intercept": r["heating_cop_intercept"],
            "heating_cop_r2": r["heating_cop_r2"],
            "heating_cop_n_obs": r["heating_cop_n_obs"],
            "cooling_cop_slope": r["cooling_cop_slope"],
            "cooling_cop_intercept": r["cooling_cop_intercept"],
            "cooling_cop_r2": r["cooling_cop_r2"],
            "cooling_cop_n_obs": r["cooling_cop_n_obs"],
            "heating_rated_cap_kw": r["heating_rated_cap_kw"],
            "heating_cap_loss_kw_per_c": r["heating_cap_loss_kw_per_c"],
            "heating_cap_peak_delta_t": r["heating_cap_peak_delta_t"],
            "heating_cap_r2": r["heating_cap_r2"],
            "heating_cap_n_obs": r["heating_cap_n_obs"],
            "cooling_rated_cap_kw": r["cooling_rated_cap_kw"],
            "cooling_cap_loss_kw_per_c": r["cooling_cap_loss_kw_per_c"],
            "cooling_cap_peak_delta_t": r["cooling_cap_peak_delta_t"],
            "cooling_cap_r2": r["cooling_cap_r2"],
            "cooling_cap_n_obs": r["cooling_cap_n_obs"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("building_id").reset_index(drop=True)
    summary_df.to_csv(output_dir / "cop_capacity_summary.csv", index=False)
    print(f"Saved per-building summary ({len(summary_df)} buildings).")

    # --- Aggregate COP ---
    cop_agg = aggregate_cop_results(all_results)
    for mode in ["heating", "cooling"]:
        agg_df = cop_agg[f"{mode}_aggregate"]
        if not agg_df.empty:
            agg_df.to_csv(output_dir / f"{mode}_cop_by_delta_t_aggregate.csv", index=False)
            slope, intercept, r2 = cop_agg[f"{mode}_slope_agg"]
            print(f"{mode.capitalize()} COP aggregate: COP = {intercept:.3f} - {slope:.4f} * delta_T  (R² = {r2:.3f})")

        per_bldg_df = cop_agg[f"{mode}_per_building"]
        if not per_bldg_df.empty:
            per_bldg_df.to_parquet(
                output_dir / f"{mode}_cop_by_delta_t_per_building.parquet",
                engine="pyarrow", index=False,
            )

    # --- Aggregate Capacity ---
    cap_agg = aggregate_capacity_results(all_results)
    for mode in ["heating", "cooling"]:
        agg_df = cap_agg[f"{mode}_cap_aggregate"]
        if not agg_df.empty:
            agg_df.to_csv(output_dir / f"{mode}_capacity_by_delta_t_aggregate.csv", index=False)
            rated, loss, peak_dt, r2 = cap_agg[f"{mode}_cap_fit_agg"]
            print(f"{mode.capitalize()} capacity aggregate: rated = {rated:.4f} kW, "
                  f"loss = {loss:.6f} kW/°C, peak at delta_T = {peak_dt:.1f}°C  (R² = {r2:.3f})")

        per_bldg_df = cap_agg[f"{mode}_cap_per_building"]
        if not per_bldg_df.empty:
            per_bldg_df.to_parquet(
                output_dir / f"{mode}_capacity_by_delta_t_per_building.parquet",
                engine="pyarrow", index=False,
            )

    print(f"All outputs saved to {output_dir}")

    # --- Plots ---
    plot_cop_spread(output_dir, metadata_df)
    plot_capacity_spread(output_dir, metadata_df)


# ---------------------------------------------------------------------------
# Plotting: COP spread by HVAC type
# ---------------------------------------------------------------------------

def plot_cop_spread(output_dir: Path, metadata_df: pd.DataFrame = None):
    """Generate COP and COP-loss plots grouped by HVAC type."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    if metadata_df is None:
        base_dir = output_dir.parents[1]
        metadata_df = load_metadata(base_dir / "in" / "TX_upgrade0.parquet")

    summary_df = pd.read_csv(output_dir / "cop_capacity_summary.csv")

    hvac_map = metadata_df[["heating_category", "cooling_category"]].copy()
    hvac_map.index.name = "building_id"
    summary_df = summary_df.merge(hvac_map, left_on="building_id", right_index=True, how="left")

    # --- Fit COP per HVAC category ---
    category_fit_rows = []
    for mode, type_col, n_obs_col, per_bldg_file in [
        ("heating", "heating_category", "heating_cop_n_obs",
         "heating_cop_by_delta_t_per_building.parquet"),
        ("cooling", "cooling_category", "cooling_cop_n_obs",
         "cooling_cop_by_delta_t_per_building.parquet"),
    ]:
        df_mode = summary_df[summary_df[n_obs_col] > 0].copy()
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

    # --- Plots ---
    mode_configs = [
        {"mode": "heating", "type_col": "heating_category",
         "slope_col": "heating_cop_slope", "intercept_col": "heating_cop_intercept",
         "n_obs_col": "heating_cop_n_obs",
         "per_bldg_file": "heating_cop_by_delta_t_per_building.parquet"},
        {"mode": "cooling", "type_col": "cooling_category",
         "slope_col": "cooling_cop_slope", "intercept_col": "cooling_cop_intercept",
         "n_obs_col": "cooling_cop_n_obs",
         "per_bldg_file": "cooling_cop_by_delta_t_per_building.parquet"},
    ]

    for cfg in mode_configs:
        mode = cfg["mode"]
        type_col = cfg["type_col"]
        df_mode = summary_df[summary_df[cfg["n_obs_col"]] > 0].copy()
        if df_mode.empty:
            continue

        hvac_types = sorted(df_mode[type_col].dropna().unique())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(hvac_types), 1)))

        # Plot 1: COP vs delta_T lines
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

        # Plot 2: Box plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        box_slope = [df_mode.loc[df_mode[type_col] == ht, cfg["slope_col"]].dropna() for ht in hvac_types]
        box_intercept = [df_mode.loc[df_mode[type_col] == ht, cfg["intercept_col"]].dropna() for ht in hvac_types]
        valid = [(ht, s, ic) for ht, s, ic in zip(hvac_types, box_slope, box_intercept) if len(s) >= 2]
        # if valid:
        #     v_types, v_slopes, v_intercepts = zip(*valid)
        #     labels = [f"{t} (n={len(s)})" for t, s in zip(v_types, v_slopes)]
        #     bp1 = ax1.boxplot(v_slopes, tick_labels=labels, patch_artist=True, showfliers=False)
        #     for patch, c in zip(bp1["boxes"], colors): patch.set_facecolor(c); patch.set_alpha(0.5)
        #     ax1.set_ylabel("COP Loss (COP/°C)"); ax1.set_title(f"{mode.capitalize()} COP Loss Slope")
        #     ax1.tick_params(axis="x", rotation=30, labelsize=8); ax1.grid(True, alpha=0.3, axis="y")
        #     bp2 = ax2.boxplot(v_intercepts, tick_labels=labels, patch_artist=True, showfliers=False)
        #     for patch, c in zip(bp2["boxes"], colors): patch.set_facecolor(c); patch.set_alpha(0.5)
        #     ax2.set_ylabel("COP Intercept"); ax2.set_title(f"{mode.capitalize()} COP Intercept")
        #     ax2.tick_params(axis="x", rotation=30, labelsize=8); ax2.grid(True, alpha=0.3, axis="y")
        if valid:
            v_types, v_slopes, v_intercepts = zip(*valid)
            labels = [f"{t} (n={len(s)})" for t, s in zip(v_types, v_slopes)]
            valid_colors = [colors[hvac_types.index(t)] for t in v_types]

            bp1 = ax1.boxplot(v_slopes, patch_artist=True, showfliers=False)
            for patch, c in zip(bp1["boxes"], valid_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
            ax1.set_xticks(range(1, len(labels) + 1))
            ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax1.set_ylabel("COP Loss (COP/°C)")
            ax1.set_title(f"{mode.capitalize()} COP Loss Slope")
            ax1.grid(True, alpha=0.3, axis="y")

            bp2 = ax2.boxplot(v_intercepts, patch_artist=True, showfliers=False)
            for patch, c in zip(bp2["boxes"], valid_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
            ax2.set_xticks(range(1, len(labels) + 1))
            ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax2.set_ylabel("COP Intercept")
            ax2.set_title(f"{mode.capitalize()} COP Intercept")
            ax2.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(output_dir / f"{mode}_cop_boxplots_by_type.png", dpi=150)
        plt.close(fig)
        print(f"Saved {mode}_cop_boxplots_by_type.png")

        # Plot 3: Scatter
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, htype in enumerate(hvac_types):
            sub = df_mode[df_mode[type_col] == htype].dropna(subset=[cfg["slope_col"], cfg["intercept_col"]])
            if sub.empty:
                continue
            ax.scatter(sub[cfg["slope_col"]], sub[cfg["intercept_col"]],
                       label=f"{htype} (n={len(sub)})",
                       color=colors[i], alpha=0.4, s=15, edgecolors="none")
        ax.set_xlabel("COP Loss Slope (COP/°C)"); ax.set_ylabel("COP Intercept")
        ax.set_title(f"{mode.capitalize()} COP: Intercept vs Loss Slope")
        ax.legend(fontsize=8, loc="best", markerscale=3); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{mode}_cop_scatter_by_type.png", dpi=150)
        plt.close(fig)
        print(f"Saved {mode}_cop_scatter_by_type.png")


# ---------------------------------------------------------------------------
# Plotting: Capacity spread by HVAC type
# ---------------------------------------------------------------------------

def plot_capacity_spread(output_dir: Path, metadata_df: pd.DataFrame = None):
    """Generate capacity and capacity-loss plots grouped by HVAC type."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    if metadata_df is None:
        base_dir = output_dir.parents[1]
        metadata_df = load_metadata(base_dir / "in" / "TX_upgrade0.parquet")

    summary_df = pd.read_csv(output_dir / "cop_capacity_summary.csv")

    hvac_map = metadata_df[["heating_category", "cooling_category"]].copy()
    hvac_map.index.name = "building_id"
    summary_df = summary_df.merge(hvac_map, left_on="building_id", right_index=True, how="left")

    # --- Fit capacity per HVAC category ---
    category_fit_rows = []
    for mode, type_col, n_obs_col, per_bldg_file in [
        ("heating", "heating_category", "heating_cap_n_obs",
         "heating_capacity_by_delta_t_per_building.parquet"),
        ("cooling", "cooling_category", "cooling_cap_n_obs",
         "cooling_capacity_by_delta_t_per_building.parquet"),
    ]:
        df_mode = summary_df[summary_df[n_obs_col] > 0].copy()
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
            binned = sub.groupby("delta_t_bin_center").apply(
                lambda g: pd.Series({
                    "cap_95pct_kw": np.average(g["cap_95pct_kw"], weights=g["cap_count"]),
                    "n_observations": g["cap_count"].sum(),
                    "n_buildings": g["building_id"].nunique(),
                }),
                include_groups=False,
            ).reset_index()
            binned = binned[binned["n_buildings"] >= 3]
            if len(binned) < 3:
                continue
            rated, loss, peak_dt, r2 = fit_capacity_from_bins(binned)
            n_bldg = df_mode[df_mode[type_col] == htype]["building_id"].nunique()
            category_fit_rows.append({
                "mode": mode,
                "hvac_type": htype,
                "rated_cap_kw": rated,
                "cap_loss_kw_per_c": loss,
                "peak_delta_t_c": peak_dt,
                "r2": r2,
                "n_buildings": n_bldg,
                "n_observations": int(binned["n_observations"].sum()),
            })

    if category_fit_rows:
        cat_df = pd.DataFrame(category_fit_rows)
        cat_df.to_csv(output_dir / "capacity_fit_by_hvac_category.csv", index=False)
        print(f"Saved capacity fits by HVAC category ({len(cat_df)} rows).")

    # --- Plots ---
    mode_configs = [
        {"mode": "heating", "type_col": "heating_category",
         "slope_col": "heating_cap_loss_kw_per_c", "intercept_col": "heating_rated_cap_kw",
         "n_obs_col": "heating_cap_n_obs",
         "per_bldg_file": "heating_capacity_by_delta_t_per_building.parquet"},
        {"mode": "cooling", "type_col": "cooling_category",
         "slope_col": "cooling_cap_loss_kw_per_c", "intercept_col": "cooling_rated_cap_kw",
         "n_obs_col": "cooling_cap_n_obs",
         "per_bldg_file": "cooling_capacity_by_delta_t_per_building.parquet"},
    ]

    for cfg in mode_configs:
        mode = cfg["mode"]
        type_col = cfg["type_col"]
        df_mode = summary_df[summary_df[cfg["n_obs_col"]] > 0].copy()
        if df_mode.empty:
            continue

        hvac_types = sorted(df_mode[type_col].dropna().unique())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(hvac_types), 1)))

        # Plot 1: Capacity vs delta_T lines
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
                    cap_mean=("cap_95pct_kw", lambda x: np.average(x, weights=sub.loc[x.index, "cap_count"])),
                    cap_25=("cap_95pct_kw", lambda x: np.percentile(x, 25)),
                    cap_75=("cap_95pct_kw", lambda x: np.percentile(x, 75)),
                    n_bldg=("building_id", "nunique"),
                ).reset_index()
                binned = binned[binned["n_bldg"] >= 3]
                if binned.empty:
                    continue
                ax.plot(binned["delta_t_bin_center"], binned["cap_mean"],
                        label=f"{htype} (n={sub['building_id'].nunique()})",
                        color=colors[i], linewidth=2)
                ax.fill_between(binned["delta_t_bin_center"],
                                binned["cap_25"], binned["cap_75"],
                                alpha=0.15, color=colors[i])
            ax.set_xlabel("Delta T (°C)")
            ax.set_ylabel("Capacity (kW)")
            ax.set_title(f"{mode.capitalize()} Capacity (95th pct, kW) vs Delta-T by HVAC Type")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / f"{mode}_capacity_vs_delta_t_by_type.png", dpi=150)
            plt.close(fig)
            print(f"Saved {mode}_capacity_vs_delta_t_by_type.png")

        # Plot 2: Box plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        box_slope = [df_mode.loc[df_mode[type_col] == ht, cfg["slope_col"]].dropna() for ht in hvac_types]
        box_intercept = [df_mode.loc[df_mode[type_col] == ht, cfg["intercept_col"]].dropna() for ht in hvac_types]
        valid = [(ht, s, ic) for ht, s, ic in zip(hvac_types, box_slope, box_intercept) if len(s) >= 2]
        if valid:
            v_types, v_slopes, v_intercepts = zip(*valid)
            labels = [f"{t} (n={len(s)})" for t, s in zip(v_types, v_slopes)]
            bp1 = ax1.boxplot(v_slopes, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch, c in zip(bp1["boxes"], colors): patch.set_facecolor(c); patch.set_alpha(0.5)
            ax1.set_ylabel("Capacity Loss (kW/°C)"); ax1.set_title(f"{mode.capitalize()} Capacity Loss Slope")
            ax1.tick_params(axis="x", rotation=30, labelsize=8); ax1.grid(True, alpha=0.3, axis="y")
            bp2 = ax2.boxplot(v_intercepts, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch, c in zip(bp2["boxes"], colors): patch.set_facecolor(c); patch.set_alpha(0.5)
            ax2.set_ylabel("Rated Capacity (kW)"); ax2.set_title(f"{mode.capitalize()} Rated Capacity")
            ax2.tick_params(axis="x", rotation=30, labelsize=8); ax2.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(output_dir / f"{mode}_capacity_boxplots_by_type.png", dpi=150)
        plt.close(fig)
        print(f"Saved {mode}_capacity_boxplots_by_type.png")

        # Plot 3: Scatter
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, htype in enumerate(hvac_types):
            sub = df_mode[df_mode[type_col] == htype].dropna(subset=[cfg["slope_col"], cfg["intercept_col"]])
            if sub.empty:
                continue
            ax.scatter(sub[cfg["slope_col"]], sub[cfg["intercept_col"]],
                       label=f"{htype} (n={len(sub)})",
                       color=colors[i], alpha=0.4, s=15, edgecolors="none")
        ax.set_xlabel("Capacity Loss (kW/°C)"); ax.set_ylabel("Rated Capacity (kW)")
        ax.set_title(f"{mode.capitalize()} Capacity: Rated vs Loss Slope")
        ax.legend(fontsize=8, loc="best", markerscale=3); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{mode}_capacity_scatter_by_type.png", dpi=150)
        plt.close(fig)
        print(f"Saved {mode}_capacity_scatter_by_type.png")


if __name__ == "__main__":
    main()