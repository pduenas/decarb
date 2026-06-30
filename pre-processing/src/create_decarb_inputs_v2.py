#!/usr/bin/env python3
"""
Batch Building Energy Data Generator

Reads all building IDs from out/ercot_substation_nrel_map.parquet
and runs generate_building_data + save_building_data for each one.

Edit the CONFIG block below, then run:
    python run_all_buildings.py
"""

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


# ===========================================================================
# CONFIG — edit these before running
# ===========================================================================

TIMESTEP             = "hourly"   # "hourly" or "15min"
YEAR                 = 2018

TARIFF_TYPE          = "flat"     # "flat" or "tou"
TARIFF_VALUE         = 0.22       # flat rate $/kWh (also TOU off-peak base)
PEAK_START           = "16:00"    # TOU peak start (ignored if flat)
PEAK_END             = "19:00"    # TOU peak end   (ignored if flat)
PEAK_PRICE           = 0.28       # TOU peak rate  (ignored if flat)

CAPACITY_CHARGE      = False
CAP_PEAK_START       = "19:00"
CAP_PEAK_END         = "21:00"
CAP_CHARGE_VALUE     = 0.5

CO2_COST             = 0.5
GAS_COST             = 0.051
LIQUID_COST          = 0.0739

# Case naming — output path becomes in/decarb_inputs/{CASE_NAME}/{bldg_id}/update_{UPDATE_NUM}/in/
UPDATE_NUM           = 0
ELEC                 = 0             # electrification level
TARIFF_LABEL         = "flat"    # short label for the tariff configuration
CASE_NAME            = f"elec_{ELEC}_{TARIFF_LABEL}"

# Set to a list of ints to run only specific buildings, e.g. [77, 250, 312]
# Set to None to run every building in the map file
BLDG_IDS_OVERRIDE    = None

# Debug mode — runs on a random sample of buildings
DEBUG                = False
DEBUG_N              = 10
DEBUG_SEED           = 42

# Set to True to skip buildings that already have all output files
RESUME               = False

# Set to True to retry only buildings listed in the batch_failures.log
RETRY_FAILED         = False

# Output file selection — set to False to skip files you've already generated
RUN_TM               = False
RUN_BDG_I            = False
RUN_BDG_II           = False
RUN_IN               = False   # in.csv
RUN_OTHER            = False   # abs.csv, pv.csv, bess.csv, wind.csv, ev.csv
RUN_CHP              = True   # chp.csv
RUN_TOPO             = False   # topo.csv
RUN_HVAC             = True   # hvac.csv
RUN_WH               = False   # wh.csv
RUN_SP               = True   # sp.csv

# ===========================================================================

FT2_TO_M2        = 0.092903
OTHER_TEMPLATE_FILES = ["abs.csv", "pv.csv", "bess.csv", "wind.csv", "ev.csv"]


# ---------------------------------------------------------------------------
# Elevation lookup helpers
# ---------------------------------------------------------------------------

def fetch_elevation_usgs(lat: float, lon: float) -> float:
    """Fetch elevation in metres for a single (lat, lon) from the USGS EPQS API."""
    url = (
        f"https://epqs.nationalmap.gov/v1/json"
        f"?x={lon}&y={lat}&units=Meters&includeDate=false"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return float(data["value"])
    except Exception:
        return 0.0


def build_elevation_cache(unique_coords: list[tuple[float, float]],
                          max_workers: int = 20) -> dict[tuple[float, float], float]:
    """
    Fetch elevations for all unique (lat, lon) pairs in parallel.
    Returns a dict mapping (lat, lon) -> elevation_metres.
    """
    cache: dict[tuple[float, float], float] = {}
    print(f"Fetching elevation for {len(unique_coords)} unique coordinates "
          f"(using {max_workers} threads)…")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_coord = {
            pool.submit(fetch_elevation_usgs, lat, lon): (lat, lon)
            for lat, lon in unique_coords
        }
        for future in as_completed(future_to_coord):
            coord = future_to_coord[future]
            cache[coord] = future.result()

    print("Elevation cache ready.")
    return cache


# ---------------------------------------------------------------------------
# Template loaders — called once at startup
# ---------------------------------------------------------------------------

def load_templates(parent_path: Path) -> dict:
    """
    Load all CSV templates from in/decarb_input_templates/ once at startup.
    Returns a dict with keys: 'bdg_i', 'in', and each name in OTHER_TEMPLATE_FILES.
    """
    template_dir = parent_path / "in" / "decarb_input_templates"
    templates = {}

    if RUN_BDG_I:
        templates["bdg_i"] = pd.read_csv(
            template_dir / "bdg_i.csv", header=None, index_col=0)

    if RUN_IN:
        templates["in"] = pd.read_csv(
            template_dir / "in.csv", header=None, index_col=0)

    if RUN_OTHER:
        for fname in OTHER_TEMPLATE_FILES:
            templates[fname] = pd.read_csv(
                template_dir / fname, header=None, index_col=0)

    if RUN_CHP or RUN_SP:
        templates["chp"] = pd.read_csv(
            template_dir / "chp.csv")

    if RUN_TOPO:
        templates["topo"] = pd.read_csv(
            template_dir / "topo.csv")

    if RUN_HVAC or RUN_SP:
        templates["hvac"] = pd.read_csv(
            template_dir / "hvac.csv")

    if RUN_WH or RUN_SP:
        templates["wh"] = pd.read_csv(
            template_dir / "wh.csv")

    if RUN_SP:
        templates["sp"] = pd.read_csv(
            template_dir / "sp.csv")

    return templates


# ---------------------------------------------------------------------------
# Per-building data loaders
# ---------------------------------------------------------------------------

def load_consumption(bldg_id: int, parent_path: Path, year: int) -> pd.DataFrame:
    """Load and index the consumption parquet/CSV for a single building."""
    parquet = parent_path / "out" / "consumption_files" / f"{bldg_id}-0.parquet"
    csv     = parent_path / "out" / "consumption_files" / f"{bldg_id}-0.csv"

    if parquet.exists():
        try:
            df = pd.read_parquet(parquet, engine="fastparquet")
        except Exception:
            df = pd.read_parquet(parquet, engine="pyarrow")
    elif csv.exists():
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(f"Consumption file not found for building {bldg_id}")

    if not isinstance(df.index, pd.DatetimeIndex):
        if len(df) == 35040:
            df.index = pd.date_range(
                start=datetime(year, 1, 1, 0, 0), periods=35040, freq="15min")
        elif len(df) == 8760:
            df.index = pd.date_range(
                start=datetime(year, 1, 1, 0, 0), periods=8760, freq="h")
        else:
            raise ValueError(
                f"Unexpected consumption data length for building {bldg_id}: {len(df)}")
    else:
        df.index = df.index - pd.Timedelta(minutes=15)

    return df


def load_setpoints(bldg_id: int, parent_path: Path, year: int) -> pd.DataFrame:
    """Load and index the setpoint timeseries for a single building."""
    parquet = parent_path / "out" / "setpoint_timeseries" / f"setpoint_{bldg_id}.parquet"
    csv     = parent_path / "out" / "setpoint_timeseries" / f"setpoint_{bldg_id}.csv"

    if parquet.exists():
        try:
            df = pd.read_parquet(parquet, engine="fastparquet")
        except Exception:
            df = pd.read_parquet(parquet, engine="pyarrow")
    elif csv.exists():
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(f"Setpoint file not found for building {bldg_id}")

    if not isinstance(df.index, pd.DatetimeIndex):
        if len(df) == 35040:
            df.index = pd.date_range(
                start=datetime(year, 1, 1, 0, 0), periods=35040, freq="15min")
        elif len(df) == 8760:
            df.index = pd.date_range(
                start=datetime(year, 1, 1, 0, 0), periods=8760, freq="h")
        else:
            raise ValueError(
                f"Unexpected setpoint data length for building {bldg_id}: {len(df)}")
    else:
        df.index = df.index - pd.Timedelta(minutes=15)

    return df


def get_building_row(bldg_id: int, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the single metadata row for a building."""
    if "bldg_id" in metadata_df.columns:
        row = metadata_df[metadata_df["bldg_id"] == bldg_id]
    elif "building_id" in metadata_df.columns:
        row = metadata_df[metadata_df["building_id"] == bldg_id]
    elif bldg_id in metadata_df.index:
        row = metadata_df.loc[bldg_id:bldg_id]
    else:
        row = metadata_df[metadata_df.index.astype(str) == str(bldg_id)]

    if len(row) == 0:
        raise ValueError(f"Building {bldg_id} not found in metadata")
    return row


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def resample_if_needed(series, timestep, method="sum"):
    """Resample a series from 15-min to hourly if needed."""
    if timestep == "hourly" and len(series) == 35040:
        if method == "sum":
            resampled = series.resample("h").sum()
        elif method == "mean":
            resampled = series.resample("h").mean()
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        if len(resampled) > 8760:
            resampled = resampled.iloc[:8760]
        return resampled
    return series


# ---------------------------------------------------------------------------
# File generators
# ---------------------------------------------------------------------------

def generate_tm_data(consumption_df, setpoint_df, building_row):
    """
    Generate the tm DataFrame for a building.
    All inputs are pre-loaded; no file I/O happens here.
    """
    timestep      = TIMESTEP
    year          = YEAR
    tariff_type   = TARIFF_TYPE
    tariff_value  = TARIFF_VALUE
    peak_start    = PEAK_START
    peak_end      = PEAK_END
    peak_price    = PEAK_PRICE
    capacity_charge       = CAPACITY_CHARGE
    capacity_peak_start   = CAP_PEAK_START
    capacity_peak_end     = CAP_PEAK_END
    capacity_charge_value = CAP_CHARGE_VALUE
    co2_cost      = CO2_COST
    gas_cost      = GAS_COST
    liquid_cost   = LIQUID_COST

    # HVAC availability from metadata
    hvac_data = {
        col: building_row[col].iloc[0] if col in building_row.columns else None
        for col in ("in.heating_unavailable_period", "in.cooling_unavailable_period")
    }

    # Step 1: Generate timestamps
    if timestep == "hourly":
        n_steps, freq = 8760, "h"
    elif timestep == "15min":
        n_steps, freq = 35040, "15min"
    else:
        raise ValueError("timestep must be 'hourly' or '15min'")

    timestamps = pd.date_range(
        start=datetime(year, 1, 1, 0, 0), periods=n_steps, freq=freq)

    dst_spring = datetime(year, 3, 11, 2, 0)
    if dst_spring in timestamps:
        mask = timestamps >= dst_spring
        timestamps = pd.DatetimeIndex([
            ts + pd.Timedelta(hours=1) if mask[i] else ts
            for i, ts in enumerate(timestamps)
        ])

    df = pd.DataFrame(index=timestamps)
    df.index.name = "pP"

    # Step 2: Tariff (pQcostBuy)
    if tariff_type == "flat":
        df["pQcostBuy"] = tariff_value
    elif tariff_type == "tou":
        off_peak_price  = peak_price / 2.0
        peak_start_time = datetime.strptime(peak_start, "%H:%M").time()
        peak_end_time   = datetime.strptime(peak_end,   "%H:%M").time()
        ts_times        = df.index.time
        if peak_start_time < peak_end_time:
            is_peak = (ts_times >= peak_start_time) & (ts_times < peak_end_time)
        else:
            is_peak = (ts_times >= peak_start_time) | (ts_times < peak_end_time)
        df["pQcostBuy"] = off_peak_price
        df.loc[is_peak, "pQcostBuy"] = peak_price
    else:
        raise ValueError("tariff_type must be 'flat' or 'tou'")

    # Step 2b: Sellback tariff (pQcostSell)
    df["pQcostSell"] = 0.18

    # Step 3: Capacity charge (pQmx, pQmxCost)
    if capacity_charge:
        cap_start_time = datetime.strptime(capacity_peak_start, "%H:%M").time()
        cap_end_time   = datetime.strptime(capacity_peak_end,   "%H:%M").time()
        ts_times       = df.index.time
        if cap_start_time < cap_end_time:
            is_cap_peak = (ts_times >= cap_start_time) & (ts_times < cap_end_time)
        else:
            is_cap_peak = (ts_times >= cap_start_time) | (ts_times < cap_end_time)
        df["pQmx"] = 0
        df.loc[is_cap_peak, "pQmx"] = 1
        df["pQmxCost"] = 0.0
        df.loc[is_cap_peak, "pQmxCost"] = capacity_charge_value
    else:
        df["pQmx"]    = 0
        df["pQmxCost"] = 0.0

    # Step 4: Constant cost columns
    df["pQco2"]  = co2_cost
    df["pGcost"] = gas_cost
    df["pLcost"] = liquid_cost

    # Step 5: Lighting (pQlight)
    lighting_cols = [
        "out.electricity.lighting_exterior.energy_consumption..kwh",
        "out.electricity.lighting_garage.energy_consumption..kwh",
        "out.electricity.lighting_interior.energy_consumption..kwh",
        "out.natural_gas.lighting.energy_consumption..kwh",
    ]
    avail = [c for c in lighting_cols if c in consumption_df.columns]
    if not avail:
        raise ValueError("No lighting consumption columns found in consumption file")
    lighting_15min = consumption_df[avail].sum(axis=1)
    if timestep == "15min":
        df["pQlight"] = lighting_15min.values * 4
    else:
        df["pQlight"] = resample_if_needed(lighting_15min, timestep, "sum").values

    # Step 6: Heating energy
    heating_cols = [
        "out.fuel_oil.heating.energy_consumption..kwh",
        "out.propane.heating.energy_consumption..kwh",
        "out.natural_gas.heating.energy_consumption..kwh",
        "out.electricity.heating.energy_consumption..kwh",
        "out.electricity.heating_fans_pumps.energy_consumption..kwh",
        "out.natural_gas.heating_hp_bkup.energy_consumption..kwh",
        "out.electricity.heating_hp_bkup.energy_consumption..kwh",
        "out.electricity.heating_hp_bkup_fa.energy_consumption..kwh",
    ]
    avail = [c for c in heating_cols if c in consumption_df.columns]
    heating_kwh = (
        resample_if_needed(consumption_df[avail].sum(axis=1), timestep, "sum").values
        if avail else None
    )

    # Step 7: Cooling energy
    cooling_cols = [
        "out.electricity.cooling.energy_consumption..kwh",
        "out.electricity.cooling_fans_pumps.energy_consumption..kwh",
    ]
    avail = [c for c in cooling_cols if c in consumption_df.columns]
    cooling_kwh = (
        resample_if_needed(consumption_df[avail].sum(axis=1), timestep, "sum").values
        if avail else None
    )

    # Step 8: EV charging
    ev_col = "out.electricity.ev_charging.energy_consumption..kwh"
    ev_kwh = (
        resample_if_needed(consumption_df[ev_col], timestep, "sum").values
        if ev_col in consumption_df.columns else None
    )

    # Step 9: Hot water energy
    hw_cols = [
        "out.electricity.hot_water.energy_consumption..kwh",
        "out.electricity.hot_water_solar_th.energy_consumption..kwh",
        "out.fuel_oil.hot_water.energy_consumption..kwh",
        "out.natural_gas.hot_water.energy_consumption..kwh",
        "out.propane.hot_water.energy_consumption..kwh",
    ]
    avail = [c for c in hw_cols if c in consumption_df.columns]
    hw_kwh = (
        resample_if_needed(consumption_df[avail].sum(axis=1), timestep, "sum").values
        if avail else None
    )

    # Step 10: Total energy
    total_cols = [
        "out.electricity.total.energy_consumption..kwh",
        "out.fuel_oil.total.energy_consumption..kwh",
        "out.natural_gas.total.energy_consumption..kwh",
        "out.propane.total.energy_consumption..kwh",
    ]
    avail = [c for c in total_cols if c in consumption_df.columns]
    if not avail:
        raise ValueError("Cannot calculate equipment power: total energy data not available")
    total_kwh = resample_if_needed(consumption_df[avail].sum(axis=1), timestep, "sum").values

    # Step 11: Equipment power (pQequip)
    equip_kwh = total_kwh.copy()
    for part in (heating_kwh, cooling_kwh, ev_kwh, hw_kwh):
        if part is not None:
            equip_kwh -= part
    df["pQequip"] = equip_kwh if timestep == "hourly" else equip_kwh * 4

    # Step 12: Outdoor temperature (pTout)
    temp_col = "out.outdoor_air_drybulb_temp..c"
    if temp_col not in consumption_df.columns:
        raise ValueError(f"Outdoor temperature column not found: {temp_col}")
    df["pTout"] = resample_if_needed(consumption_df[temp_col], timestep, "mean").values

    # Step 13: Temperature setpoints (pTmx, pTmn)
    if "max_comfort_temp" not in setpoint_df.columns:
        raise ValueError("max_comfort_temp column not found in setpoint file")
    if "min_comfort_temp" not in setpoint_df.columns:
        raise ValueError("min_comfort_temp column not found in setpoint file")
    df["pTmx"] = resample_if_needed(setpoint_df["max_comfort_temp"], timestep, "mean").values
    df["pTmn"] = resample_if_needed(setpoint_df["min_comfort_temp"], timestep, "mean").values

    # Step 14: HVAC on/off status (pTon)
    if "mode" not in setpoint_df.columns:
        raise ValueError("mode column not found in setpoint file")
    if timestep == "15min":
        mode_series = setpoint_df["mode"]
    else:
        mode_series = resample_if_needed(setpoint_df["mode"], timestep, "mean").round()

    df["pTon"] = 1

    def apply_unavail(unavail_str, mode_vals):
        if not unavail_str or str(unavail_str).lower() in ("never", "nan", "none"):
            return
        if str(unavail_str).lower() in ("year round", "year-round"):
            df.loc[(mode_vals == 1) | (mode_vals == 0), "pTon"] = 0
            return
        try:
            parts = str(unavail_str).split("-")
            if len(parts) == 2:
                s = datetime.strptime(f"{parts[0].strip()} {year}", "%b %d %Y")
                e = datetime.strptime(f"{parts[1].strip()} {year}", "%b %d %Y")
                if e < s:
                    period_mask = (
                        (df.index.month > s.month) |
                        ((df.index.month == s.month) & (df.index.day >= s.day)) |
                        (df.index.month < e.month) |
                        ((df.index.month == e.month) & (df.index.day <= e.day))
                    )
                else:
                    period_mask = (
                        ((df.index.month > s.month) |
                         ((df.index.month == s.month) & (df.index.day >= s.day))) &
                        ((df.index.month < e.month) |
                         ((df.index.month == e.month) & (df.index.day <= e.day)))
                    )
                season_mask = (mode_vals == 1) | (mode_vals == 0)
                df.loc[period_mask & season_mask, "pTon"] = 0
        except Exception:
            pass

    apply_unavail(hvac_data.get("in.heating_unavailable_period"), mode_series.values)

    def apply_cooling_unavail(unavail_str, mode_vals):
        if not unavail_str or str(unavail_str).lower() in ("never", "nan", "none"):
            return
        if str(unavail_str).lower() in ("year round", "year-round"):
            df.loc[(mode_vals == -1) | (mode_vals == 0), "pTon"] = 0
            return
        try:
            parts = str(unavail_str).split("-")
            if len(parts) == 2:
                s = datetime.strptime(f"{parts[0].strip()} {year}", "%b %d %Y")
                e = datetime.strptime(f"{parts[1].strip()} {year}", "%b %d %Y")
                if e < s:
                    period_mask = (
                        (df.index.month > s.month) |
                        ((df.index.month == s.month) & (df.index.day >= s.day)) |
                        (df.index.month < e.month) |
                        ((df.index.month == e.month) & (df.index.day <= e.day))
                    )
                else:
                    period_mask = (
                        ((df.index.month > s.month) |
                         ((df.index.month == s.month) & (df.index.day >= s.day))) &
                        ((df.index.month < e.month) |
                         ((df.index.month == e.month) & (df.index.day <= e.day)))
                    )
                season_mask = (mode_vals == -1) | (mode_vals == 0)
                df.loc[period_mask & season_mask, "pTon"] = 0
        except Exception:
            pass

    apply_cooling_unavail(hvac_data.get("in.cooling_unavailable_period"), mode_series.values)

    # Step 15: Hot water demand (pHWdem)
    hw_load_cols = [
        "out.load.hot_water.energy_delivered..kbtu",
        "out.load.hot_water_solar_thermal..kbtu",
        "out.load.hot_water_tank_losses..kbtu",
    ]
    avail = [c for c in hw_load_cols if c in consumption_df.columns]
    if avail:
        hw_kbtu = consumption_df[avail].sum(axis=1) * 0.293071
        df["pHWdem"] = resample_if_needed(hw_kbtu, timestep, "sum").values
    else:
        df["pHWdem"] = 0.0

    # Step 16: Solar radiation and wind speed
    for out_col, src_col, scale in [
        ("pSdni", "out.weather.direct_normal_solar_radiation..watt_per_m2", 1000.0),
        ("pSdhi", "out.weather.diffuse_solar_radiation..watt_per_m2",       1000.0),
    ]:
        if src_col in consumption_df.columns:
            df[out_col] = resample_if_needed(
                consumption_df[src_col], timestep, "mean").values / scale
        else:
            df[out_col] = 0.0

    wind_col = "out.weather.wind_speed..meter_per_second"
    df["pWms"] = (
        resample_if_needed(consumption_df[wind_col], timestep, "mean").values
        if wind_col in consumption_df.columns else 0.0
    )

    # Step 17: EV columns (placeholder zeros)
    df["pEVdem"]  = 0.0
    df["pEVtm"]   = 0.0
    df["pEVtype"] = 0

    # Step 18: Occupants (pBppl)
    occ_sched_col = "out.schedules.occupants"
    occ_meta_col  = "in.occupants"
    if occ_sched_col in consumption_df.columns:
        occ_sched = resample_if_needed(consumption_df[occ_sched_col], timestep, "mean")
        if occ_meta_col in building_row.columns:
            raw = str(building_row[occ_meta_col].iloc[0]).strip()
            total = 15 if raw == "10+" else int(raw)
            df["pBppl"] = (occ_sched.fillna(0).values * total).round().astype(int)
        else:
            df["pBppl"] = occ_sched.fillna(0).round().astype(int)
    else:
        df["pBppl"] = 0

    return df


def create_tm_file(tm_df: pd.DataFrame, output_dir: Path) -> None:
    """Save tm.csv for a building."""
    out = tm_df.copy()
    formatted = out.index.strftime("%Y-%m-%dT%H:%M").tolist()
    formatted = ["2018-11-04T00:59" if ts == "2018-11-04T01:00" else ts for ts in formatted]
    out.index = formatted
    out.index.name = "pP"
    out.to_csv(output_dir / "tm.csv", sep=",")


def create_bdg_i_file(output_dir: Path, template: pd.DataFrame,
                      building_row: pd.DataFrame, elevation_cache: dict,
                      pBhvac: int = 1) -> None:
    """Create bdg_i.csv from template with per-building values filled in."""
    df = template.copy()

    def set_val(key, value):
        if key in df.index:
            df.loc[key, df.columns[0]] = value

    lat = float(building_row["in.weather_file_latitude"].iloc[0])
    lon = float(building_row["in.weather_file_longitude"].iloc[0])
    set_val("pBalt", elevation_cache.get((lat, lon), 0.0))
    set_val("pBlat", lat)
    set_val("pBlon", lon)

    sqft = float(building_row["in.sqft..ft2"].iloc[0])
    set_val("pBfoot", sqft * FT2_TO_M2)

    insulation = str(building_row["in.insulation_slab"].iloc[0])
    set_val("pBslab", 0 if insulation.strip().lower() == "none" else 1)

    wall_ft2 = float(building_row["out.params.wall_area_above_grade_exterior..ft2"].iloc[0])
    wall_per_side = (wall_ft2 * FT2_TO_M2) / 4.0
    for col in ("pBwest", "pBeast", "pBnorth", "pBsouth"):
        set_val(col, wall_per_side)

    set_val("pBhvac", pBhvac)
    df.to_csv(output_dir / "bdg_i.csv", header=False)


def create_bdg_ii_file(output_dir: Path, bldg_id: int,
                       regression_coeffs: dict) -> None:
    """Save bdg_ii.csv with thermal model regression coefficients."""
    key = str(bldg_id)
    if key not in regression_coeffs:
        raise KeyError(f"Building {bldg_id} not found in regression_coeff.json")
    coeffs = regression_coeffs[key]
    # k2 and k3 are in °C/W from regression; model uses kW, so multiply by 1000
    pd.DataFrame([{
        "pBk1": coeffs["k1"],
        "pBk2": coeffs["k2"] * 1000,
        "pBk3": coeffs["k3"] * 1000,
    }]).to_csv(output_dir / "bdg_ii.csv", index=False)


def create_in_file(output_dir: Path, template: pd.DataFrame,
                   consumption_df: pd.DataFrame) -> None:
    """Create in.csv from template with pTin0 set from first timestep indoor temperature."""
    df = template.copy()
    temp_col = "out.indoor_temperature.conditioned_space..c"
    if temp_col not in consumption_df.columns:
        raise ValueError(f"Indoor temperature column not found: {temp_col}")
    if "pTin0" in df.index:
        df.loc["pTin0", df.columns[0]] = consumption_df[temp_col].iloc[0]
    df.to_csv(output_dir / "in.csv", header=False)


def create_other_files(output_dir: Path, templates: dict) -> None:
    """Copy each OTHER template CSV as-is into the building output directory."""
    for fname in OTHER_TEMPLATE_FILES:
        templates[fname].to_csv(output_dir / fname, header=False)


def _resolve_identifier(original: str, candidates: list, rng, bldg_id: int, role: str,
                        fuel_scoped: bool = False) -> str:
    """
    Try to resolve an identifier against a list of candidates.
    Order: exact → fuzzy close match → random (fuel-scoped if requested).
    Prints a warning on any substitution.
    """
    import difflib
    if original in candidates:
        return original
    # Fuzzy match
    close = difflib.get_close_matches(original, candidates, n=1, cutoff=0.6)
    if close:
        print(f"  WARNING bldg {bldg_id}: {role} '{original}' → fuzzy match '{close[0]}'")
        return close[0]
    # Random fallback — scope to same fuel if requested
    pool = candidates
    if fuel_scoped:
        fuel = original.split()[0] if original.split() else ""
        fuel_pool = [c for c in candidates if str(c).startswith(fuel)]
        if fuel_pool:
            pool = fuel_pool
    chosen = rng.choice(pool)
    print(f"  WARNING bldg {bldg_id}: {role} '{original}' → random fallback '{chosen}'")
    return chosen


def create_sp_file(output_dir: Path, template: pd.DataFrame,
                   building_row: pd.DataFrame,
                   hvac_df: pd.DataFrame, chp_df: pd.DataFrame,
                   wh_df: pd.DataFrame) -> None:
    """Create sp.csv for ELEC=0 (baseline) case."""
    import random
    rng = random.Random(42)

    bldg_id = building_row.iloc[0].get("bldg_id", building_row.iloc[0].get("building_id", "?"))
    df = template.copy()
    # Strip trailing zeros row before any modifications
    zeros_mask = df["pCHP0"].astype(str).isin(["0", "0.0"])
    zeros_row  = df[zeros_mask].copy()
    df         = df[~zeros_mask].copy()

    # --- Build identifiers ---
    heating_fuel = str(building_row["in.heating_fuel"].iloc[0])
    heating_eff  = str(building_row["in.hvac_heating_efficiency"].iloc[0])
    heating_id   = f"{heating_fuel} {heating_eff}".replace(",", "")
    cooling_id   = str(building_row["in.hvac_cooling_efficiency"].iloc[0]).replace(",", "")
    wh_id        = str(building_row["in.water_heater_efficiency"].iloc[0])

    # --- Hardcoded remaps (always warn) ---
    HEATING_REMAPS = {
        "Electricity MSHP SEER 14.5 8.2 HSPF": "Electricity ASHP SEER 15 8.5 HSPF",
        "Electricity MSHP SEER 29.3 14 HSPF":  "Electricity ASHP SEER 24 12 HSPF",
    }
    if heating_id in HEATING_REMAPS:
        remapped = HEATING_REMAPS[heating_id]
        print(f"  WARNING bldg {bldg_id}: heating '{heating_id}' → hardcoded remap '{remapped}'")
        heating_id = remapped

    if cooling_id == "Non-Ducted Heat Pump":
        print(f"  WARNING bldg {bldg_id}: cooling 'Non-Ducted Heat Pump' → 'Ducted Heat Pump'")
        cooling_id = "Ducted Heat Pump"

    if "Heat Pump" in wh_id:
        remapped = "Electric Heat Pump 80 gal"
        print(f"  WARNING bldg {bldg_id}: wh '{wh_id}' → '{remapped}'")
        wh_id = remapped

    # --- Resolve special heating cases ---
    heating_none = heating_id.strip().lower().endswith("none")
    if "Shared Heating" in heating_id:
        valid_heating = [t for t in hvac_df["ty"].tolist() + chp_df["ty"].tolist()
                         if t and str(t).lower() != "none"]
        fuel = heating_id.split()[0] if heating_id.split() else ""
        fuel_pool = [t for t in valid_heating if str(t).startswith(fuel)]
        pool = fuel_pool if fuel_pool else valid_heating
        chosen = rng.choice(pool)
        print(f"  WARNING bldg {bldg_id}: heating '{heating_id}' → random shared allocation '{chosen}'")
        heating_id = chosen

    # --- Resolve special cooling cases ---
    cooling_none      = cooling_id.strip().lower() == "none"
    cooling_ducted_hp = cooling_id == "Ducted Heat Pump"
    cooling_shared    = "Shared Cooling" in cooling_id

    if cooling_shared:
        valid_cooling = [t for t in hvac_df["ty"].values
                         if t and str(t).lower() != "none" and "Ducted Heat Pump" not in str(t)]
        chosen = rng.choice(valid_cooling)
        print(f"  WARNING bldg {bldg_id}: cooling '{cooling_id}' → random shared allocation '{chosen}'")
        cooling_id = chosen

    # --- Fuzzy/random fallback resolution ---
    all_hvac_chp = [t for t in hvac_df["ty"].tolist() + chp_df["ty"].tolist()
                    if t and str(t).lower() != "none"]
    valid_hvac   = [t for t in hvac_df["ty"].values
                    if t and str(t).lower() != "none" and "Ducted Heat Pump" not in str(t)]
    valid_wh     = [t for t in wh_df["ty"].values if t and str(t).lower() != "none"]

    if not heating_none:
        heating_in_hvac = heating_id in hvac_df["ty"].values
        heating_in_chp  = heating_id in chp_df["ty"].values
        if not heating_in_hvac and not heating_in_chp:
            heating_id = _resolve_identifier(heating_id, all_hvac_chp, rng, bldg_id, "heating", fuel_scoped=True)

    if not cooling_none and not cooling_ducted_hp:
        if cooling_id not in hvac_df["ty"].values:
            cooling_id = _resolve_identifier(cooling_id, valid_hvac, rng, bldg_id, "cooling")

    if wh_id not in wh_df["ty"].values:
        wh_id = _resolve_identifier(wh_id, valid_wh, rng, bldg_id, "wh")

    # --- Always: reset all CHP and HVAC flags ---
    df["pCHPz0"] = 0
    df["pCHPyn"] = "NO"
    df["pHVACz0"] = 0
    df["pHVACyn"] = "NO"

    # --- Heating logic ---
    if heating_none:
        pass  # all CHP flags stay 0
    elif heating_id in hvac_df["ty"].values:
        # Heating unit is in hvac.csv — goes to pHVAC0
        hvac_heating_row = hvac_df[hvac_df["ty"] == heating_id]
        is_ashp = (not hvac_heating_row.empty and
                   float(hvac_heating_row["pHVmx"].iloc[0]) > 0 and
                   float(hvac_heating_row["pACmx"].iloc[0]) > 0)
        if heating_id in df["pHVAC0"].values:
            df.loc[df["pHVAC0"] == heating_id, "pHVACz0"] = 1
        else:
            df.iloc[0, df.columns.get_loc("pHVAC0")] = heating_id
            df.iloc[0, df.columns.get_loc("pHVACz0")] = 1
        # If heating unit is not combined (no cooling), also place cooling identifier
        if not is_ashp and not cooling_none and not cooling_ducted_hp:
            if cooling_id in df["pHVAC0"].values:
                df.loc[df["pHVAC0"] == cooling_id, "pHVACz0"] = 1
            else:
                # Find a row that is not already set to z0=1
                available = df[df["pHVACz0"] != 1].index
                if len(available) > 0:
                    df.iloc[available[0], df.columns.get_loc("pHVAC0")] = cooling_id
                    df.iloc[available[0], df.columns.get_loc("pHVACz0")] = 1
                else:
                    df.iloc[0, df.columns.get_loc("pHVAC0")] = cooling_id
                    df.iloc[0, df.columns.get_loc("pHVACz0")] = 1
    else:
        # Heating unit is in chp.csv — goes to pCHP0
        if heating_id in df["pCHP0"].values:
            df.loc[df["pCHP0"] == heating_id, "pCHPz0"] = 1
        else:
            df.iloc[0, df.columns.get_loc("pCHP0")] = heating_id
            df.iloc[0, df.columns.get_loc("pCHPz0")] = 1

        if not cooling_none and not cooling_ducted_hp:
            if cooling_id in df["pHVAC0"].values:
                df.loc[df["pHVAC0"] == cooling_id, "pHVACz0"] = 1
            else:
                available = df[df["pHVACz0"] != 1].index
                if len(available) > 0:
                    df.iloc[available[0], df.columns.get_loc("pHVAC0")] = cooling_id
                    df.iloc[available[0], df.columns.get_loc("pHVACz0")] = 1
                else:
                    df.iloc[0, df.columns.get_loc("pHVAC0")] = cooling_id
                    df.iloc[0, df.columns.get_loc("pHVACz0")] = 1

    # --- WH logic ---
    df["pWHz0"] = 0
    df["pWHyn"] = "NO"
    if wh_id in df["pWH0"].values:
        df.loc[df["pWH0"] == wh_id, "pWHz0"] = 1
    else:
        df.iloc[0, df.columns.get_loc("pWH0")] = wh_id
        df.iloc[0, df.columns.get_loc("pWHz0")] = 1

    # Re-append trailing zeros row
    df = pd.concat([df, zeros_row], ignore_index=True)
    df.to_csv(output_dir / "sp.csv", index=False)
    return df


def create_wh_file(output_dir: Path, template: pd.DataFrame) -> None:
    """Copy wh.csv template as-is into the building output directory."""
    template.to_csv(output_dir / "wh.csv", index=False)


def _build_chp_df(template: pd.DataFrame, bldg_id: int,
                  hp_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-building chp DataFrame without saving to disk."""
    df = template.copy()
    row = hp_summary_df[hp_summary_df["bldg_id"] == bldg_id]
    if len(row) == 0:
        raise ValueError(f"Building {bldg_id} not found in heat pump summary file")
    heating_capacity_kw = float(row["heating_capacity_kw"].iloc[0])
    zeros_mask = df["ty"].astype(str).isin(["0", "0.0"])
    zeros_row  = df[zeros_mask].copy()
    df         = df[~zeros_mask].copy()
    df["mx"] = heating_capacity_kw
    return pd.concat([df, zeros_row], ignore_index=True)


def _build_hvac_df(template: pd.DataFrame, bldg_id: int,
                   hp_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-building hvac DataFrame without saving to disk."""
    df = template.copy()
    row = hp_summary_df[hp_summary_df["bldg_id"] == bldg_id]
    if len(row) == 0:
        raise ValueError(f"Building {bldg_id} not found in heat pump summary file")
    count      = float(row["count"].iloc[0])
    heating_kw = float(row["heating_capacity_kw"].iloc[0]) * count
    cooling_kw = float(row["cooling_capacity_kw"].iloc[0]) * count
    zeros_mask = df["ty"].astype(str).isin(["0", "0.0"])
    zeros_row  = df[zeros_mask].copy()
    df         = df[~zeros_mask].copy()
    df.loc[df["pHVmx"] != 0, "pHVmx"] = heating_kw
    df.loc[df["pACmx"] != 0, "pACmx"] = cooling_kw
    hp_index = str(row["hp_index"].iloc[0])
    mitsubishi_row = {
        "ty":      f"Mitsubishi ASHP {hp_index}",
        "pHVmx":   heating_kw,
        "pACmx":   cooling_kw,
        "pHVeff":  float(row["cop_heating"].iloc[0]),
        "pACeff":  float(row["cop_cooling"].iloc[0]),
        "temp":    float(row["design_temperature_c"].iloc[0]),
        "pHVmx_":  float(row["heating_capacity_loss"].iloc[0]) * count,
        "pACmx_":  float(row["cooling_capacity_loss"].iloc[0]) * count,
        "pHVeff_": float(row["cop_loss_heating"].iloc[0]),
        "pACeff_": float(row["cop_loss_cooling"].iloc[0]),
        "inv":     float(row["capital_cost"].iloc[0]),
        "fom":     0,
        "vom":     0,
        "life":    15,
    }
    return pd.concat([df, pd.DataFrame([mitsubishi_row]), zeros_row], ignore_index=True)


def create_hvac_file(output_dir: Path, template: pd.DataFrame,
                     bldg_id: int, hp_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create hvac.csv from template, updating capacities and adding Mitsubishi row."""
    df = _build_hvac_df(template, bldg_id, hp_summary_df)
    df.to_csv(output_dir / "hvac.csv", index=False)
    return df


def create_topo_file(output_dir: Path, template: pd.DataFrame) -> None:
    """Copy topo.csv template as-is into the building output directory."""
    template.to_csv(output_dir / "topo.csv", index=False)


def create_chp_file(output_dir: Path, template: pd.DataFrame,
                    bldg_id: int, hp_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create chp.csv from template with mx column set to building heating_capacity_kw."""
    df = _build_chp_df(template, bldg_id, hp_summary_df)
    df.to_csv(output_dir / "chp.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Per-building orchestrator
# ---------------------------------------------------------------------------

def process_building(bldg_id: int, parent_path: Path, metadata_df: pd.DataFrame,
                     regression_coeffs: dict, templates: dict,
                     elevation_cache: dict, hp_summary_df: pd.DataFrame) -> dict:
    """
    Load all building-specific data once, then generate and save all output files.
    This is the single entry point per building — no file I/O happens outside here.
    """
    t0     = time.perf_counter()
    result = {"bldg_id": bldg_id, "success": False, "elapsed": 0.0, "error": None}

    try:
        # --- Load per-building data ---
        building_row   = get_building_row(bldg_id, metadata_df)
        consumption_df = load_consumption(bldg_id, parent_path, YEAR)
        setpoint_df    = load_setpoints(bldg_id, parent_path, YEAR)

        # --- Create output directory ---
        output_dir = (parent_path / "in" / "decarb_inputs" / CASE_NAME
                      / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Generate and save each file ---
        if RUN_TM:
            tm_df = generate_tm_data(consumption_df, setpoint_df, building_row)
            create_tm_file(tm_df, output_dir)

        if RUN_BDG_II:
            create_bdg_ii_file(output_dir, bldg_id, regression_coeffs)

        if RUN_IN:
            create_in_file(output_dir, templates["in"], consumption_df)

        if RUN_OTHER:
            create_other_files(output_dir, templates)

        # Always build modified chp/hvac dfs (needed for sp.csv)
        chp_df  = create_chp_file(output_dir, templates["chp"], bldg_id, hp_summary_df) if RUN_CHP else _build_chp_df(templates["chp"], bldg_id, hp_summary_df)
        hvac_df = create_hvac_file(output_dir, templates["hvac"], bldg_id, hp_summary_df) if RUN_HVAC else _build_hvac_df(templates["hvac"], bldg_id, hp_summary_df)

        if RUN_TOPO:
            create_topo_file(output_dir, templates["topo"])

        if RUN_WH:
            create_wh_file(output_dir, templates["wh"])

        # Generate sp first so we know how many HVAC units are selected
        sp_df = None
        if RUN_SP:
            sp_df = create_sp_file(
                output_dir, templates["sp"], building_row,
                hvac_df, chp_df, templates["wh"],
            )

        # bdg_i depends on sp to set pBhvac correctly
        if RUN_BDG_I:
            pBhvac = int((sp_df["pHVACz0"] == 1).sum()) if sp_df is not None else 1
            pBhvac = max(pBhvac, 1)
            create_bdg_i_file(output_dir, templates["bdg_i"],
                              building_row, elevation_cache, pBhvac=pBhvac)

        result["success"] = True

    except Exception:
        result["error"] = traceback.format_exc()

    result["elapsed"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def load_building_ids(map_file: Path) -> list[int]:
    try:
        df = pd.read_parquet(map_file, engine="fastparquet")
    except Exception:
        df = pd.read_parquet(map_file, engine="pyarrow")

    if "bldg_id" not in df.columns:
        raise ValueError(
            f"'bldg_id' column not found in {map_file}. "
            f"Available columns: {list(df.columns)}"
        )
    return sorted(df["bldg_id"].dropna().astype(int).unique().tolist())


def output_exists(bldg_id: int, parent_path: Path) -> bool:
    """Return True if ALL enabled output files already exist for this building."""
    output_dir = (parent_path / "in" / "decarb_inputs" / CASE_NAME
                  / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
    checks = []
    if RUN_TM:
        checks.append((output_dir / "tm.csv").exists())
    if RUN_BDG_I:
        checks.append((output_dir / "bdg_i.csv").exists())
    if RUN_BDG_II:
        checks.append((output_dir / "bdg_ii.csv").exists())
    if RUN_IN:
        checks.append((output_dir / "in.csv").exists())
    if RUN_OTHER:
        checks.extend((output_dir / f).exists() for f in OTHER_TEMPLATE_FILES)
    if RUN_CHP:
        checks.append((output_dir / "chp.csv").exists())
    if RUN_TOPO:
        checks.append((output_dir / "topo.csv").exists())
    if RUN_HVAC:
        checks.append((output_dir / "hvac.csv").exists())
    if RUN_WH:
        checks.append((output_dir / "wh.csv").exists())
    if RUN_SP:
        checks.append((output_dir / "sp.csv").exists())
    return all(checks) if checks else False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parent_path = Path(__file__).parents[1]
    map_file    = parent_path / "out" / "ercot_substation_nrel_map.parquet"
    n_workers   = os.cpu_count()

    # Determine building list
    if BLDG_IDS_OVERRIDE is not None:
        bldg_ids = sorted(BLDG_IDS_OVERRIDE)
    else:
        bldg_ids = load_building_ids(map_file)

    if RETRY_FAILED:
        log_path = parent_path / "in" / "decarb_inputs" / CASE_NAME / "batch_failures.log"
        if not log_path.exists():
            print(f"RETRY_FAILED is set but no log found at {log_path}")
            raise SystemExit(1)
        import re
        failed_ids = set(int(x) for x in re.findall(r"bldg_id=(\d+)", log_path.read_text()))
        bldg_ids = [b for b in bldg_ids if b in failed_ids]
        print(f"Retry mode: {len(bldg_ids)} buildings from failure log")

    if DEBUG:
        import random
        rng = random.Random(DEBUG_SEED)
        bldg_ids = rng.sample(bldg_ids, min(DEBUG_N, len(bldg_ids)))
        print(f"Debug mode: running {len(bldg_ids)} randomly sampled buildings (seed={DEBUG_SEED})")

    if RESUME:
        before   = len(bldg_ids)
        bldg_ids = [b for b in bldg_ids if not output_exists(b, parent_path)]
        skipped  = before - len(bldg_ids)
        if skipped:
            print(f"Resume: skipping {skipped} completed, {len(bldg_ids)} remaining")

    if not bldg_ids:
        print("Nothing to do — all buildings already have output files.")
        raise SystemExit(0)

    # --- Load all shared resources once ---

    # Regression coefficients
    coeff_file = parent_path / "out" / "thermal_model" / "baseline" / "regression_coeff.json"
    with open(coeff_file, "r") as f:
        regression_coeffs = json.load(f)

    # Metadata
    meta_parquet = parent_path / "in" / "TX_upgrade0.parquet"
    meta_csv     = parent_path / "in" / "TX_upgrade0.csv"
    if meta_parquet.exists():
        try:
            metadata_df = pd.read_parquet(meta_parquet, engine="fastparquet")
        except Exception:
            metadata_df = pd.read_parquet(meta_parquet, engine="pyarrow")
    else:
        metadata_df = pd.read_csv(meta_csv)

    # Templates
    templates = load_templates(parent_path)

    # Heat pump summary
    hp_summary_df = pd.read_csv(
        parent_path / "out" / "heatpump_files" / "building_hp_summary.csv")

    # Elevation cache (only needed for bdg_i)
    if RUN_BDG_I:
        lat_col = "in.weather_file_latitude"
        lon_col = "in.weather_file_longitude"
        id_col  = "bldg_id" if "bldg_id" in metadata_df.columns else "building_id"
        subset  = metadata_df[metadata_df[id_col].isin(bldg_ids)]
        unique_coords = list(
            subset[[lat_col, lon_col]].drop_duplicates().itertuples(index=False, name=None)
        )
        elevation_cache = build_elevation_cache(unique_coords, max_workers=20)
    else:
        elevation_cache = {}

    print(f"Starting batch: {len(bldg_ids)} buildings  |  case: {CASE_NAME}  |  workers: {n_workers}")

    results    = []
    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_building,
                bldg_id, parent_path, metadata_df,
                regression_coeffs, templates, elevation_cache, hp_summary_df
            ): bldg_id
            for bldg_id in bldg_ids
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            elapsed = time.perf_counter() - wall_start
            status  = "OK" if result["success"] else "FAILED"
            print(f"[{len(results)}/{len(bldg_ids)}] bldg {result['bldg_id']:>6}  "
                  f"{status}  {elapsed:.0f}s elapsed")
            if not result["success"]:
                print(f"  ERROR: {result['error'].splitlines()[-1]}")

    wall_elapsed = time.perf_counter() - wall_start
    succeeded    = [r for r in results if r["success"]]
    failed       = [r for r in results if not r["success"]]

    print(f"Done: {len(succeeded)}/{len(results)} succeeded in {wall_elapsed:.0f}s")

    if failed:
        log_path = parent_path / "in" / "decarb_inputs" / CASE_NAME / "batch_failures.log"
        with open(log_path, "w") as f:
            f.write(f"Batch run {datetime.now().isoformat()}\n\n")
            for r in failed:
                f.write(f"=== bldg_id={r['bldg_id']} ===\n{r['error']}\n\n")
        print(f"Failure log: {log_path}")