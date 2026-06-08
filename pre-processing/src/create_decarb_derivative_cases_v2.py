#!/usr/bin/env python3
"""
Derivative Case Generator for Decarb Inputs

Two-phase pipeline:
  Phase 1: Build elec_100_flat from elec_0_flat
           → electrify sp.csv for all fossil buildings, copy everything else
  Phase 2: Build tariff variants (e.g. elec_100_tou) from elec_100_flat
           → copy all files, only modify tariff columns in tm.csv

v2: supports multiple TOU peak periods (morning + evening).

Edit the CONFIG block below, then run:
    python create_decarb_derivative_cases_v2.py
"""

import math
import os
import shutil
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd


# ===========================================================================
# CONFIG — edit these before running
# ===========================================================================

BASE_CASE            = "elec_0_flat"       # input case folder name
BASE_RESULTS         = "elec_0_flat"       # results folder name (for TOU calculation)
INPUT_FOLDER         = "decarb_inputs_alt" # folder under in/ containing case folders
UPDATE_NUM           = 0

# Tariff configurations — add/edit entries here
# Each key becomes part of the output folder name: elec_100_{key}
# Supported fields:
#   type:       "flat" or "tou"
#   rate:       flat rate $/kWh (flat only)
#   peak_price, off_peak, peak_start, peak_end (tou only)
#   capacity_charge: True/False
#   cap_charge_value, cap_peak_start, cap_peak_end (if capacity_charge)
FLAT_RATE            = 0.22        # $/kWh for the base flat tariff

# TOU peak periods (exclusive start, inclusive end per end-of-period convention)
# ("06:00","09:00") -> timestamps 07,08,09 -> real hours 6-7, 7-8, 8-9am
PEAK_PERIODS = [
    ("05:00", "09:00"),   # morning: 6am-10am (4 hrs)
    ("18:00", "22:00"),   # evening: 6pm-10pm (4 hrs)
]

TARIFF_CONFIGS = {
    "tou_v2": {
        "type":         "tou",
        "peak_periods": PEAK_PERIODS,
        "peak_price":   None,   # computed at runtime (revenue-neutral, on_peak = 2x off_peak)
        "off_peak":     None,
    },
    # "tou_cap": {
    #     "type":         "tou",
    #     "peak_periods": PEAK_PERIODS,
    #     "peak_price":   None,   # computed at runtime
    #     "off_peak":     None,
    #     "capacity_charge": True,
    #     # cap_charge_value, cap_peak_start, cap_peak_end: set after user defines method
    # },
}

SEED = 42

# Set to a list of ints to run only specific buildings, e.g. [77, 250, 312]
BLDG_IDS_OVERRIDE = None

# Debug mode — runs on a random sample of buildings
DEBUG     = False
DEBUG_N   = 10

# Set to True to skip buildings/cases that already have all output files
RESUME    = False

# Set to True to skip Phase 1 (elec_100_flat) and only run tariff variants
SKIP_PHASE1 = True

# ===========================================================================

# All input files per building
ALL_FILES = [
    "abs.csv", "bdg_i.csv", "bdg_ii.csv", "bess.csv", "chp.csv",
    "ev.csv", "hvac.csv", "in.csv", "pv.csv", "sp.csv", "tm.csv",
    "topo.csv", "wh.csv", "wind.csv",
]


# ---------------------------------------------------------------------------
# TOU rate calculation (revenue-neutral)
# ---------------------------------------------------------------------------

def _build_peak_mask(times, peak_periods):
    """Build a boolean mask for peak hours from a list of (start, end) period strings."""
    import numpy as np
    mask = np.zeros(len(times), dtype=bool)
    for start_str, end_str in peak_periods:
        # Exclusive start, inclusive end: ("16:00","19:00") -> timestamps 17,18,19
        ps = datetime.strptime(start_str, "%H:%M").time()
        pe = datetime.strptime(end_str,   "%H:%M").time()
        if ps < pe:
            mask |= (times > ps) & (times <= pe)
        else:
            mask |= (times > ps) | (times <= pe)
    return mask


def calculate_tou_rates(parent_path: Path, base_case: str,
                        flat_rate: float,
                        peak_periods: list) -> tuple[float, float, dict]:
    """
    Compute revenue-neutral TOU rates from the base case aggregated profile.

    Constraint: on_peak = 2 x off_peak, and total revenue equals flat case.
    Supports multiple peak periods.

    Returns (off_peak, on_peak, summary).
    """
    # Load aggregated hourly grid purchases
    profile_path = (parent_path / "out" / "decarb_results" / base_case
                    / f"{base_case}_aggregated_ts.csv")
    profile_df = pd.read_csv(profile_path)
    grid_col = "Grid Purchases [GW]"

    if grid_col not in profile_df.columns:
        raise ValueError(f"Column '{grid_col}' not found in {profile_path}")
    if len(profile_df) != 8760:
        raise ValueError(f"Expected 8760 rows, got {len(profile_df)} in {profile_path}")

    # Assign hourly timestamps for 2018
    timestamps = pd.date_range(start="2018-01-01", periods=8760, freq="h")
    profile_df.index = timestamps

    # Identify peak hours (union of all peak periods)
    is_peak = _build_peak_mask(profile_df.index.time, peak_periods)

    e_total   = profile_df[grid_col].sum()
    e_peak    = profile_df.loc[is_peak, grid_col].sum()
    e_offpeak = e_total - e_peak

    # Revenue requirement from flat tariff
    rr = flat_rate * e_total

    # Solve: off_peak * E_offpeak + 2 * off_peak * E_peak = RR
    off_peak_exact = rr / (e_offpeak + 2 * e_peak)

    # Round up to nearest $0.01
    off_peak = math.ceil(off_peak_exact * 100) / 100
    on_peak  = round(2 * off_peak, 2)

    # Revenue under the rounded TOU rates (slightly higher due to rounding up)
    # Note: energy is in GWh, rates in $/kWh -> multiply by 1e6 for actual $
    rr_dollars     = rr * 1e6
    rr_tou_dollars = (off_peak * e_offpeak + on_peak * e_peak) * 1e6
    rr_diff_pct    = (rr_tou_dollars - rr_dollars) / rr_dollars * 100

    peak_label = ", ".join(f"{s}-{e}" for s, e in peak_periods)
    peak_hrs_per_day = sum(is_peak) / 365

    summary = {
        "E_total_GWh":          round(e_total, 2),
        "E_peak_GWh":           round(e_peak, 2),
        "E_offpeak_GWh":        round(e_offpeak, 2),
        "peak_share_pct":       round(e_peak / e_total * 100, 2),
        "peak_periods":         peak_label,
        "peak_hrs_per_day":     round(peak_hrs_per_day, 1),
        "flat_rate":            flat_rate,
        "revenue_flat":         round(rr_dollars, 2),
        "off_peak_exact":       round(off_peak_exact, 6),
        "off_peak_rounded":     off_peak,
        "on_peak":              on_peak,
        "revenue_tou":          round(rr_tou_dollars, 2),
        "revenue_diff_pct":     round(rr_diff_pct, 4),
    }

    print(f"TOU rate calculation:")
    print(f"  Peak periods  = {peak_label}  ({peak_hrs_per_day:.0f} hrs/day)")
    print(f"  E_total       = {e_total:,.2f} GWh")
    print(f"  E_peak        = {e_peak:,.2f} GWh")
    print(f"  E_offpeak     = {e_offpeak:,.2f} GWh")
    print(f"  Peak share    = {e_peak / e_total * 100:.2f}%")
    print(f"  Flat rate     = ${flat_rate}/kWh")
    print(f"  Revenue req.  = ${rr_dollars:,.0f}")
    print(f"  off_peak      = ${off_peak_exact:.6f} -> ${off_peak:.2f}/kWh (rounded up)")
    print(f"  on_peak       = ${on_peak:.2f}/kWh")
    print(f"  Revenue (TOU) = ${rr_tou_dollars:,.0f}  ({rr_diff_pct:+.4f}% vs flat)")

    return off_peak, on_peak, summary


# ---------------------------------------------------------------------------
# Capacity charge calculation (1-part demand charge)
# ---------------------------------------------------------------------------

def calculate_capacity_charge(parent_path: Path, base_case: str,
                              flat_rate: float,
                              peak_periods: list) -> dict:
    """
    Compute a 1-part demand charge based on the flat case aggregate profile.

    Approach:
      RevReq_capacity = E_peak [GWh] * flat_rate [$/kWh] * 1e6
          (i.e. flat-case revenue collected during peak hours)
      system_peak_kW  = max hourly Grid Purchases over all hours * 1e6
      n_consumers     = sum of 'count' in building_mapping.csv
      per_consumer_peak_kW = system_peak_kW / n_consumers
      DC [$/kW]       = RevReq_capacity / system_peak_kW

    Each building would pay DC * their peak demand (kW).
    If all buildings peaked together, total revenue = DC * system_peak = RevReq_capacity.

    Returns a summary dict.
    """
    # --- System peak from aggregated profile ---
    profile_path = (parent_path / "out" / "decarb_results" / base_case
                    / f"{base_case}_aggregated_ts.csv")
    profile_df = pd.read_csv(profile_path)
    grid_col = "Grid Purchases [GW]"

    timestamps = pd.date_range(start="2018-01-01", periods=8760, freq="h")
    profile_df.index = timestamps

    # Energy during peak hours (for revenue requirement)
    is_peak = _build_peak_mask(profile_df.index.time, peak_periods)
    e_peak_gwh = profile_df.loc[is_peak, grid_col].sum()        # GWh
    e_total_gwh = profile_df[grid_col].sum()

    # System peak [GW] -> [kW]
    system_peak_gw = profile_df[grid_col].max()
    system_peak_kw = system_peak_gw * 1e6

    # System peak restricted to the peak-period hours
    peak_period_peak_gw = profile_df.loc[is_peak, grid_col].max()
    peak_period_peak_kw = peak_period_peak_gw * 1e6

    # --- Consumer count from building mapping ---
    mapping_path = (parent_path / "out" / "decarb_results" / base_case
                    / "building_mapping.csv")
    mapping_df = pd.read_csv(mapping_path)
    n_consumers = int(mapping_df["count"].sum())
    n_sim_bldgs = len(mapping_df)

    # --- Revenue requirement for capacity charges ---
    rev_req_capacity = e_peak_gwh * 1e6 * flat_rate        # $

    # --- Demand charge ---
    # DC = RevReq / (sum of building peaks ~ system peak)
    dc_per_kw_exact = rev_req_capacity / system_peak_kw           # $/kW (raw)
    dc_per_kw_pp_exact = rev_req_capacity / peak_period_peak_kw   # $/kW (peak-period)

    # Round UP to nearest $10/kW
    dc_per_kw    = math.ceil(dc_per_kw_exact    / 10) * 10
    dc_per_kw_pp = math.ceil(dc_per_kw_pp_exact / 10) * 10

    # Each consumer's assumed peak (equal distribution of system peak)
    per_consumer_peak_kw = system_peak_kw / n_consumers

    # Each consumer pays DC * their peak; if they all peak at once:
    charge_per_consumer = dc_per_kw * per_consumer_peak_kw

    peak_label = ", ".join(f"{s}-{e}" for s, e in peak_periods)
    peak_hrs_per_day = sum(is_peak) / 365

    print(f"\nCapacity charge calculation (1-part DC):")
    print(f"  Peak periods        = {peak_label}  ({peak_hrs_per_day:.0f} hrs/day)")
    print(f"  E_total             = {e_total_gwh:,.2f} GWh")
    print(f"  E_peak              = {e_peak_gwh:,.2f} GWh")
    print(f"  Flat rate           = ${flat_rate}/kWh")
    print(f"  RevReq_capacity     = E_peak * flat_rate = ${rev_req_capacity:,.0f}")
    print(f"  System peak (year)  = {system_peak_gw:.4f} GW  =  {system_peak_kw:,.0f} kW")
    print(f"  System peak (peak)  = {peak_period_peak_gw:.4f} GW  =  {peak_period_peak_kw:,.0f} kW")
    print(f"  n_sim_buildings     = {n_sim_bldgs:,}")
    print(f"  n_consumers (real)  = {n_consumers:,}")
    print(f"  Per-consumer peak   = {per_consumer_peak_kw:.4f} kW")
    print(f"  DC (vs yearly peak) = ${dc_per_kw_exact:.4f} -> ${dc_per_kw:.0f}/kW (rounded up to nearest $10)")
    print(f"  DC (vs peak-period) = ${dc_per_kw_pp_exact:.4f} -> ${dc_per_kw_pp:.0f}/kW (rounded up to nearest $10)")
    print(f"  Charge per consumer = ${charge_per_consumer:,.2f}  (if all peak together)")

    return {
        "peak_periods":         peak_label,
        "E_peak_GWh":           round(e_peak_gwh, 2),
        "flat_rate":            flat_rate,
        "rev_req_capacity":     round(rev_req_capacity, 2),
        "system_peak_kW":       round(system_peak_kw, 2),
        "peak_period_peak_kW":  round(peak_period_peak_kw, 2),
        "n_consumers":          n_consumers,
        "n_sim_buildings":      n_sim_bldgs,
        "per_consumer_peak_kW": round(per_consumer_peak_kw, 4),
        "DC_per_kW":            round(dc_per_kw, 4),
        "DC_per_kW_peak_period":round(dc_per_kw_pp, 4),
    }


# ---------------------------------------------------------------------------
# Tariff modification
# ---------------------------------------------------------------------------

def modify_tm(tm_df: pd.DataFrame, tariff_cfg: dict) -> pd.DataFrame:
    """Return a copy of tm_df with tariff columns overwritten per tariff_cfg."""
    df = tm_df.copy()

    if tariff_cfg["type"] == "flat":
        df["pQcostBuy"] = tariff_cfg["rate"]
    elif tariff_cfg["type"] == "tou":
        peak_price = tariff_cfg["peak_price"]
        off_peak   = tariff_cfg["off_peak"]

        times   = pd.to_datetime(df.index).time
        is_peak = _build_peak_mask(times, tariff_cfg["peak_periods"])

        df["pQcostBuy"] = off_peak
        df.loc[is_peak, "pQcostBuy"] = peak_price

    # Capacity charge
    if tariff_cfg.get("capacity_charge"):
        cap_start = datetime.strptime(tariff_cfg["cap_peak_start"], "%H:%M").time()
        cap_end   = datetime.strptime(tariff_cfg["cap_peak_end"],   "%H:%M").time()
        times     = pd.to_datetime(df.index).time
        if cap_start < cap_end:
            is_cap = (times >= cap_start) & (times < cap_end)
        else:
            is_cap = (times >= cap_start) | (times < cap_end)
        df["pQmx"]     = 0
        df.loc[is_cap, "pQmx"] = 1
        df["pQmxCost"]  = 0.0
        df.loc[is_cap, "pQmxCost"] = tariff_cfg["cap_charge_value"]
    else:
        df["pQmx"]     = 0
        df["pQmxCost"] = 0.0

    return df


# ---------------------------------------------------------------------------
# SP electrification
# ---------------------------------------------------------------------------

def electrify_sp(sp_df: pd.DataFrame, hvac_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a modified sp_df where:
    - All CHP units are deactivated (pCHPz0=0)
    - All HVAC units are deactivated (pHVACz0=0)
    - The Mitsubishi ASHP row is activated in HVAC (pHVACz0=1, pHVACyn=YES)
    """
    df = sp_df.copy()

    # Find the Mitsubishi ASHP identifier from this building's hvac.csv
    mitsubishi_rows = hvac_df[hvac_df["ty"].astype(str).str.startswith("Mitsubishi ASHP")]
    if mitsubishi_rows.empty:
        # No Mitsubishi row — skip electrification for this building
        return df

    mitsubishi_id = str(mitsubishi_rows["ty"].iloc[0])

    # Separate trailing zeros row
    zeros_mask = df["pCHP0"].astype(str).isin(["0", "0.0"])
    zeros_row  = df[zeros_mask].copy()
    df         = df[~zeros_mask].copy()

    # Deactivate all CHP and HVAC
    df["pCHPz0"]  = 0
    df["pCHPyn"]  = "NO"
    df["pHVACz0"] = 0
    df["pHVACyn"] = "NO"

    # Activate Mitsubishi ASHP in HVAC
    if mitsubishi_id in df["pHVAC0"].values:
        df.loc[df["pHVAC0"] == mitsubishi_id, "pHVACz0"] = 1
        df.loc[df["pHVAC0"] == mitsubishi_id, "pHVACyn"] = "YES"
    else:
        # Place it in the first row
        df.iloc[0, df.columns.get_loc("pHVAC0")]  = mitsubishi_id
        df.iloc[0, df.columns.get_loc("pHVACz0")] = 1
        df.iloc[0, df.columns.get_loc("pHVACyn")] = "YES"

    return pd.concat([df, zeros_row], ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 1: Electrify a building (elec_0_flat → elec_100_flat)
# ---------------------------------------------------------------------------

def process_electrify(bldg_id: int, parent_path: Path,
                      electrify: bool) -> dict:
    """
    Copy all files from base case to elec_100_flat.
    If electrify=True, rewrite sp.csv with Mitsubishi ASHP activated.
    """
    case_name = "elec_100_flat"
    t0     = time.perf_counter()
    result = {"bldg_id": bldg_id, "case": case_name, "success": False,
              "elapsed": 0.0, "error": None}

    try:
        base_dir   = (parent_path / "in" / INPUT_FOLDER / BASE_CASE
                      / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
        output_dir = (parent_path / "in" / INPUT_FOLDER / case_name
                      / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from base case
        for fname in ALL_FILES:
            src = base_dir / fname
            if src.exists():
                shutil.copy2(src, output_dir / fname)

        # Overwrite sp.csv with electrified version
        if electrify:
            sp_df   = pd.read_csv(base_dir / "sp.csv")
            hvac_df = pd.read_csv(base_dir / "hvac.csv")
            sp_modified = electrify_sp(sp_df, hvac_df)
            sp_modified.to_csv(output_dir / "sp.csv", index=False)

        result["success"] = True

    except Exception:
        result["error"] = traceback.format_exc()

    result["elapsed"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Tariff variant: copy from a source case, only modify tm.csv
# ---------------------------------------------------------------------------

def process_tariff(bldg_id: int, parent_path: Path,
                   source_case: str, case_name: str, tariff_cfg: dict) -> dict:
    """
    Copy all files from source_case, only modify tariff columns in tm.csv.
    """
    t0     = time.perf_counter()
    result = {"bldg_id": bldg_id, "case": case_name, "success": False,
              "elapsed": 0.0, "error": None}

    try:
        source_dir = (parent_path / "in" / INPUT_FOLDER / source_case
                      / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
        output_dir = (parent_path / "in" / INPUT_FOLDER / case_name
                      / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from source case
        for fname in ALL_FILES:
            src = source_dir / fname
            if src.exists():
                shutil.copy2(src, output_dir / fname)

        # Overwrite tm.csv with modified tariff
        tm_df = pd.read_csv(source_dir / "tm.csv", index_col=0)
        tm_modified = modify_tm(tm_df, tariff_cfg)
        tm_modified.to_csv(output_dir / "tm.csv")

        result["success"] = True

    except Exception:
        result["error"] = traceback.format_exc()

    result["elapsed"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parent_path = Path(__file__).parents[1]
    n_workers   = os.cpu_count()

    # --- Discover building IDs from base case ---
    base_case_dir = parent_path / "in" / INPUT_FOLDER / BASE_CASE
    if BLDG_IDS_OVERRIDE is not None:
        bldg_ids = sorted(BLDG_IDS_OVERRIDE)
    else:
        bldg_ids = sorted(
            int(d.name) for d in base_case_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

    if DEBUG:
        # Debug mode: compute and print TOU rates + capacity charge, then exit
        for key, cfg in TARIFF_CONFIGS.items():
            if cfg["type"] == "tou" and cfg.get("peak_price") is None:
                print(f"\n=== Tariff: {key} ===")
                off_peak, on_peak, summary = calculate_tou_rates(
                    parent_path, BASE_RESULTS, FLAT_RATE,
                    cfg["peak_periods"],
                )
                if cfg.get("capacity_charge"):
                    calculate_capacity_charge(
                        parent_path, BASE_RESULTS, FLAT_RATE,
                        cfg["peak_periods"],
                    )
        raise SystemExit(0)

    # --- Load metadata to classify buildings ---
    meta_parquet = parent_path / "in" / "TX_upgrade0.parquet"
    meta_csv     = parent_path / "in" / "TX_upgrade0.csv"
    if meta_parquet.exists():
        try:
            metadata_df = pd.read_parquet(meta_parquet, engine="fastparquet")
        except Exception:
            metadata_df = pd.read_parquet(meta_parquet, engine="pyarrow")
    else:
        metadata_df = pd.read_csv(meta_csv)

    id_col   = "bldg_id" if "bldg_id" in metadata_df.columns else "building_id"
    fuel_col = "in.heating_fuel"

    # Classify fossil vs electric buildings
    meta_subset = metadata_df[metadata_df[id_col].isin(bldg_ids)][[id_col, fuel_col]].copy()
    meta_subset = meta_subset.drop_duplicates(subset=id_col)

    fossil_fuels = {"Natural Gas", "Propane", "Fuel Oil"}
    fossil_bldg_ids = sorted(
        meta_subset[meta_subset[fuel_col].isin(fossil_fuels)][id_col].tolist()
    )
    electric_bldg_ids = sorted(
        set(bldg_ids) - set(fossil_bldg_ids)
    )

    # All fossil buildings get electrified (100%)
    electrify_set = set(fossil_bldg_ids)

    print(f"Buildings: {len(bldg_ids)} total, "
          f"{len(fossil_bldg_ids)} fossil (all electrified), "
          f"{len(electric_bldg_ids)} already electric")

    # --- Compute TOU rates from base case profile ---
    for key, cfg in TARIFF_CONFIGS.items():
        if cfg["type"] == "tou" and cfg.get("peak_price") is None:
            print(f"\n=== Tariff: {key} ===")
            off_peak, on_peak, summary = calculate_tou_rates(
                parent_path, BASE_RESULTS, FLAT_RATE,
                cfg["peak_periods"],
            )
            cfg["off_peak"]   = off_peak
            cfg["peak_price"] = on_peak

            if cfg.get("capacity_charge"):
                cap_summary = calculate_capacity_charge(
                    parent_path, BASE_RESULTS, FLAT_RATE,
                    cfg["peak_periods"],
                )
                cfg["cap_charge_value"] = cap_summary["DC_per_kW"]

    # ===================================================================
    # Phase 1: elec_100_flat (electrify sp.csv, keep flat tariff)
    # ===================================================================
    results    = []
    all_failed = []

    if SKIP_PHASE1:
        print("\n--- Phase 1: SKIPPED (SKIP_PHASE1 = True) ---")
    else:
        phase1_tasks = []
        for bldg_id in bldg_ids:
            if RESUME:
                out_dir = (parent_path / "in" / INPUT_FOLDER / "elec_100_flat"
                           / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
                if all((out_dir / f).exists() for f in ALL_FILES):
                    continue
            phase1_tasks.append((bldg_id, parent_path, bldg_id in electrify_set))

        print(f"\n--- Phase 1: elec_100_flat ---")
        print(f"  {len(phase1_tasks)} buildings to process")

        wall_start = time.perf_counter()

        if phase1_tasks:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(process_electrify, *task): task
                    for task in phase1_tasks
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if len(results) % 500 == 0 or not result["success"]:
                        elapsed = time.perf_counter() - wall_start
                        status  = "OK" if result["success"] else "FAILED"
                        print(f"  [{len(results)}/{len(phase1_tasks)}] bldg {result['bldg_id']:>6} "
                              f"{status}  {elapsed:.0f}s elapsed")
                        if not result["success"]:
                            print(f"    ERROR: {result['error'].splitlines()[-1]}")

        phase1_ok = sum(1 for r in results if r["success"])
        all_failed.extend(r for r in results if not r["success"])
        print(f"  Phase 1 done: {phase1_ok}/{len(results)} succeeded "
              f"in {time.perf_counter() - wall_start:.0f}s")

    # ===================================================================
    # Phase 2: tariff-only variants
    #   elec_0_tou   -> from elec_0_flat   (no electrification, TOU tariff)
    #   elec_100_tou -> from elec_100_flat  (electrified, TOU tariff)
    # ===================================================================
    tariff_variants = {k: v for k, v in TARIFF_CONFIGS.items() if k != "flat"}

    # Each tariff variant is applied to both the base case and the electrified case
    tariff_jobs = []
    for tariff_label, tariff_cfg in tariff_variants.items():
        tariff_jobs.append((BASE_CASE,       f"elec_0_{tariff_label}",   tariff_cfg))
        tariff_jobs.append(("elec_100_flat", f"elec_100_{tariff_label}", tariff_cfg))

    phase2_tasks = []
    for source_case, case_name, tariff_cfg in tariff_jobs:
        for bldg_id in bldg_ids:
            if RESUME:
                out_dir = (parent_path / "in" / INPUT_FOLDER / case_name
                           / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
                if all((out_dir / f).exists() for f in ALL_FILES):
                    continue
            phase2_tasks.append((bldg_id, parent_path, source_case, case_name, tariff_cfg))

    if phase2_tasks:
        print(f"\n--- Phase 2: tariff variants ---")
        case_names = sorted(set(t[3] for t in phase2_tasks))
        print(f"  Cases: {', '.join(case_names)}")
        print(f"  {len(phase2_tasks)} tasks total")

        phase2_results = []
        wall_start2    = time.perf_counter()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_tariff, *task): task
                for task in phase2_tasks
            }
            for future in as_completed(futures):
                result = future.result()
                phase2_results.append(result)
                if len(phase2_results) % 500 == 0 or not result["success"]:
                    elapsed = time.perf_counter() - wall_start2
                    status  = "OK" if result["success"] else "FAILED"
                    print(f"  [{len(phase2_results)}/{len(phase2_tasks)}] "
                          f"bldg {result['bldg_id']:>6} {result['case']}  "
                          f"{status}  {elapsed:.0f}s elapsed")
                    if not result["success"]:
                        print(f"    ERROR: {result['error'].splitlines()[-1]}")

        phase2_ok = sum(1 for r in phase2_results if r["success"])
        all_failed.extend(r for r in phase2_results if not r["success"])
        print(f"  Phase 2 done: {phase2_ok}/{len(phase2_results)} succeeded "
              f"in {time.perf_counter() - wall_start2:.0f}s")
        results.extend(phase2_results)

    # ===================================================================
    # Compress each case as .tar.gz (for Linux transfer)
    # ===================================================================
    decarb_dir = parent_path / "in" / INPUT_FOLDER
    all_cases  = [] if SKIP_PHASE1 else ["elec_100_flat"]
    all_cases += [f"elec_0_{t}" for t in tariff_variants]
    all_cases += [f"elec_100_{t}" for t in tariff_variants]

    print(f"\n--- Compressing cases ---")
    for case_name in all_cases:
        case_dir = decarb_dir / case_name
        if not case_dir.exists():
            continue
        tar_path = decarb_dir / f"{case_name}.tar.gz"
        print(f"  {case_name} -> {tar_path.name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(case_dir, arcname=case_name)
        print(f"done ({time.perf_counter() - t0:.0f}s)")

    # ===================================================================
    # Summary
    # ===================================================================
    succeeded = sum(1 for r in results if r["success"])
    print(f"\nTotal: {succeeded}/{len(results)} succeeded")

    if all_failed:
        log_path = parent_path / "in" / INPUT_FOLDER / "derivative_batch_failures.log"
        with open(log_path, "w") as f:
            f.write(f"Derivative batch run {datetime.now().isoformat()}\n\n")
            for r in all_failed:
                f.write(f"=== bldg_id={r['bldg_id']} case={r['case']} ===\n{r['error']}\n\n")
        print(f"Failure log: {log_path}")
