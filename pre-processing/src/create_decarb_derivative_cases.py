#!/usr/bin/env python3
"""
Derivative Case Generator for Decarb Inputs

Two-phase pipeline:
  Phase 1: Build elec_100_flat from elec_0_flat
           → electrify sp.csv for all fossil buildings, copy everything else
  Phase 2: Build tariff variants (e.g. elec_100_tou) from elec_100_flat
           → copy all files, only modify tariff columns in tm.csv

Edit the CONFIG block below, then run:
    python create_decarb_derivative_cases.py
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
BASE_RESULTS         = "elec_0_flat"           # results folder name (for TOU calculation)
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
PEAK_START           = "16:00"     # TOU peak period start
PEAK_END             = "19:00"     # TOU peak period end

TARIFF_CONFIGS = {
    "flat": {
        "type":       "flat",
        "rate":       FLAT_RATE,
    },
    "tou": {
        "type":       "tou",
        "peak_start": PEAK_START,
        "peak_end":   PEAK_END,
        # peak_price and off_peak are computed at runtime from the base case
        # aggregated profile (revenue-neutral: on_peak = 2 x off_peak)
        "peak_price": None,
        "off_peak":   None,
    },
}

SEED = 42

# Set to a list of ints to run only specific buildings, e.g. [77, 250, 312]
BLDG_IDS_OVERRIDE = None

# Debug mode — runs on a random sample of buildings
DEBUG     = False
DEBUG_N   = 10

# Set to True to skip buildings/cases that already have all output files
RESUME    = False

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

def calculate_tou_rates(parent_path: Path, base_case: str,
                        flat_rate: float,
                        peak_start_str: str, peak_end_str: str) -> tuple[float, float]:
    """
    Compute revenue-neutral TOU rates from the base case aggregated profile.

    Constraint: on_peak = 2 x off_peak, and total revenue equals flat case.

    Returns (off_peak, on_peak) in $/kWh.
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

    # Identify peak hours
    peak_start = datetime.strptime(peak_start_str, "%H:%M").time()
    peak_end   = datetime.strptime(peak_end_str,   "%H:%M").time()
    times = profile_df.index.time
    # Exclusive start, inclusive end to match decarb end-of-period convention
    if peak_start < peak_end:
        is_peak = (times > peak_start) & (times <= peak_end)
    else:
        is_peak = (times > peak_start) | (times <= peak_end)

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

    summary = {
        "E_total_GWh":          round(e_total, 2),
        "E_peak_GWh":           round(e_peak, 2),
        "E_offpeak_GWh":        round(e_offpeak, 2),
        "peak_share_pct":       round(e_peak / e_total * 100, 2),
        "peak_hours":           f"{peak_start_str}-{peak_end_str}",
        "flat_rate":            flat_rate,
        "revenue_flat":         round(rr_dollars, 2),
        "off_peak_exact":       round(off_peak_exact, 6),
        "off_peak_rounded":     off_peak,
        "on_peak":              on_peak,
        "revenue_tou":          round(rr_tou_dollars, 2),
        "revenue_diff_pct":     round(rr_diff_pct, 4),
    }

    print(f"TOU rate calculation:")
    print(f"  E_total       = {e_total:,.2f} GWh")
    print(f"  E_peak        = {e_peak:,.2f} GWh  ({peak_start_str}-{peak_end_str})")
    print(f"  E_offpeak     = {e_offpeak:,.2f} GWh")
    print(f"  Peak share    = {e_peak / e_total * 100:.2f}%")
    print(f"  Flat rate     = ${flat_rate}/kWh")
    print(f"  Revenue req.  = ${rr_dollars:,.0f}")
    print(f"  off_peak      = ${off_peak_exact:.6f} -> ${off_peak:.2f}/kWh (rounded up)")
    print(f"  on_peak       = ${on_peak:.2f}/kWh")
    print(f"  Revenue (TOU) = ${rr_tou_dollars:,.0f}  ({rr_diff_pct:+.4f}% vs flat)")

    return off_peak, on_peak, summary


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
        peak_start = datetime.strptime(tariff_cfg["peak_start"], "%H:%M").time()
        peak_end   = datetime.strptime(tariff_cfg["peak_end"],   "%H:%M").time()

        # decarb uses end-of-period timestamps: timestamp T = hour ending at T
        times = pd.to_datetime(df.index).time
        if peak_start < peak_end:
            is_peak = (times > peak_start) & (times <= peak_end)
        else:
            is_peak = (times > peak_start) | (times <= peak_end)

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
        base_dir   = (parent_path / "in" / "decarb_inputs" / BASE_CASE
                      / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
        output_dir = (parent_path / "in" / "decarb_inputs" / case_name
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
        source_dir = (parent_path / "in" / "decarb_inputs" / source_case
                      / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
        output_dir = (parent_path / "in" / "decarb_inputs" / case_name
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
    base_case_dir = parent_path / "in" / "decarb_inputs" / BASE_CASE
    if BLDG_IDS_OVERRIDE is not None:
        bldg_ids = sorted(BLDG_IDS_OVERRIDE)
    else:
        bldg_ids = sorted(
            int(d.name) for d in base_case_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

    if DEBUG:
        # Debug mode: just compute and print TOU rates, then exit
        if "tou" in TARIFF_CONFIGS and TARIFF_CONFIGS["tou"].get("peak_price") is None:
            tou_cfg = TARIFF_CONFIGS["tou"]
            off_peak, on_peak, summary = calculate_tou_rates(
                parent_path, BASE_RESULTS, FLAT_RATE,
                tou_cfg["peak_start"], tou_cfg["peak_end"],
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
    if "tou" in TARIFF_CONFIGS and TARIFF_CONFIGS["tou"].get("peak_price") is None:
        tou_cfg = TARIFF_CONFIGS["tou"]
        off_peak, on_peak, summary = calculate_tou_rates(
            parent_path, BASE_RESULTS, FLAT_RATE,
            tou_cfg["peak_start"], tou_cfg["peak_end"],
        )
        tou_cfg["off_peak"]   = off_peak
        tou_cfg["peak_price"] = on_peak

    # ===================================================================
    # Phase 1: elec_100_flat (electrify sp.csv, keep flat tariff)
    # ===================================================================
    phase1_tasks = []
    for bldg_id in bldg_ids:
        if RESUME:
            out_dir = (parent_path / "in" / "decarb_inputs" / "elec_100_flat"
                       / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
            if all((out_dir / f).exists() for f in ALL_FILES):
                continue
        phase1_tasks.append((bldg_id, parent_path, bldg_id in electrify_set))

    print(f"\n--- Phase 1: elec_100_flat ---")
    print(f"  {len(phase1_tasks)} buildings to process")

    results    = []
    all_failed = []
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
                out_dir = (parent_path / "in" / "decarb_inputs" / case_name
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
    decarb_dir = parent_path / "in" / "decarb_inputs"
    all_cases  = ["elec_100_flat"]
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
        log_path = parent_path / "in" / "decarb_inputs" / "derivative_batch_failures.log"
        with open(log_path, "w") as f:
            f.write(f"Derivative batch run {datetime.now().isoformat()}\n\n")
            for r in all_failed:
                f.write(f"=== bldg_id={r['bldg_id']} case={r['case']} ===\n{r['error']}\n\n")
        print(f"Failure log: {log_path}")
