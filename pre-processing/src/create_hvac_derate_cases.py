#!/usr/bin/env python3
"""
HVAC Derate Derivative Cases

For each base case in BASE_CASES, generate derivative input cases where the
HVAC capacity columns (pHVmx, pACmx) in hvac.csv are derated by DERATE_PCTS.

Test set: 100 buildings, randomly sampled from the pool of buildings that did
NOT fail in any of the 8 FILTER_CASES and that have ts.csv present in every
one of those 8 cases' out folders. The 100-building list is written to
hvac_derate_test_set.csv and reused on subsequent runs.

Sources only from <case>/<bldg_id>/update_0/in/  -- never copies out/.

Edit the CONFIG block below and run:
    python create_hvac_derate_cases.py
"""

import os
import random
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd


# ===========================================================================
# CONFIG
# ===========================================================================

INPUT_FOLDER = "decarb_inputs"
UPDATE_NUM   = 0
N_TEST       = 100
DERATE_PCTS  = [10, 20, 30, 40, 50]
INCLUDE_BASE = True   # also emit <base>_derate_base (no derate, same 100 bldgs)
SEED         = 42

# Cases used for filtering the test set (no failures + ts.csv must exist in all)
FILTER_CASES = [
    "elec_0_flat",      "elec_100_flat",
    "elec_0_flat_cap",  "elec_100_flat_cap",
    "elec_0_tou",    "elec_100_tou",
    "elec_0_tou_cap",   "elec_100_tou_cap",
]

# Base cases for which derivative folders are actually generated this run
BASE_CASES = ["elec_0_flat"]

DEBUG      = False
RESUME     = False
COMPRESS   = True

TEST_SET_FILE = "hvac_derate_test_set.csv"

# Buildings to exclude up-front (bad thermal-model fits, etc.)
BAD_BLDGS_FILE = "out/thermal_model/bad_resstock_bldgs_ihg.csv"

# Single combined archive name (when COMPRESS=True)
ARCHIVE_NAME = "elec_0_flat_derate_all.tar.gz"

# ===========================================================================

# All 14 input files per building
ALL_FILES = [
    "abs.csv", "bdg_i.csv", "bdg_ii.csv", "bess.csv", "chp.csv",
    "ev.csv", "hvac.csv", "in.csv", "pv.csv", "sp.csv", "tm.csv",
    "topo.csv", "wh.csv", "wind.csv",
]


# ---------------------------------------------------------------------------
# Phase A — Test-set builder
# ---------------------------------------------------------------------------

def _load_failed(parent: Path, case: str) -> set:
    """Load failed_buildings.csv for a results case; return set of int ids."""
    path = parent / "in" / INPUT_FOLDER / f"{case}_results" / "failed_buildings.csv"
    if not path.exists():
        print(f"  WARN: missing {path}")
        return set()
    df = pd.read_csv(path)
    return set(int(x) for x in df["bldg_id"].tolist())


def _load_bad_bldgs(parent: Path) -> set:
    """Load bad-building list (bad thermal-model fits). Returns set of int ids."""
    path = parent / BAD_BLDGS_FILE
    if not path.exists():
        print(f"  WARN: missing {path} -- no buildings excluded")
        return set()
    df = pd.read_csv(path)
    return set(int(x) for x in df["bldg_id"].tolist())


def _has_ts_in_all_filter_cases(parent: Path, bldg_id: int) -> bool:
    """True iff ts.csv exists at <case>_results/<bldg_id>/update_0/out/ts.csv for all filter cases."""
    for case in FILTER_CASES:
        ts_path = (parent / "in" / INPUT_FOLDER / f"{case}_results"
                   / str(bldg_id) / f"update_{UPDATE_NUM}" / "out" / "ts.csv")
        if not ts_path.exists():
            return False
    return True


def build_test_set(parent: Path) -> list:
    """Build (or reuse) the 100-building test set."""
    test_set_path = Path(__file__).parent / TEST_SET_FILE

    if test_set_path.exists():
        df = pd.read_csv(test_set_path)
        ids = [int(x) for x in df["bldg_id"].tolist()]
        print(f"\nReusing existing test set from {test_set_path} ({len(ids)} buildings)")
        return ids

    print(f"\n=== Phase A: building test set ===")

    # 1. Failed-building union across all filter cases
    failed_union = set()
    for case in FILTER_CASES:
        failed = _load_failed(parent, case)
        print(f"  {case}_results: {len(failed)} failed")
        failed_union |= failed
    print(f"  Failed union (any of 8 cases): {len(failed_union)}")

    # 1b. Bad-thermal-model exclusion list
    bad_bldgs = _load_bad_bldgs(parent)
    print(f"  Bad thermal-model buildings: {len(bad_bldgs)}")

    # 2. Candidate pool from elec_0_flat input dir
    base_dir = parent / "in" / INPUT_FOLDER / "elec_0_flat"
    candidates = sorted(
        int(d.name) for d in base_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    print(f"  Candidate buildings (elec_0_flat dir): {len(candidates)}")

    # 3. Drop failed + bad buildings
    exclude = failed_union | bad_bldgs
    non_failed = [b for b in candidates if b not in exclude]
    print(f"  After excluding failed+bad: {len(non_failed)}")

    # 4. Shuffle with seed
    rng = random.Random(SEED)
    rng.shuffle(non_failed)

    # 5. Walk, accept ones that have ts.csv in all 8 cases
    chosen = []
    skipped_missing_ts = 0
    for bldg_id in non_failed:
        if _has_ts_in_all_filter_cases(parent, bldg_id):
            chosen.append(bldg_id)
            if len(chosen) >= N_TEST:
                break
        else:
            skipped_missing_ts += 1

    print(f"  Accepted: {len(chosen)}, Skipped (missing ts.csv somewhere): {skipped_missing_ts}")

    if len(chosen) < N_TEST:
        raise RuntimeError(
            f"Only found {len(chosen)} eligible buildings; need {N_TEST}. "
            f"Pool exhausted ({len(non_failed)} non-failed)."
        )

    # 6. Save
    pd.DataFrame({"bldg_id": chosen}).to_csv(test_set_path, index=False)
    print(f"  Wrote {test_set_path}")

    return chosen


# ---------------------------------------------------------------------------
# Phase B — Per-building derate worker
# ---------------------------------------------------------------------------

def process_derate(bldg_id: int, parent_path: Path,
                   base_case: str, case_name: str, pct) -> dict:
    """Copy 14 input CSVs and derate hvac.csv. pct=None -> no derate (base copy)."""
    t0     = time.perf_counter()
    result = {"bldg_id": bldg_id, "case": case_name, "success": False,
              "elapsed": 0.0, "error": None}

    try:
        src_dir = (parent_path / "in" / INPUT_FOLDER / base_case
                   / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
        out_dir = (parent_path / "in" / INPUT_FOLDER / case_name
                   / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Explicit per-file copy — never copytree (avoids accidentally pulling out/)
        import shutil
        for fname in ALL_FILES:
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)

        # Derate hvac.csv (skip when pct is None -- _derate_base passthrough)
        if pct is not None:
            factor = 1.0 - (pct / 100.0)
            hvac_df = pd.read_csv(out_dir / "hvac.csv")
            hvac_df["pHVmx"] = hvac_df["pHVmx"] * factor
            hvac_df["pACmx"] = hvac_df["pACmx"] * factor
            hvac_df.to_csv(out_dir / "hvac.csv", index=False)

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

    # --- Phase A: test set
    test_set = build_test_set(parent_path)
    print(f"\nTest set: {len(test_set)} buildings")
    print(f"  First 5: {test_set[:5]}")
    print(f"  Last 5:  {test_set[-5:]}")

    if DEBUG:
        print("\nDEBUG=True -> exiting before writing derivative cases.")
        raise SystemExit(0)

    # --- Phase B: build (case, pct, bldg_id) task list
    # variants: [(suffix, pct_or_None), ...]
    variants = []
    if INCLUDE_BASE:
        variants.append(("base", None))
    variants.extend((str(pct), pct) for pct in DERATE_PCTS)

    tasks = []
    for base_case in BASE_CASES:
        for suffix, pct in variants:
            case_name = f"{base_case}_derate_{suffix}"
            for bldg_id in test_set:
                if RESUME:
                    out_dir = (parent_path / "in" / INPUT_FOLDER / case_name
                               / str(bldg_id) / f"update_{UPDATE_NUM}" / "in")
                    if all((out_dir / f).exists() for f in ALL_FILES):
                        continue
                tasks.append((bldg_id, parent_path, base_case, case_name, pct))

    if not tasks:
        print("\nNo tasks to run (RESUME found all outputs).")
    else:
        print(f"\n=== Phase B: generating derivative cases ===")
        case_names = sorted({t[3] for t in tasks})
        print(f"  Cases: {', '.join(case_names)}")
        print(f"  Tasks: {len(tasks)}")

        results     = []
        all_failed  = []
        wall_start  = time.perf_counter()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_derate, *t): t for t in tasks}
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                if len(results) % 100 == 0 or not r["success"]:
                    elapsed = time.perf_counter() - wall_start
                    status  = "OK" if r["success"] else "FAILED"
                    print(f"  [{len(results)}/{len(tasks)}] "
                          f"bldg {r['bldg_id']:>6} {r['case']}  "
                          f"{status}  {elapsed:.0f}s elapsed")
                    if not r["success"]:
                        print(f"    ERROR: {r['error'].splitlines()[-1]}")
                if not r["success"]:
                    all_failed.append(r)

        ok = sum(1 for r in results if r["success"])
        print(f"  Phase B done: {ok}/{len(results)} succeeded "
              f"in {time.perf_counter() - wall_start:.0f}s")

        if all_failed:
            log_path = parent_path / "in" / INPUT_FOLDER / "hvac_derate_failures.log"
            with open(log_path, "w") as f:
                f.write(f"HVAC derate run {datetime.now().isoformat()}\n\n")
                for r in all_failed:
                    f.write(f"=== bldg_id={r['bldg_id']} case={r['case']} ===\n{r['error']}\n\n")
            print(f"  Failure log: {log_path}")

    # --- Phase C: single combined archive
    if COMPRESS:
        decarb_dir = parent_path / "in" / INPUT_FOLDER
        archive_path = decarb_dir / ARCHIVE_NAME

        case_dirs = []
        for base_case in BASE_CASES:
            suffixes = (["base"] if INCLUDE_BASE else []) + [str(p) for p in DERATE_PCTS]
            for suffix in suffixes:
                cd = decarb_dir / f"{base_case}_derate_{suffix}"
                if cd.exists():
                    case_dirs.append(cd)

        if not case_dirs:
            print("\nNo case dirs found to compress.")
        else:
            print(f"\n=== Phase C: compressing {len(case_dirs)} cases -> {archive_path.name} ===")
            t0 = time.perf_counter()
            with tarfile.open(archive_path, "w:gz") as tar:
                for cd in case_dirs:
                    print(f"  + {cd.name}")
                    tar.add(cd, arcname=cd.name)
            print(f"  done ({time.perf_counter() - t0:.0f}s)  -> {archive_path}")
