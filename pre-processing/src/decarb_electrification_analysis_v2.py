"""
Electrification Interpolation — 0% to 100%
============================================
Starting from elec_0_flat_results (partially electrified baseline),
identifies non-electrified buildings via metadata parquet, then
randomly electrifies them in batches of whole building-instances
until reaching 10%, 20%, ... 90% of total instances electrified.

For each percentage level, saves:
  - aggregated_buy_MW.csv         (8760 rows, columns = pct levels)
  - aggregated_heating_MW.csv
  - aggregated_cooling_MW.csv
  - summary.csv                   (peak, total consumption, total cost per level)

Outputs go to:  ERCOT_ROOT/out/decarb_results_alt/elec_all_flat/
"""

import os
import random
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Configuration ──────────────────────────────────────────────────────────────
ERCOT_ROOT       = r"D:\shared\ercot_project"
UPGRADE          = "update_0"
RANDOM_SEED      = 42
N_WORKERS        = 20
TOLERANCE_PCT    = 0.1          # stop adding buildings within 0.1% of target
PCT_STEPS        = list(range(65, 100, 5))    # 65,70,75,...,95  (60%=baseline, 100%=endpoint)

CASE_0           = "elec_0_flat_results"
CASE_100         = "elec_100_flat_results"

DECARB_IN_ROOT   = os.path.join(ERCOT_ROOT, "in",  "decarb_inputs_alt")
DECARB_OUT_ROOT  = os.path.join(ERCOT_ROOT, "out", "decarb_results_alt")
METADATA_PARQUET = os.path.join(ERCOT_ROOT, "in",  "TX_upgrade0.parquet")
OUTPUT_DIR       = os.path.join(DECARB_OUT_ROOT, "elec_all_flat")

# Column names inside per-building ts.csv
# hvac_HT / hvac_AC = electricity consumed by heating/cooling (kW) — what we aggregate
# HVACht / HVACac   = thermal energy delivered (kWh, ~2x due to COP) — NOT used here
BUY_COL      = "buy"
HEATING_COL1 = "HVACht"   # electric heat pump heating
HEATING_COL2 = "CHPht"    # CHP heating (combined with HVACht for total heating)
COOLING_COL  = "HVACac"   # cooling

# Column names in the aggregated CSVs produced previously
AGG_BUY_COL     = "Grid Purchases [GW]"
AGG_HEATING_COL = "Heating Generated [GWh]"
AGG_COOLING_COL = "Cooling Generated [GWh]"

# Tariff column in tm.csv (electricity purchase price $/kWh)
TARIFF_COL = "pQcostBuy"

# In TX_upgrade0.parquet, a building is already electrified if
# in.heating_fuel == "Electricity". Any other value (Natural Gas, Propane,
# Fuel Oil, etc.) means it is NOT yet electrified and is a candidate to switch.
HEATING_FUEL_COL      = "in.heating_fuel"
HEATING_FUEL_ELEC_VAL = "Electricity"

# GenX output config
GENX_OUTPUT_DIR = os.path.join(ERCOT_ROOT, "out", "genx_inputs", "demand_inputs")
VOLL            = 2_000_000   # Value of lost load ($/MWh)
DEMAND_SEGMENT  = 1
COST_CURTAIL    = 1
MAX_CURTAIL     = 1
COST_MWH        = 2000
# ──────────────────────────────────────────────────────────────────────────────


def load_building_mapping():
    """Load building_mapping.csv from the elec_100 output directory."""
    path = os.path.join(DECARB_OUT_ROOT, CASE_100, "building_mapping.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"building_mapping.csv not found: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded building_mapping.csv: {len(df)} rows, columns: {list(df.columns)}")
    return df


def load_metadata():
    """Load TX_upgrade0.parquet and return DataFrame."""
    print(f"Loading metadata from: {METADATA_PARQUET}")
    df = pd.read_parquet(METADATA_PARQUET)
    print(f"  Metadata shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:20])}{'...' if len(df.columns) > 20 else ''}")
    return df


def identify_non_electrified_buildings(mapping_df, metadata_df):
    """
    A building is already electrified in elec_0 if in.heating_fuel == "Electricity".
    Any other value (Natural Gas, Propane, Fuel Oil, etc.) means it is NOT yet
    electrified and is a candidate to switch to the elec_100 profile.

    Each row in mapping_df is one weighted instance of a building.

    Returns:
        non_elec_ids       : set of unique bldg_ids not yet electrified
        non_elec_instances : list of bldg_ids, one per mapping row, non-elec only
        all_instances      : list of all bldg_ids, one entry per mapping row
    """
    print("\nIdentifying non-electrified buildings via \'in.heating_fuel\'...")
    print(f"  Mapping rows (total instances): {len(mapping_df)}")

    if HEATING_FUEL_COL not in metadata_df.columns:
        raise ValueError(
            f"Column \"{HEATING_FUEL_COL}\" not found in metadata parquet.\n"
            f"Available columns: {list(metadata_df.columns[:30])}"
        )

    # Print distribution so user can verify
    print(f"  in.heating_fuel value counts:")
    for val, cnt in metadata_df[HEATING_FUEL_COL].value_counts().items():
        marker = " <- already electrified" if val == HEATING_FUEL_ELEC_VAL else " <- candidate to switch"
        print(f"    {str(val):<30s} {cnt:>7d}{marker}")

    already_elec_ids = set(
        metadata_df.loc[
            metadata_df[HEATING_FUEL_COL] == HEATING_FUEL_ELEC_VAL, "bldg_id"
        ].unique()
    )
    non_elec_meta_ids = set(
        metadata_df.loc[
            metadata_df[HEATING_FUEL_COL] != HEATING_FUEL_ELEC_VAL, "bldg_id"
        ].unique()
    )

    all_mapped_ids = set(mapping_df["new_bldg_id"].unique())

    # Intersect with mapping
    non_elec_ids = non_elec_meta_ids & all_mapped_ids
    already_ids  = already_elec_ids  & all_mapped_ids

    print(f"  Not yet electrified: {len(non_elec_ids):>6d} unique buildings")
    print(f"  Already electrified: {len(already_ids):>6d} unique buildings")
    unmapped = all_mapped_ids - non_elec_ids - already_ids
    if unmapped:
        print(f"  WARNING: {len(unmapped)} mapped bldg_ids not in parquet -> "
              f"treated as already electrified.")

    # Expand to instance lists (one entry per mapping row)
    all_instances      = mapping_df["new_bldg_id"].tolist()
    non_elec_instances = mapping_df[
        mapping_df["new_bldg_id"].isin(non_elec_ids)
    ]["new_bldg_id"].tolist()

    total = len(all_instances)
    n_non = len(non_elec_instances)
    print(f"  Total instances:            {total:>8d}")
    print(f"  Non-electrified instances:  {n_non:>8d}  ({100*n_non/total:.1f}%)")
    print(f"  Already-electrified:        {total-n_non:>8d}  ({100*(total-n_non)/total:.1f}%)")

    return non_elec_ids, non_elec_instances, all_instances


def build_electrification_batches(non_elec_instances, total_instances, pct_steps, seed=RANDOM_SEED):
    """
    Randomly shuffle non-electrified instances and determine which buildings
    to 'switch on' at each percentage step.

    Strategy:
      - Shuffle all non-electrified instances randomly (seeded)
      - For each target %, compute target count = total_instances * pct / 100
        minus the already-electrified count (baseline)
      - Walk through shuffled list adding whole-building chunks (all instances
        of a given bldg_id at once) until within TOLERANCE_PCT of target

    Returns dict: {pct: [list of bldg_ids newly electrified UP TO this pct]}
    """
    random.seed(seed)
    rng = random.Random(seed)

    # Group instances by bldg_id to add whole buildings at once
    from collections import defaultdict
    instance_groups = defaultdict(int)
    for bldg_id in non_elec_instances:
        instance_groups[bldg_id] += 1

    unique_bldgs = list(instance_groups.keys())
    rng.shuffle(unique_bldgs)

    total = total_instances
    already_elec_count = total - len(non_elec_instances)  # instances already electrified in elec_0

    print(f"\n  Total instances:              {total}")
    print(f"  Already electrified:          {already_elec_count} ({100*already_elec_count/total:.1f}%)")
    print(f"  Non-electrified to assign:    {len(non_elec_instances)}")

    batches = {}   # pct -> sorted list of new bldg_ids electrified up to that pct
    cumulative_new = []       # bldg_ids added so far (in order)
    cumulative_new_count = 0  # instance count added so far
    bldg_pointer = 0

    for pct in sorted(pct_steps):
        target_new_instances = int(round(total * pct / 100.0)) - already_elec_count
        target_new_instances = max(0, target_new_instances)
        tolerance_count = total * TOLERANCE_PCT / 100.0

        print(f"\n  Target {pct}%: need {target_new_instances} new instances "
              f"(tolerance ±{tolerance_count:.0f})")

        # Keep adding whole buildings until within tolerance
        while bldg_pointer < len(unique_bldgs):
            deficit = target_new_instances - cumulative_new_count
            if abs(deficit) <= tolerance_count:
                break
            if deficit <= 0:
                break
            next_bldg = unique_bldgs[bldg_pointer]
            next_count = instance_groups[next_bldg]
            cumulative_new.append(next_bldg)
            cumulative_new_count += next_count
            bldg_pointer += 1

        actual_pct = (already_elec_count + cumulative_new_count) / total * 100
        print(f"    Added {len(cumulative_new)} unique buildings, "
              f"{cumulative_new_count} instances → actual {actual_pct:.2f}%")
        batches[pct] = list(cumulative_new)  # snapshot at this pct

    return batches, already_elec_count


def load_tariff(case_name):
    """
    Load tariff from tm.csv found in any building folder under the case.
    Uses column 'pQcostBuy' (electricity purchase price $/kWh).
    First column is a timestamp index; remaining columns are data.
    Returns a numpy array of length 8760.
    """
    case_dir = os.path.join(DECARB_IN_ROOT, case_name)
    for entry in sorted(os.listdir(case_dir)):
        try:
            int(entry)
        except ValueError:
            continue
        tm_path = os.path.join(case_dir, entry, UPGRADE, "in", "tm.csv")
        if os.path.exists(tm_path):
            print(f"  Loading tariff from: {tm_path}")
            tm = pd.read_csv(tm_path, index_col=0)   # first col is timestamp
            if TARIFF_COL not in tm.columns:
                raise ValueError(
                    f"Column '{TARIFF_COL}' not found in tm.csv. "
                    f"Available: {list(tm.columns)}"
                )
            rates = tm[TARIFF_COL].values[:8760].astype(np.float64)
            if len(rates) < 8760:
                rates = np.pad(rates, (0, 8760 - len(rates)), mode="edge")
            print(f"  Tariff '{TARIFF_COL}': "
                  f"min={rates.min():.4f}, max={rates.max():.4f} $/kWh, "
                  f"mean={rates.mean():.4f} $/kWh")
            return rates
    raise FileNotFoundError(f"No tm.csv found in any building under {case_dir}")


def _load_building_ts(args):
    """Worker: load a single building's ts.csv and return (bldg_id, buy, heating, cooling)."""
    bldg_id, case_name = args
    ts_path = os.path.join(
        DECARB_IN_ROOT, case_name, str(bldg_id), UPGRADE, "out", "ts.csv"
    )
    if not os.path.exists(ts_path):
        return bldg_id, None, None, None
    try:
        df = pd.read_csv(ts_path, usecols=[BUY_COL, HEATING_COL1, HEATING_COL2, COOLING_COL])
        buy     = df[BUY_COL].values[:8760].astype(np.float64)
        heating = (df[HEATING_COL1].values[:8760] + df[HEATING_COL2].values[:8760]).astype(np.float64)  # HVACht + CHPht
        cooling = df[COOLING_COL].values[:8760].astype(np.float64)
        return bldg_id, buy, heating, cooling
    except Exception as e:
        print(f"  ERROR loading {case_name}/{bldg_id}: {e}")
        return bldg_id, None, None, None


def get_valid_bldg_ids(bldg_ids, case_names):
    """
    Return only bldg_ids that have a ts.csv in ALL of the given case directories.
    Runs quickly since it only checks file existence, no I/O.
    """
    valid = []
    missing_by_case = {c: [] for c in case_names}
    for bldg_id in bldg_ids:
        ok = True
        for case_name in case_names:
            ts_path = os.path.join(
                DECARB_IN_ROOT, case_name, str(bldg_id), UPGRADE, "out", "ts.csv"
            )
            if not os.path.exists(ts_path):
                missing_by_case[case_name].append(bldg_id)
                ok = False
        if ok:
            valid.append(bldg_id)
    for case_name, missing in missing_by_case.items():
        if missing:
            print(f"  Missing ts.csv in {case_name}: {len(missing)} buildings "
                  f"-> {missing[:10]}{'...' if len(missing) > 10 else ''}")
    print(f"  Valid buildings (ts.csv in all cases): {len(valid)} / {len(bldg_ids)}")
    return set(valid)


def load_all_building_profiles(bldg_ids, case_name, label=""):
    """
    Parallel load of ts.csv for a list of unique bldg_ids from a given case.
    Any missing buildings are filled with the average profile of all loaded buildings.
    Returns dict: {bldg_id: (buy_8760, heating_8760, cooling_8760)}
    """
    print(f"  Loading {len(bldg_ids)} unique building profiles from {label or case_name}...")
    args = [(b, case_name) for b in bldg_ids]
    profiles = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_load_building_ts, a): a for a in args}
        done = 0
        for future in as_completed(futures):
            done += 1
            bldg_id, buy, heating, cooling = future.result()
            if buy is not None:
                profiles[bldg_id] = (buy, heating, cooling)
            if done % 2000 == 0:
                print(f"    Loaded {done}/{len(bldg_ids)}...")

    n_loaded = len(profiles)
    n_missing = len(bldg_ids) - n_loaded
    print(f"  Successfully loaded: {n_loaded}/{len(bldg_ids)}")

    if n_missing > 0:
        # Compute average profile across all loaded buildings
        all_buy     = np.mean([p[0] for p in profiles.values()], axis=0)
        all_heating = np.mean([p[1] for p in profiles.values()], axis=0)
        all_cooling = np.mean([p[2] for p in profiles.values()], axis=0)
        avg_profile = (all_buy, all_heating, all_cooling)

        missing_ids = set(bldg_ids) - set(profiles.keys())
        print(f"  Filling {n_missing} missing buildings with average profile: "
              f"{sorted(missing_ids)[:10]}{'...' if n_missing > 10 else ''}")
        for bldg_id in missing_ids:
            profiles[bldg_id] = avg_profile

    return profiles


def aggregate_profiles(
    mapping_df,
    non_elec_bldg_ids_at_pct,   # set of new_bldg_ids being switched at this pct level
    profiles_0,                  # dict new_bldg_id -> (buy,heat,cool) from elec_0
    profiles_100,                # dict new_bldg_id -> (buy,heat,cool) from elec_100
):
    """
    mapping_df has columns: org_bldg_id, new_bldg_id, count
    Each unique new_bldg_id has a ts.csv profile, weighted by count.

    Step 1: sum ALL unique new_bldg_ids weighted by count using elec_0 as baseline.
    Step 2: for switched buildings, add delta (elec_100 - elec_0) * count.

    Units: profiles in kW, count is number of households -> sum in kW -> /1000 -> MW.
    Endpoints are in GW from pre-aggregated CSV -> *1000 -> MW. Both match.
    """
    # Precompute count per unique new_bldg_id
    counts = mapping_df.groupby("new_bldg_id")["count"].sum()

    # Step 1: baseline — all buildings weighted by count using elec_0
    buy_agg  = np.zeros(8760, dtype=np.float64)
    heat_agg = np.zeros(8760, dtype=np.float64)
    cool_agg = np.zeros(8760, dtype=np.float64)

    for bldg_id, cnt in counts.items():
        p0 = profiles_0.get(bldg_id)
        if p0 is None:
            continue
        buy_agg  += p0[0] * cnt
        heat_agg += p0[1] * cnt
        cool_agg += p0[2] * cnt

    # Step 2: switched buildings — add delta * count
    switched_counts = counts[counts.index.isin(non_elec_bldg_ids_at_pct)]
    for bldg_id, cnt in switched_counts.items():
        p0   = profiles_0.get(bldg_id)
        p100 = profiles_100.get(bldg_id)
        if p0 is None or p100 is None:
            continue
        buy_agg  += (p100[0] - p0[0]) * cnt
        heat_agg += (p100[1] - p0[1]) * cnt
        cool_agg += (p100[2] - p0[2]) * cnt

    # kW * count (households) -> sum in kW -> /1000 -> MW
    return buy_agg / 1000.0, heat_agg / 1000.0, cool_agg / 1000.0


def compute_summary(buy_agg, heat_agg, cool_agg, tariff_rates, pct_label):
    """Compute peak, total consumption, and total cost."""
    # Convert GW→MW for peak if needed — keep in whatever unit ts.csv uses
    peak_buy    = float(np.max(buy_agg))
    peak_heat   = float(np.max(heat_agg))
    peak_cool   = float(np.max(cool_agg))
    total_buy   = float(np.sum(buy_agg))
    total_heat  = float(np.sum(heat_agg))
    total_cool  = float(np.sum(cool_agg))

    # Cost: sum(buy_hourly * rate_hourly)
    n = min(len(buy_agg), len(tariff_rates))
    total_cost  = float(np.dot(buy_agg[:n] * 1000.0, tariff_rates[:n]))  # MW -> kWh (*1000) * $/kWh

    return {
        "electrification_pct":        pct_label,
        "peak_buy":                    peak_buy,
        "peak_heating":                peak_heat,
        "peak_cooling":                peak_cool,
        "total_buy":                   total_buy,
        "total_heating":               total_heat,
        "total_cooling":               total_cool,
        "total_cost":                  total_cost,
    }


def load_aggregated_csv(case_name):
    """Load the pre-existing aggregated CSV for a case (0% or 100% endpoints)."""
    path = os.path.join(DECARB_OUT_ROOT, case_name, f"{case_name}_aggregated_ts.csv")
    if not os.path.exists(path):
        # Try alternate naming
        alt = os.path.join(DECARB_OUT_ROOT, case_name, "aggregated_ts.csv")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"Aggregated CSV not found. Tried:\n  {path}\n  {alt}"
            )
    df = pd.read_csv(path)
    return df


# =============================================================================
# GenX Demand File Writer
# =============================================================================

def write_genx_demand(buy_agg_mw, pct_label, case_suffix=""):
    """
    Write a GenX-ready Demand_data.csv for a given aggregated buy profile.

    buy_agg_mw : 8760-length array in MW (all arrays are MW throughout)
    pct_label  : e.g. "elec_060pct"
    case_suffix: e.g. "_flat" or "_tou" for tariff variant
    """
    case_name   = f"{pct_label}{case_suffix}"
    out_dir     = os.path.join(GENX_OUTPUT_DIR, case_name)
    os.makedirs(out_dir, exist_ok=True)

    load_mw = buy_agg_mw  # already in MW

    # Build the 8760-row dataframe
    rows = []
    for i in range(8760):
        time_idx = i + 1
        if i == 0:
            # First row carries all preamble values
            rows.append({
                "Voll":                              VOLL,
                "Demand_Segment":                    DEMAND_SEGMENT,
                "Cost_of_Demand_Curtailment_per_MW": COST_CURTAIL,
                "Max_Demand_Curtailment":            MAX_CURTAIL,
                "$/MWh":                             COST_MWH,
                "Rep_Periods":                       1,
                "Timesteps_per_Rep_Period":           8760,
                "Sub_Weights":                       8760,
                "Time_Index":                        time_idx,
                "Load_MW_z1":                        round(load_mw[i], 5),
            })
        else:
            # Remaining rows: preamble columns are empty
            rows.append({
                "Voll":                              "",
                "Demand_Segment":                    "",
                "Cost_of_Demand_Curtailment_per_MW": "",
                "Max_Demand_Curtailment":            "",
                "$/MWh":                             "",
                "Rep_Periods":                       "",
                "Timesteps_per_Rep_Period":           "",
                "Sub_Weights":                       "",
                "Time_Index":                        time_idx,
                "Load_MW_z1":                        round(load_mw[i], 5),
            })

    df = pd.DataFrame(rows, columns=[
        "Voll", "Demand_Segment", "Cost_of_Demand_Curtailment_per_MW",
        "Max_Demand_Curtailment", "$/MWh", "Rep_Periods",
        "Timesteps_per_Rep_Period", "Sub_Weights", "Time_Index", "Load_MW_z1"
    ])

    out_path = os.path.join(out_dir, "Demand_data.csv")
    df.to_csv(out_path, index=False)
    print(f"    GenX demand file -> {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"  Electrification Interpolation — elec_all_flat")
    print(f"{'=' * 70}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # ── 1. Load building mapping (same for both cases) ──────────────────────
    print(f"\n[1/7] Loading building mapping...")
    mapping_df = load_building_mapping()
    print(f"  Total instances (rows): {len(mapping_df)}")
    print(f"  Unique buildings:       {mapping_df['new_bldg_id'].nunique()}")

    # ── 2. Load metadata parquet ─────────────────────────────────────────────
    print(f"\n[2/7] Loading metadata parquet...")
    metadata_df = load_metadata()

    # ── 3. Identify non-electrified buildings ────────────────────────────────
    print(f"\n[3/7] Identifying non-electrified buildings...")
    non_elec_ids, non_elec_instances, all_instances = identify_non_electrified_buildings(
        mapping_df, metadata_df
    )

    # Missing buildings in either case will be filled with average profile at load time

    # ── 4. Build electrification batches ─────────────────────────────────────
    print(f"\n[4/7] Building electrification batches (10%→90%)...")
    batches, already_elec_count = build_electrification_batches(
        non_elec_instances, len(all_instances), PCT_STEPS
    )

    # ── 5. Load tariff ───────────────────────────────────────────────────────
    print(f"\n[5/7] Loading tariff...")
    tariff_rates = load_tariff(CASE_0)

    # ── 6. Load all required per-building profiles ───────────────────────────
    # Collect ALL unique bldg_ids we'll ever need from each case
    print(f"\n[6/7] Loading per-building ts.csv profiles...")

    # All unique buildings in mapping
    all_unique_bldgs = mapping_df["new_bldg_id"].unique().tolist()

    # For elec_100: only the non-electrified buildings (candidates to switch)
    non_elec_unique = list(non_elec_ids)

    # All buildings passed to loader — any missing ts.csv filled with average profile
    profiles_0   = load_all_building_profiles(all_unique_bldgs, CASE_0,   "elec_0")
    profiles_100 = load_all_building_profiles(non_elec_unique,  CASE_100, "elec_100")

    # All buildings are now covered — missing ones get average profile filled in
    # so mapping_df stays complete and instance counts remain consistent

    # ── 7. Load endpoints (0% and 100%) from pre-aggregated CSVs ────────────
    print(f"\n[7/7] Loading pre-aggregated endpoint CSVs...")
    try:
        df_0   = load_aggregated_csv(CASE_0)
        df_100 = load_aggregated_csv(CASE_100)
        buy_0   = df_0[AGG_BUY_COL].values[:8760]
        heat_0  = df_0[AGG_HEATING_COL].values[:8760]
        cool_0  = df_0[AGG_COOLING_COL].values[:8760]
        buy_100  = df_100[AGG_BUY_COL].values[:8760]
        heat_100 = df_100[AGG_HEATING_COL].values[:8760]
        cool_100 = df_100[AGG_COOLING_COL].values[:8760]
        endpoints_loaded = True
        print("  Endpoint CSVs loaded successfully.")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  Will compute 0% and 100% from per-building profiles instead.")
        endpoints_loaded = False

    # ── 8. Aggregate profiles for each percentage level ──────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Aggregating profiles for each electrification level...")
    print(f"{'─' * 70}")

    # Storage for output CSVs
    all_buy     = {}   # pct_label -> 8760 array
    all_heat    = {}
    all_cool    = {}
    summaries   = []

    # Include 0% and 100% endpoints
    all_levels = [60] + sorted(PCT_STEPS) + [100]

    for pct in all_levels:
        pct_label = f"elec_{pct:03d}pct"
        print(f"\n  → {pct}% electrification ({pct_label})")

        if pct == 60 and endpoints_loaded:
            # Aggregated CSVs are in GW; convert to kW to match per-building profile units
            buy_agg  = buy_0  * 1000.0
            heat_agg = heat_0 * 1000.0
            cool_agg = cool_0 * 1000.0
            print(f"    Using pre-aggregated elec_0 CSV (60% baseline).")
        elif pct == 100 and endpoints_loaded:
            # Aggregated CSVs are in GW; convert to kW to match per-building profile units
            buy_agg  = buy_100  * 1000.0
            heat_agg = heat_100 * 1000.0
            cool_agg = cool_100 * 1000.0
            print(f"    Using pre-aggregated elec_100 CSV.")
        else:
            # Determine which non-electrified buildings are switched on at this pct
            if pct == 60:
                newly_elec_set = set()
            elif pct == 100:
                newly_elec_set = non_elec_ids
            else:
                newly_elec_set = set(batches[pct])

            print(f"    Newly electrified unique buildings: {len(newly_elec_set)}")
            buy_agg, heat_agg, cool_agg = aggregate_profiles(
                mapping_df,
                newly_elec_set,
                profiles_0,
                profiles_100,
            )

        all_buy[pct_label]  = buy_agg
        all_heat[pct_label] = heat_agg
        all_cool[pct_label] = cool_agg

        summary = compute_summary(buy_agg, heat_agg, cool_agg, tariff_rates, pct_label)
        summaries.append(summary)

        peak_hr = int(np.argmax(buy_agg)) % 24
        print(f"    Peak buy: {summary['peak_buy']:.2f}  |  "
              f"Total buy: {summary['total_buy']:.2f}  |  "
              f"Peak hour: {peak_hr:02d}:00  |  "
              f"Cost: ${summary['total_cost']:,.0f}")

        # Write GenX Demand_data.csv for this electrification level
        write_genx_demand(buy_agg, pct_label, case_suffix="_flat")  # buy_agg in MW

    # ── 9. Save output CSVs ──────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Saving output files to: {OUTPUT_DIR}")
    print(f"{'─' * 70}")

    hour_index = list(range(8760))

    # Buy / Electricity
    df_buy = pd.DataFrame(all_buy, index=hour_index)
    df_buy.index.name = "hour"
    buy_path = os.path.join(OUTPUT_DIR, "aggregated_buy_MW.csv")
    df_buy.to_csv(buy_path)
    print(f"  Saved: aggregated_buy_MW.csv  ({df_buy.shape})")

    # Heating
    df_heat = pd.DataFrame(all_heat, index=hour_index)
    df_heat.index.name = "hour"
    heat_path = os.path.join(OUTPUT_DIR, "aggregated_heating_MW.csv")
    df_heat.to_csv(heat_path)
    print(f"  Saved: aggregated_heating_MW.csv  ({df_heat.shape})")

    # Cooling
    df_cool = pd.DataFrame(all_cool, index=hour_index)
    df_cool.index.name = "hour"
    cool_path = os.path.join(OUTPUT_DIR, "aggregated_cooling_MW.csv")
    df_cool.to_csv(cool_path)
    print(f"  Saved: aggregated_cooling_MW.csv  ({df_cool.shape})")

    # Summary
    df_summary = pd.DataFrame(summaries)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"  Saved: summary.csv")

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 90}")
    print(df_summary.to_string(index=False, float_format="{:,.2f}".format))

    print(f"\n{'=' * 70}")
    print(f"  Done!")
    print(f"{'=' * 70}\n")