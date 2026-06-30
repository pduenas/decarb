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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

HEATING_FUEL_COL      = "in.heating_fuel"
HEATING_FUEL_ELEC_VAL = "Electricity"

# GenX output config
WRITE_GENX_INPUTS = False   # set True to write Demand_data.csv files for GenX
GENX_OUTPUT_DIR   = os.path.join(ERCOT_ROOT, "out", "genx_inputs", "demand_inputs")
VOLL              = 2_000_000
DEMAND_SEGMENT    = 1
COST_CURTAIL      = 1
MAX_CURTAIL       = 1
COST_MWH          = 2000
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
    Returns non_elec_ids, non_elec_instances, all_instances.
    """
    print("\nIdentifying non-electrified buildings via 'in.heating_fuel'...")
    print(f"  Mapping rows (total instances): {len(mapping_df)}")

    if HEATING_FUEL_COL not in metadata_df.columns:
        raise ValueError(
            f"Column \"{HEATING_FUEL_COL}\" not found in metadata parquet.\n"
            f"Available columns: {list(metadata_df.columns[:30])}"
        )

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
    non_elec_ids   = non_elec_meta_ids & all_mapped_ids
    already_ids    = already_elec_ids  & all_mapped_ids

    print(f"  Not yet electrified: {len(non_elec_ids):>6d} unique buildings")
    print(f"  Already electrified: {len(already_ids):>6d} unique buildings")
    unmapped = all_mapped_ids - non_elec_ids - already_ids
    if unmapped:
        print(f"  WARNING: {len(unmapped)} mapped bldg_ids not in parquet -> "
              f"treated as already electrified.")

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
    """
    random.seed(seed)
    rng = random.Random(seed)

    from collections import defaultdict
    instance_groups = defaultdict(int)
    for bldg_id in non_elec_instances:
        instance_groups[bldg_id] += 1

    unique_bldgs = list(instance_groups.keys())
    rng.shuffle(unique_bldgs)

    total              = total_instances
    already_elec_count = total - len(non_elec_instances)

    print(f"\n  Total instances:              {total}")
    print(f"  Already electrified:          {already_elec_count} ({100*already_elec_count/total:.1f}%)")
    print(f"  Non-electrified to assign:    {len(non_elec_instances)}")

    batches              = {}
    cumulative_new       = []
    cumulative_new_count = 0
    bldg_pointer         = 0

    for pct in sorted(pct_steps):
        target_new_instances = int(round(total * pct / 100.0)) - already_elec_count
        target_new_instances = max(0, target_new_instances)
        tolerance_count      = total * TOLERANCE_PCT / 100.0

        print(f"\n  Target {pct}%: need {target_new_instances} new instances "
              f"(tolerance ±{tolerance_count:.0f})")

        while bldg_pointer < len(unique_bldgs):
            deficit = target_new_instances - cumulative_new_count
            if abs(deficit) <= tolerance_count:
                break
            if deficit <= 0:
                break
            next_bldg  = unique_bldgs[bldg_pointer]
            next_count = instance_groups[next_bldg]
            cumulative_new.append(next_bldg)
            cumulative_new_count += next_count
            bldg_pointer += 1

        actual_pct = (already_elec_count + cumulative_new_count) / total * 100
        print(f"    Added {len(cumulative_new)} unique buildings, "
              f"{cumulative_new_count} instances → actual {actual_pct:.2f}%")
        batches[pct] = list(cumulative_new)

    return batches, already_elec_count


def load_tariff(case_name):
    """Load hourly tariff $/kWh from tm.csv."""
    case_dir = os.path.join(DECARB_IN_ROOT, case_name)
    for entry in sorted(os.listdir(case_dir)):
        try:
            int(entry)
        except ValueError:
            continue
        tm_path = os.path.join(case_dir, entry, UPGRADE, "in", "tm.csv")
        if os.path.exists(tm_path):
            print(f"  Loading tariff from: {tm_path}")
            tm = pd.read_csv(tm_path, index_col=0)
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
    """Worker: load a single building's ts.csv."""
    bldg_id, case_name = args
    ts_path = os.path.join(
        DECARB_IN_ROOT, case_name, str(bldg_id), UPGRADE, "out", "ts.csv"
    )
    if not os.path.exists(ts_path):
        return bldg_id, None, None, None
    try:
        df      = pd.read_csv(ts_path, usecols=[BUY_COL, HEATING_COL1, HEATING_COL2, COOLING_COL])
        buy     = df[BUY_COL].values[:8760].astype(np.float64)
        heating = (df[HEATING_COL1].values[:8760] + df[HEATING_COL2].values[:8760]).astype(np.float64)
        cooling = df[COOLING_COL].values[:8760].astype(np.float64)
        return bldg_id, buy, heating, cooling
    except Exception as e:
        print(f"  ERROR loading {case_name}/{bldg_id}: {e}")
        return bldg_id, None, None, None


def load_all_building_profiles(bldg_ids, case_name, label=""):
    """Parallel load of ts.csv for a list of unique bldg_ids."""
    print(f"  Loading {len(bldg_ids)} unique building profiles from {label or case_name}...")
    args     = [(b, case_name) for b in bldg_ids]
    profiles = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_load_building_ts, a): a for a in args}
        done    = 0
        for future in as_completed(futures):
            done += 1
            bldg_id, buy, heating, cooling = future.result()
            if buy is not None:
                profiles[bldg_id] = (buy, heating, cooling)
            if done % 2000 == 0:
                print(f"    Loaded {done}/{len(bldg_ids)}...")

    n_loaded  = len(profiles)
    n_missing = len(bldg_ids) - n_loaded
    print(f"  Successfully loaded: {n_loaded}/{len(bldg_ids)}")

    if n_missing > 0:
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


def aggregate_profiles(mapping_df, non_elec_bldg_ids_at_pct, profiles_0, profiles_100):
    """
    Aggregate buy/heating/cooling across all buildings, switching the
    non-electrified set to elec_100 profiles. Units: kW -> MW (/1000).
    """
    counts   = mapping_df.groupby("new_bldg_id")["count"].sum()
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

    switched_counts = counts[counts.index.isin(non_elec_bldg_ids_at_pct)]
    for bldg_id, cnt in switched_counts.items():
        p0   = profiles_0.get(bldg_id)
        p100 = profiles_100.get(bldg_id)
        if p0 is None or p100 is None:
            continue
        buy_agg  += (p100[0] - p0[0]) * cnt
        heat_agg += (p100[1] - p0[1]) * cnt
        cool_agg += (p100[2] - p0[2]) * cnt

    return buy_agg / 1000.0, heat_agg / 1000.0, cool_agg / 1000.0


def compute_summary(buy_agg, heat_agg, cool_agg, tariff_rates, pct_label):
    """Compute peak, total consumption, and total cost."""
    n          = min(len(buy_agg), len(tariff_rates))
    total_cost = float(np.dot(buy_agg[:n] * 1000.0, tariff_rates[:n]))
    return {
        "electrification_pct": pct_label,
        "peak_buy":            float(np.max(buy_agg)),
        "peak_heating":        float(np.max(heat_agg)),
        "peak_cooling":        float(np.max(cool_agg)),
        "total_buy":           float(np.sum(buy_agg)),
        "total_heating":       float(np.sum(heat_agg)),
        "total_cooling":       float(np.sum(cool_agg)),
        "total_cost":          total_cost,
    }


def load_aggregated_csv(case_name):
    """Load the pre-existing aggregated CSV for a case (0% or 100% endpoints)."""
    path = os.path.join(DECARB_OUT_ROOT, case_name, f"{case_name}_aggregated_ts.csv")
    if not os.path.exists(path):
        alt = os.path.join(DECARB_OUT_ROOT, case_name, "aggregated_ts.csv")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"Aggregated CSV not found. Tried:\n  {path}\n  {alt}"
            )
    return pd.read_csv(path)


# =============================================================================
# Stacked Load Plots
# =============================================================================

def plot_stacked_loads(all_buy, all_heat, all_cool, output_dir):
    from matplotlib.patches import Patch

    pct_labels = sorted(all_buy.keys())
    n_levels   = len(pct_labels)
    days       = np.arange(365)

    month_start_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels     = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    DISTINCT_COLORS = [
        "#2196F3",  # blue
        "#E65100",  # deep orange
        "#4CAF50",  # green
        "#9C27B0",  # purple
        "#E91E63",  # pink
        "#00838F",  # dark cyan
        "#F9A825",  # amber
        "#3F51B5",  # indigo
        "#00BCD4",  # cyan
        "#F44336",  # red
    ]
    colors = [DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i in range(n_levels)]
    colors[-1] = "#1a1a1a"  # 100% electrification -> near black

    bar_w = 0.9 / n_levels

    datasets = [
        ("Electrical Demand", all_buy,  "electrical_demand_stacked.png"),
        ("Heating Load",      all_heat, "heating_load_stacked.png"),
        ("Cooling Load",      all_cool, "cooling_load_stacked.png"),
    ]

    for title, data_dict, filename in datasets:

        # Daily peak per level in GW
        daily_peaks = {}
        for lbl in pct_labels:
            daily_peaks[lbl] = data_dict[lbl][:8760].reshape(365, 24).max(axis=1) / 1000.0

        fig, ax = plt.subplots(figsize=(22, 6))

        legend_elements = []
        for i, (lbl, color) in enumerate(zip(pct_labels, colors)):
            pct_num = lbl.replace("elec_", "").replace("pct", "").lstrip("0") or "0"
            x    = days - 0.45 + bar_w * (i + 0.5)
            peak = daily_peaks[lbl]

            ax.bar(x, peak, width=bar_w, color=color, alpha=0.85, linewidth=0)
            ax.plot(x, peak, color=color, linewidth=0.8, alpha=0.9)
            legend_elements.append(Patch(facecolor=color, alpha=0.85, label=f"{pct_num}%"))

        ax.set_title("Daily Peaks of Electricity Demand in Texas by Heating Electrification Level",
                     fontsize=16, fontweight="bold", pad=14)
        ax.set_xlabel("Day of Year", fontsize=14)
        ax.set_ylabel("Daily Electricity Demand Peak [GW]", fontsize=14)
        ax.set_xticks(month_start_days)
        ax.set_xticklabels(month_labels, fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_xlim(-0.5, 364.5)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        ax.legend(
            handles=legend_elements[::-1],
            title="Electrification\nLevel",
            fontsize=12,
            title_fontsize=13,
            loc="upper right",
            framealpha=0.9,
        )

        plt.tight_layout()
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {plot_path}")

        # ── Endpoint-only plot (60% vs 100%) ──────────────────────────────
        endpoint_labels = ["elec_060pct", "elec_100pct"]
        if all(lbl in data_dict for lbl in endpoint_labels):
            ep_colors = [colors[pct_labels.index("elec_060pct")], "#1a1a1a"]  # 60%=original, 100%=near black
            ep_bar_w  = 0.9 / 2

            fig2, ax2 = plt.subplots(figsize=(22, 6))
            legend_elements2 = []

            for j, (lbl, color) in enumerate(zip(endpoint_labels, ep_colors)):
                pct_num = lbl.replace("elec_", "").replace("pct", "").lstrip("0") or "0"
                x2   = days - 0.45 + ep_bar_w * (j + 0.5)
                peak = data_dict[lbl][:8760].reshape(365, 24).max(axis=1) / 1000.0

                ax2.bar(x2, peak, width=ep_bar_w, color=color, alpha=0.85, linewidth=0)
                ax2.plot(x2, peak, color=color, linewidth=0.8, alpha=0.9)
                legend_elements2.append(Patch(facecolor=color, alpha=0.85, label=f"{pct_num}%"))

            ax2.set_title("Daily Peaks of Electricity Demand in Texas by Heating Electrification Level",
                         fontsize=16, fontweight="bold", pad=14)
            ax2.set_xlabel("Day of Year", fontsize=14)
            ax2.set_ylabel("Daily Electricity Demand Peak [GW]", fontsize=14)
            ax2.set_xticks(month_start_days)
            ax2.set_xticklabels(month_labels, fontsize=12)
            ax2.tick_params(axis="y", labelsize=12)
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax2.set_xlim(-0.5, 364.5)
            ax2.set_ylim(bottom=0)
            ax2.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.legend(
                handles=legend_elements2[::-1],
                title="Electrification\nLevel",
                fontsize=12,
                title_fontsize=13,
                loc="upper right",
                framealpha=0.9,
            )

            plt.tight_layout()
            ep_path = os.path.join(output_dir, filename.replace(".png", "_endpoints.png"))
            plt.savefig(ep_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"  Plot saved: {ep_path}")


# =============================================================================
# GenX Demand File Writer
# =============================================================================

def write_genx_demand(buy_agg_mw, pct_label, case_suffix=""):
    """Write a GenX-ready Demand_data.csv for a given aggregated buy profile."""
    case_name = f"{pct_label}{case_suffix}"
    out_dir   = os.path.join(GENX_OUTPUT_DIR, case_name)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for i in range(8760):
        time_idx = i + 1
        if i == 0:
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
                "Load_MW_z1":                        round(buy_agg_mw[i], 5),
            })
        else:
            rows.append({
                "Voll": "", "Demand_Segment": "",
                "Cost_of_Demand_Curtailment_per_MW": "", "Max_Demand_Curtailment": "",
                "$/MWh": "", "Rep_Periods": "", "Timesteps_per_Rep_Period": "",
                "Sub_Weights": "", "Time_Index": time_idx,
                "Load_MW_z1": round(buy_agg_mw[i], 5),
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

    # ── 1. Load building mapping ─────────────────────────────────────────────
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

    # ── 4. Build electrification batches ─────────────────────────────────────
    print(f"\n[4/7] Building electrification batches...")
    batches, already_elec_count = build_electrification_batches(
        non_elec_instances, len(all_instances), PCT_STEPS
    )

    # ── 5. Load tariff ───────────────────────────────────────────────────────
    print(f"\n[5/7] Loading tariff...")
    tariff_rates = load_tariff(CASE_0)

    # ── 6. Load per-building profiles ────────────────────────────────────────
    print(f"\n[6/7] Loading per-building ts.csv profiles...")
    all_unique_bldgs = mapping_df["new_bldg_id"].unique().tolist()
    non_elec_unique  = list(non_elec_ids)

    profiles_0   = load_all_building_profiles(all_unique_bldgs, CASE_0,   "elec_0")
    profiles_100 = load_all_building_profiles(non_elec_unique,  CASE_100, "elec_100")

    # ── 7. Load endpoint aggregated CSVs ─────────────────────────────────────
    print(f"\n[7/7] Loading pre-aggregated endpoint CSVs...")
    try:
        df_0   = load_aggregated_csv(CASE_0)
        df_100 = load_aggregated_csv(CASE_100)
        buy_0    = df_0[AGG_BUY_COL].values[:8760]
        heat_0   = df_0[AGG_HEATING_COL].values[:8760]
        cool_0   = df_0[AGG_COOLING_COL].values[:8760]
        buy_100  = df_100[AGG_BUY_COL].values[:8760]
        heat_100 = df_100[AGG_HEATING_COL].values[:8760]
        cool_100 = df_100[AGG_COOLING_COL].values[:8760]
        endpoints_loaded = True
        print("  Endpoint CSVs loaded successfully.")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  Will compute 0% and 100% from per-building profiles instead.")
        endpoints_loaded = False

    # ── 8. Aggregate for each pct level ──────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Aggregating profiles for each electrification level...")
    print(f"{'─' * 70}")

    all_buy  = {}
    all_heat = {}
    all_cool = {}
    summaries = []

    all_levels = [60] + sorted(PCT_STEPS) + [100]

    for pct in all_levels:
        pct_label = f"elec_{pct:03d}pct"
        print(f"\n  → {pct}% electrification ({pct_label})")

        if pct == 60 and endpoints_loaded:
            buy_agg  = buy_0  * 1000.0
            heat_agg = heat_0 * 1000.0
            cool_agg = cool_0 * 1000.0
            print(f"    Using pre-aggregated elec_0 CSV (60% baseline).")
        elif pct == 100 and endpoints_loaded:
            buy_agg  = buy_100  * 1000.0
            heat_agg = heat_100 * 1000.0
            cool_agg = cool_100 * 1000.0
            print(f"    Using pre-aggregated elec_100 CSV.")
        else:
            if pct == 60:
                newly_elec_set = set()
            elif pct == 100:
                newly_elec_set = non_elec_ids
            else:
                newly_elec_set = set(batches[pct])

            print(f"    Newly electrified unique buildings: {len(newly_elec_set)}")
            buy_agg, heat_agg, cool_agg = aggregate_profiles(
                mapping_df, newly_elec_set, profiles_0, profiles_100,
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

        if WRITE_GENX_INPUTS:
            write_genx_demand(buy_agg, pct_label, case_suffix="_flat")

    # ── 9. Save output CSVs ──────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Saving output files to: {OUTPUT_DIR}")
    print(f"{'─' * 70}")

    hour_index = list(range(8760))

    df_buy = pd.DataFrame(all_buy, index=hour_index)
    df_buy.index.name = "hour"
    df_buy.to_csv(os.path.join(OUTPUT_DIR, "aggregated_buy_MW.csv"))
    print(f"  Saved: aggregated_buy_MW.csv  ({df_buy.shape})")

    df_heat = pd.DataFrame(all_heat, index=hour_index)
    df_heat.index.name = "hour"
    df_heat.to_csv(os.path.join(OUTPUT_DIR, "aggregated_heating_MW.csv"))
    print(f"  Saved: aggregated_heating_MW.csv  ({df_heat.shape})")

    df_cool = pd.DataFrame(all_cool, index=hour_index)
    df_cool.index.name = "hour"
    df_cool.to_csv(os.path.join(OUTPUT_DIR, "aggregated_cooling_MW.csv"))
    print(f"  Saved: aggregated_cooling_MW.csv  ({df_cool.shape})")

    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    print(f"  Saved: summary.csv")

    # ── 10. Stacked load plots ───────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Generating stacked load plots...")
    print(f"{'─' * 70}")
    plot_stacked_loads(all_buy, all_heat, all_cool, OUTPUT_DIR)

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 90}")
    print(df_summary.to_string(index=False, float_format="{:,.2f}".format))

    print(f"\n{'=' * 70}")
    print(f"  Done!")
    print(f"{'=' * 70}\n")