"""
Diagnostic: are midnight-peakers actually peaking at midnight, or is
np.argmax just picking the first occurrence of a tied / near-flat maximum?

For a sample of midnight-peaker buildings, prints:
  - the annual max value
  - how many other hours are within X% of that max (1%, 0.1%, 0.01%)
  - the buy value at Jan 1 01:00 (idx=0) vs the median, the 99th pct,
    and the max hour
  - whether the building's profile is essentially flat
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd


# ── Config ─────────────────────────────────────────────────
CASE_NAME = "elec_0_tou_cap_results"
ERCOT_ROOT = Path(r"D:\shared\ercot_project")
UPGRADE = "update_0"
N_SAMPLE = 10
TOLERANCES = [0.01, 0.001, 0.0001]   # 1%, 0.1%, 0.01% of the max
RNG_SEED = 0
# ───────────────────────────────────────────────────────────

CASE_DIR = ERCOT_ROOT / "in" / "decarb_inputs_alt" / CASE_NAME
RESULTS_DIR = ERCOT_ROOT / "out" / "decarb_results" / CASE_NAME
PEAK_CSV = RESULTS_DIR / "per_building_peak_hours_decarb.csv"


def load_buy(bldg_id):
    ts_path = CASE_DIR / str(int(bldg_id)) / UPGRADE / "out" / "ts.csv"
    if not ts_path.exists():
        return None
    df = pd.read_csv(ts_path, usecols=["buy"])
    return df["buy"].values.astype(np.float64)


def main():
    print(f"Loading {PEAK_CSV}")
    peaks = pd.read_csv(PEAK_CSV)
    print(f"  {len(peaks):,} buildings total")

    # Midnight-peakers: summer or winter peak at hour 00
    mid = peaks[(peaks["summer_peak_hour"] == 0) |
                (peaks["winter_peak_hour"] == 0)].copy()
    print(f"  {len(mid):,} midnight-peakers")

    # Subgroups: those whose full_peak_idx == 0 (Jan 1, 01:00) vs others
    jan1_01 = mid[mid["full_peak_idx"] == 0]
    other = mid[mid["full_peak_idx"] != 0]
    print(f"    of which full_peak_idx == 0 (Jan 1, 01:00): {len(jan1_01):,}")
    print(f"    other midnight-peakers: {len(other):,}\n")

    rng = np.random.default_rng(RNG_SEED)

    def sample_and_diagnose(label, df_subset):
        print(f"\n{'=' * 72}")
        print(f"  Sampling {min(N_SAMPLE, len(df_subset))} buildings from: "
              f"{label}")
        print(f"{'=' * 72}")
        if len(df_subset) == 0:
            print("  (empty)")
            return
        sample = df_subset.sample(min(N_SAMPLE, len(df_subset)),
                                  random_state=rng.integers(1e9))
        for _, row in sample.iterrows():
            bid = int(row["bldg_id"])
            buy = load_buy(bid)
            if buy is None:
                print(f"  bldg {bid}: ts.csv missing, skip")
                continue
            n = len(buy)
            mx = float(buy.max())
            argmax_idx = int(np.argmax(buy))
            median = float(np.median(buy))
            p99 = float(np.percentile(buy, 99))
            jan1_01_val = float(buy[0])  # hour idx 0 = Jan 1 01:00
            july1_09_val = float(buy[4344 + 8]) if n > 4352 else float("nan")

            # how many hours are within tolerance of the max
            tols = []
            for t in TOLERANCES:
                if mx == 0:
                    tols.append(n)
                else:
                    n_close = int(np.sum(buy >= mx * (1 - t)))
                    tols.append(n_close)

            std = float(buy.std())
            print(f"\n  bldg {bid}  ({label}):")
            print(f"    full_peak_idx (reported): {int(row['full_peak_idx'])}  "
                  f"argmax from ts.csv: {argmax_idx}  "
                  f"summer_peak_hour={int(row['summer_peak_hour'])}, "
                  f"winter_peak_hour={int(row['winter_peak_hour'])}")
            print(f"    max          : {mx:14.6f}")
            print(f"    median       : {median:14.6f}    "
                  f"p99: {p99:14.6f}    std: {std:14.6f}")
            print(f"    buy[Jan 1 01]: {jan1_01_val:14.6f}    "
                  f"(= max? {jan1_01_val == mx})")
            print(f"    buy[Jul 1 09]: {july1_09_val:14.6f}")
            print(f"    hours within 1.0%  of max: {tols[0]:>5d} / {n}")
            print(f"    hours within 0.1%  of max: {tols[1]:>5d} / {n}")
            print(f"    hours within 0.01% of max: {tols[2]:>5d} / {n}")
            # exact ties at max
            exact_ties = int(np.sum(buy == mx))
            print(f"    EXACT ties at the max value: {exact_ties}")

    sample_and_diagnose("midnight-peakers with full_peak_idx == 0", jan1_01)
    sample_and_diagnose("midnight-peakers with full_peak_idx != 0", other)

    # Overall distribution stat: of ALL buildings, how many have an exact tie
    # at the max value with at least one other hour?
    print(f"\n{'=' * 72}")
    print("  Overall: how many buildings have flat-ish profiles?")
    print(f"{'=' * 72}")
    print("  (sampling 100 buildings across all groups for this scan)")
    scan = peaks.sample(min(100, len(peaks)),
                        random_state=rng.integers(1e9))
    flat_count = 0
    multi_max = 0
    for _, row in scan.iterrows():
        buy = load_buy(int(row["bldg_id"]))
        if buy is None:
            continue
        mx = buy.max()
        if mx == 0:
            continue
        n_within_1pct = int(np.sum(buy >= mx * 0.99))
        n_exact = int(np.sum(buy == mx))
        if n_within_1pct >= 100:
            flat_count += 1
        if n_exact > 1:
            multi_max += 1
    print(f"  buildings with >=100 hours within 1% of max: {flat_count} / 100")
    print(f"  buildings with multiple hours at EXACT max: {multi_max} / 100")


if __name__ == "__main__":
    main()