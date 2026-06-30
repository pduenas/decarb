import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Tuple, Dict
from pathlib import Path
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")


def fahrenheit_to_celsius(value: float) -> float:
    return (value - 32) * 5 / 9


def load_and_preprocess_parquet(file_path: str, available_bldg_ids: list) -> pd.DataFrame:
    building_data = pd.read_parquet(file_path, engine="fastparquet")

    col_list = [
        "bldg_id",
        "in.heating_setpoint",
        "in.heating_setpoint_has_offset",
        "in.heating_setpoint_offset_magnitude",
        "in.heating_setpoint_offset_period",
        "in.cooling_setpoint",
        "in.cooling_setpoint_has_offset",
        "in.cooling_setpoint_offset_magnitude",
        "in.cooling_setpoint_offset_period",
    ]

    building_data = building_data[col_list]
    building_data.set_index("bldg_id", inplace=True)
    building_data = building_data.loc[building_data.index.isin(available_bldg_ids)]
    print("Done preprocessing buildings")

    return building_data


def get_base_setpoints(building_id: int, building_data: pd.DataFrame) -> Tuple[int, int, int]:
    row = building_data.loc[building_id]

    heating_setpoint = int(row["in.heating_setpoint"].replace("F", ""))
    cooling_setpoint = int(row["in.cooling_setpoint"].replace("F", ""))

    if heating_setpoint > cooling_setpoint:
        avg = (heating_setpoint + cooling_setpoint) / 2
        heating_setpoint = avg
        cooling_setpoint = avg
        shoulder_setpoint = avg
    else:
        shoulder_setpoint = (heating_setpoint + cooling_setpoint) / 2

    return heating_setpoint, cooling_setpoint, shoulder_setpoint


def generate_building_schedule(mode: str, building_id: int, building_data: pd.DataFrame) -> np.ndarray:
    building_data = building_data[building_data.index == building_id]

    night_start = 22
    night_end = 7
    day_start = 9
    day_end = 17

    schedule = np.zeros(24)

    has_offset = (
        "in.heating_setpoint_has_offset"
        if mode == "heating"
        else "in.cooling_setpoint_has_offset"
    )
    offset_magnitude = (
        "in.heating_setpoint_offset_magnitude"
        if mode == "heating"
        else "in.cooling_setpoint_offset_magnitude"
    )
    offset_period = (
        "in.heating_setpoint_offset_period"
        if mode == "heating"
        else "in.cooling_setpoint_offset_period"
    )

    if building_data[has_offset].item() == "Yes":
        offset_magnitude = int(building_data[offset_magnitude].item().replace("F", ""))
        offset_period = building_data[offset_period].item()

        if offset_period[-1] == "h":
            offset_period_number = int(offset_period[-3:-1])
        else:
            offset_period_number = 0

        if "Night" in offset_period:
            night_start += offset_period_number
            night_end += offset_period_number

            if night_start >= 24:
                night_start = night_start % 24
                schedule[night_start:night_end] = 1
            else:
                schedule[night_start:] = 1
                schedule[:night_end] = 1

        if "Day" in offset_period:
            day_start += offset_period_number
            day_end += offset_period_number

            if day_start > day_end:
                schedule[day_start:] = 1
                schedule[:day_end] = 1
            else:
                schedule[day_start:day_end] = 1

        schedule *= offset_magnitude

    return schedule


def generate_building_setpoint_timeseries(
    mode: str,
    building_id: int,
    building_data: pd.DataFrame,
    freq: str,
    base_setpoint: float,
) -> pd.DataFrame:

    timeseries = pd.date_range(start="2018-01-01", end="2018-12-31", freq=freq)
    timeseries = pd.DataFrame(timeseries, columns=["timestamp"])
    timeseries["setpoint"] = base_setpoint

    if mode in ("heating", "cooling"):
        schedule = generate_building_schedule(mode, building_id, building_data)
        timeseries["setpoint"] = base_setpoint + schedule[timeseries["timestamp"].dt.hour.values]

    if mode == "heating":
        timeseries["min_comfort_temp"] = timeseries["setpoint"]
        timeseries["max_comfort_temp"] = timeseries["setpoint"] + 1.8
    elif mode == "cooling":
        timeseries["min_comfort_temp"] = timeseries["setpoint"] - 1.8
        timeseries["max_comfort_temp"] = timeseries["setpoint"]
    else:
        timeseries["min_comfort_temp"] = timeseries["setpoint"] - 0.9
        timeseries["max_comfort_temp"] = timeseries["setpoint"] + 0.9

    for col in ["setpoint", "min_comfort_temp", "max_comfort_temp"]:
        timeseries[col] = (timeseries[col] - 32) * 5 / 9

    return timeseries


def generate_full_setpoint_timeseries(building_data: pd.DataFrame, freq: str) -> Dict[int, Dict[str, pd.DataFrame]]:

    full_timeseries_dataset = {}

    for i in range(len(building_data)):
        building_id = building_data.index[i]

        heating_sp, cooling_sp, shoulder_sp = get_base_setpoints(
            building_id, building_data
        )

        heating_ts = generate_building_setpoint_timeseries(
            "heating", building_id, building_data, freq, heating_sp
        )
        cooling_ts = generate_building_setpoint_timeseries(
            "cooling", building_id, building_data, freq, cooling_sp
        )
        shoulder_ts = generate_building_setpoint_timeseries(
            "shoulder", building_id, building_data, freq, shoulder_sp
        )

        heating_ts.set_index("timestamp", inplace=True)
        cooling_ts.set_index("timestamp", inplace=True)
        shoulder_ts.set_index("timestamp", inplace=True)

        full_timeseries_dataset[building_id] = {
            "heating": heating_ts,
            "cooling": cooling_ts,
            "shoulder": shoulder_ts,
        }

    return full_timeseries_dataset


def compute_statistical_comfort_envelope(
    building_id: int,
    combined_ts: pd.DataFrame,
    consumption_path: Path,
    window_days: int = 30,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> pd.DataFrame:

    consumption_file = consumption_path / f"{building_id}-0.parquet"
    consumption_data = pd.read_parquet(consumption_file, engine="fastparquet")
    consumption_data["timestamp"] = pd.to_datetime(consumption_data["timestamp"])
    consumption_data.set_index("timestamp", inplace=True)

    indoor_temp = consumption_data[
        "out.indoor_temperature.conditioned_space..c"
    ].reindex(combined_ts.index)

    freq = pd.infer_freq(combined_ts.index)
    samples_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    window_size = window_days * samples_per_day

    rolling_max = indoor_temp.rolling(window=window_size, min_periods=1).quantile(upper_q)
    rolling_min = indoor_temp.rolling(window=window_size, min_periods=1).quantile(lower_q)

    # Stabilize initial window
    first_max = rolling_max.iloc[window_size - 1]
    first_min = rolling_min.iloc[window_size - 1]
    rolling_max.iloc[:window_size] = first_max
    rolling_min.iloc[:window_size] = first_min

    # --- Enforce minimum comfort band ONLY if envelope is too narrow ---
    min_band_width = 1
    min_half_width = 0.5

    band_width = rolling_max - rolling_min
    narrow_mask = band_width < min_band_width

    adjusted_max = rolling_max.copy()
    adjusted_min = rolling_min.copy()

    adjusted_max[narrow_mask] = np.maximum(
        rolling_max[narrow_mask],
        indoor_temp[narrow_mask] + min_half_width
    )

    adjusted_min[narrow_mask] = np.minimum(
        rolling_min[narrow_mask],
        indoor_temp[narrow_mask] - min_half_width
    )

    # --- Round to nearest 0.5°C increments ---
    adjusted_min = np.floor(adjusted_min * 2) / 2
    adjusted_max = np.ceil(adjusted_max * 2) / 2

    combined_ts = combined_ts.copy()
    combined_ts["min_comfort_temp"] = adjusted_min
    combined_ts["max_comfort_temp"] = adjusted_max

    return combined_ts


def process_single_building(
    building_id: int,
    building_data: pd.DataFrame,
    schedule_path: Path,
    consumption_path: Path,
    save_path: Path,
    ts_resolution: str,
    test_mode: bool,
) -> Tuple[int, pd.DataFrame]:
    """
    Full pipeline for a single building: generate setpoint timeseries,
    combine with season schedule, compute comfort envelope, and optionally save.
    Returns (building_id, combined_df).
    """
    heating_sp, cooling_sp, shoulder_sp = get_base_setpoints(building_id, building_data)

    heating_ts = generate_building_setpoint_timeseries("heating", building_id, building_data, ts_resolution, heating_sp)
    cooling_ts = generate_building_setpoint_timeseries("cooling", building_id, building_data, ts_resolution, cooling_sp)
    shoulder_ts = generate_building_setpoint_timeseries("shoulder", building_id, building_data, ts_resolution, shoulder_sp)

    heating_ts.set_index("timestamp", inplace=True)
    cooling_ts.set_index("timestamp", inplace=True)
    shoulder_ts.set_index("timestamp", inplace=True)

    mode_ts = {"heating": heating_ts, "cooling": cooling_ts, "shoulder": shoulder_ts}

    # Load season schedule
    season_schedule = pd.read_parquet(
        schedule_path / f"schedule_{building_id}.parquet",
        engine="fastparquet"
    )
    season_schedule["timestamp"] = pd.to_datetime(season_schedule["timestamp"])
    season_schedule.set_index("timestamp", inplace=True)

    # Reindex to schedule index
    mode_ts = {k: v.reindex(season_schedule.index) for k, v in mode_ts.items()}

    # Combine setpoints by season schedule
    schedule_to_mode = {1: "heating", 0: "shoulder", -1: "cooling"}
    combined = pd.DataFrame(index=season_schedule.index)
    combined["mode"] = season_schedule["schedule"]

    for mode_value, mode_name in schedule_to_mode.items():
        mask = season_schedule["schedule"] == mode_value
        combined.loc[mask, "setpoint_temp"] = mode_ts[mode_name].loc[mask, "setpoint"]
        combined.loc[mask, "min_comfort_temp"] = mode_ts[mode_name].loc[mask, "min_comfort_temp"]
        combined.loc[mask, "max_comfort_temp"] = mode_ts[mode_name].loc[mask, "max_comfort_temp"]

    # Compute statistical comfort envelope
    combined = compute_statistical_comfort_envelope(building_id, combined, consumption_path)
    # Add ResStock-style comfort band (±1°C around setpoint)
    combined["min_resstock_temp"] = combined["setpoint_temp"] - 1.0
    combined["max_resstock_temp"] = combined["setpoint_temp"] + 1.0

    # Save if not in test mode
    if not test_mode:
        combined.to_parquet(save_path / f"setpoint_{building_id}.parquet", engine="pyarrow")

    return building_id, combined


def generate_temperature_profiles(
    combine: bool = True,
    ts_resolution: str = "15min",
    test_mode: bool = False,
    skip_existing: bool = True,
    max_workers: int = None,
) -> dict:
    """
    Generate setpoint temperature timeseries for all buildings in parallel.
    Returns a dict: {building_id: combined_timeseries_df}

    Args:
        combine: Whether to combine heating/cooling/shoulder modes
        ts_resolution: Time resolution for timeseries (default: "15min")
        test_mode: If True, only process 5 random buildings
        skip_existing: If True, skip buildings that already have output files
        max_workers: Number of parallel workers. Defaults to min(24, os.cpu_count()).
    """
    # --- Paths ---
    current_path = Path(__file__).resolve()
    parent_path = current_path.parents[1]
    input_path = parent_path / "in"
    save_path = parent_path / "out" / "setpoint_timeseries_test_2"
    save_path.mkdir(parents=True, exist_ok=True)
    schedule_path = parent_path / "out" / "schedules_test"
    consumption_path = parent_path / "out" / "consumption_files"

    # --- Available building IDs ---
    available_bldg_ids = [
        int(f.stem.replace("schedule_", ""))
        for f in sorted(schedule_path.glob("schedule_*.parquet"))
    ]

    if test_mode:
        rng = np.random.default_rng(seed=42)
        available_bldg_ids = list(
            rng.choice(available_bldg_ids, size=100, replace=False)
        )

    # --- Filter out existing files if skip_existing is True ---
    if skip_existing and not test_mode:
        existing_files = {
            int(f.stem.replace("setpoint_", ""))
            for f in save_path.glob("setpoint_*.parquet")
        }
        original_count = len(available_bldg_ids)
        available_bldg_ids = [
            bid for bid in available_bldg_ids if bid not in existing_files
        ]
        skipped_count = original_count - len(available_bldg_ids)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} buildings with existing output files")
        if len(available_bldg_ids) == 0:
            print("All buildings already processed. Returning empty dict.")
            return {}

    # --- Load building metadata ---
    metadata_path = input_path / "TX_upgrade0.parquet"
    building_data = load_and_preprocess_parquet(metadata_path, available_bldg_ids)

    if not combine:
        return generate_full_setpoint_timeseries(building_data, ts_resolution)

    # --- Parallel processing ---
    n_workers = min(max_workers or 24, os.cpu_count() or 1, len(available_bldg_ids))
    print(f"Processing {len(available_bldg_ids)} buildings with {n_workers} workers...")

    full_timeseries_dict = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_single_building,
                building_id,
                building_data,
                schedule_path,
                consumption_path,
                save_path,
                ts_resolution,
                test_mode,
            ): building_id
            for building_id in available_bldg_ids
        }

        for future in as_completed(futures):
            building_id = futures[future]
            try:
                bid, combined = future.result()
                full_timeseries_dict[bid] = combined
                completed += 1
                if completed % 100 == 0 or completed == len(available_bldg_ids):
                    print(f"  {completed}/{len(available_bldg_ids)} buildings complete")
            except Exception as e:
                print(f"  ERROR processing building {building_id}: {e}")
    return full_timeseries_dict


def test_plot_setpoints(timeseries_dict: Dict[int, pd.DataFrame], num_buildings: int = 3):
    """Plots indoor temperature as scatter points and comfort temperature bands
    for the first N buildings, and prints the percentage of indoor temp datapoints
    that fall within the comfort envelope.
    """
    current_path = Path(__file__).resolve()
    parent_path = current_path.parents[1]
    consumption_path = parent_path / "out" / "consumption_files"

    building_ids = list(timeseries_dict.keys())[:num_buildings]

    fig, axes = plt.subplots(
        num_buildings, 1, figsize=(14, 4 * num_buildings), sharex=True
    )
    if num_buildings == 1:
        axes = [axes]

    for ax, building_id in zip(axes, building_ids):
        df = timeseries_dict[building_id]

        # Load indoor temperature
        consumption_file = consumption_path / f"{building_id}-0.parquet"
        consumption_data = pd.read_parquet(consumption_file, engine="fastparquet")
        consumption_data["timestamp"] = pd.to_datetime(consumption_data["timestamp"])
        consumption_data.set_index("timestamp", inplace=True)

        indoor_temp = consumption_data[
            "out.indoor_temperature.conditioned_space..c"
        ].reindex(df.index)

        # Calculate comfort statistics
        valid = indoor_temp.dropna()
        envelope = df.loc[valid.index]
        within = (
            (valid >= envelope["min_comfort_temp"])
            & (valid <= envelope["max_comfort_temp"])
        ).sum()
        pct_within = (within / len(valid)) * 100

        print(
            f"Building {building_id}: "
            f"{pct_within:.1f}% of indoor temp datapoints within comfort envelope"
        )

        # --- Plot ---
        ax.fill_between(
            df.index,
            df["min_comfort_temp"],
            df["max_comfort_temp"],
            color="steelblue",
            alpha=0.25,
            label="Comfort band",
        )
        
        ax.fill_between(
            df.index,
            df["min_resstock_temp"],
            df["max_resstock_temp"],
            color="orange",
            alpha=0.15,
            label="ResStock band",
        )
                
        ax.plot(df.index, df["min_comfort_temp"], color="steelblue", linewidth=0.6, linestyle="--")
        ax.plot(df.index, df["max_comfort_temp"], color="steelblue", linewidth=0.6, linestyle="--")

        ax.scatter(
            indoor_temp.index,
            indoor_temp.values,
            s=2,
            alpha=0.4,
            color="black",
            label="Indoor temperature",
        )

        ax.plot(
            df.index,
            df["setpoint_temp"],
            color="black",
            linewidth=1.0,
            alpha=0.7,
            label="Setpoint",
        )

        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"Building {building_id}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", markerscale=4)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlabel("Month")

    plt.tight_layout()
    plt.savefig("test_setpoints_plot.png", dpi=150)
    plt.show()

def save_comfort_success_summary(full_timeseries_dict: dict, consumption_path: Path, save_path: Path):
    """
    Calculates the success rate of indoor temperatures within comfort bands
    (statistical and ResStock ±1°C) for all buildings and saves a summary CSV.
    """
    summary_records = []

    for bid, df in full_timeseries_dict.items():
        # Load indoor temperature
        consumption_file = consumption_path / f"{bid}-0.parquet"
        consumption_data = pd.read_parquet(consumption_file, engine="fastparquet")
        consumption_data["timestamp"] = pd.to_datetime(consumption_data["timestamp"])
        consumption_data.set_index("timestamp", inplace=True)
        indoor_temp = consumption_data["out.indoor_temperature.conditioned_space..c"].reindex(df.index).dropna()

        # Calculate success rates
        stat_within = ((indoor_temp >= df.loc[indoor_temp.index, "min_comfort_temp"]) &
                       (indoor_temp <= df.loc[indoor_temp.index, "max_comfort_temp"])).mean() * 100
        resstock_within = ((indoor_temp >= df.loc[indoor_temp.index, "min_resstock_temp"]) &
                           (indoor_temp <= df.loc[indoor_temp.index, "max_resstock_temp"])).mean() * 100

        summary_records.append({
            "building_id": bid,
            "statistical_comfort_success_%": stat_within,
            "resstock_comfort_success_%": resstock_within
        })

    summary_df = pd.DataFrame(summary_records)
    summary_csv_file = save_path / "comfort_success_summary.csv"
    summary_df.to_csv(summary_csv_file, index=False)
    print(f"Saved comfort success summary for {len(summary_df)} buildings at {summary_csv_file}")
    
    
if __name__ == "__main__":
    test_mode_flag = True
    result = generate_temperature_profiles(
        test_mode=test_mode_flag,
        skip_existing=True,  # Set to False to force reprocessing
        max_workers=24,
    )
    if test_mode_flag:
        test_plot_setpoints(result)
        
    current_path = Path(__file__).resolve()
    parent_path = current_path.parents[1]
    consumption_path = parent_path / "out" / "consumption_files"
    save_path = parent_path / "out" / "setpoint_timeseries_test_2"

    save_comfort_success_summary(result, consumption_path, save_path)