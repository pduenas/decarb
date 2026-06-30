import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Tuple, Dict
from pathlib import Path
import os
import warnings

warnings.filterwarnings("ignore")

def fahrenheit_to_celsius(value: float) -> float:
    """Converts a temperature value from Fahrenheit to Celsius"""
    return (value - 32) * 5 / 9


def generate_building_schedule(mode: str, building_id: int, building_data: pd.DataFrame) -> np.ndarray:
    """Generates a building's setpoint offset schedule

    Args:
        mode (str): Either heating or cooling
        building_id (int): ID of the building
        building_data (pd.DataFrame): Heating/cooling setpoint dataset

    Returns:
        np.ndarray: Building's schedule during the day (when to apply offset)
    """
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
                schedule[night_start:]= 1
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


def get_base_setpoints(building_id: int, building_data: pd.DataFrame) -> Tuple[int, int, int]:
    """Calculates the base heating, cooling, and shoulder setpoints for a building,
    applying the sanity check that heating setpoint should not exceed cooling setpoint.

    Args:
        building_id (int): ID of the building
        building_data (pd.DataFrame): Building metadata

    Returns:
        Tuple[int, int, int]: (heating_setpoint, cooling_setpoint, shoulder_setpoint)
    """
    row = building_data.loc[building_id]

    heating_setpoint = int(row["in.heating_setpoint"].replace("F", ""))
    cooling_setpoint = int(row["in.cooling_setpoint"].replace("F", ""))

    # Sanity check: if heating > cooling, set both to their average
    if heating_setpoint > cooling_setpoint:
        avg = (heating_setpoint + cooling_setpoint) / 2
        heating_setpoint = avg
        cooling_setpoint = avg
        shoulder_setpoint = avg
    else:
        # Shoulder is the average of heating and cooling
        shoulder_setpoint = (heating_setpoint + cooling_setpoint) / 2

    return heating_setpoint, cooling_setpoint, shoulder_setpoint


def generate_building_setpoint_timeseries(mode: str, building_id: int, building_data: pd.DataFrame, freq: str, base_setpoint: float) -> pd.DataFrame:
    """Generates a building's setpoint timeseries for a given mode, including
    min and max comfort temperature bands. All values are computed in °F first,
    then converted to °C before returning.

    Comfort bands (in °F, before conversion to °C):
        - Heating (winter): min = setpoint, max = setpoint + 1.8 (i.e. +1°C)
        - Cooling (summer): min = setpoint - 1.8, max = setpoint (i.e. -1°C)
        - Shoulder:         min = setpoint - 0.9, max = setpoint + 0.9 (i.e. ±0.5°C)

    Args:
        mode (str): One of 'heating', 'cooling', or 'shoulder'
        building_id (int): ID of the building
        building_data (pd.DataFrame): Building metadata
        freq (str): Timeseries frequency
        base_setpoint (float): The base setpoint value to use (in °F)

    Returns:
        pd.DataFrame: Building's setpoint timeseries with columns:
            setpoint, min_comfort_temp, max_comfort_temp (all in °C)
    """
    # Generate empty timeseries for a full year (2018)
    timeseries = pd.date_range(start="2018-01-01", end="2018-12-31", freq=freq)
    timeseries = pd.DataFrame(timeseries, columns=["timestamp"])
    timeseries["setpoint"] = base_setpoint

    # Shoulder has no offset, so only apply schedule for heating/cooling
    if mode in ("heating", "cooling"):
        schedule = generate_building_schedule(mode, building_id, building_data)

        def apply_offset(row, schedule):
            hour = row["timestamp"].hour
            return row["setpoint"] + schedule[hour]

        timeseries["setpoint"] = timeseries.apply(apply_offset, axis=1, schedule=schedule)

    # Compute comfort bands in °F based on mode
    # 1.8°F = 1°C, so bands map to ±1°C for heating/cooling and ±0.5°C for shoulder
    if mode == "heating":
        timeseries["min_comfort_temp"] = timeseries["setpoint"]
        timeseries["max_comfort_temp"] = timeseries["setpoint"] + 1.8
    elif mode == "cooling":
        timeseries["min_comfort_temp"] = timeseries["setpoint"] - 1.8
        timeseries["max_comfort_temp"] = timeseries["setpoint"]
    else:  # shoulder
        timeseries["min_comfort_temp"] = timeseries["setpoint"] - 0.9
        timeseries["max_comfort_temp"] = timeseries["setpoint"] + 0.9

    # Convert all temperature columns from °F to °C
    for col in ["setpoint", "min_comfort_temp", "max_comfort_temp"]:
        timeseries[col] = timeseries[col].apply(fahrenheit_to_celsius)

    return timeseries


def load_and_preprocess_parquet(file_path: str, available_bldg_ids: list) -> pd.DataFrame:
    """Load data from a path in parquet format, filtered to only buildings
    that have a corresponding season schedule.

    Args:
        file_path (str): path to the parquet file
        available_bldg_ids (list): list of building IDs that have season schedules

    Returns:
        pd.DataFrame: DataFrame containing building metadata
    """
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

    return building_data


def generate_full_setpoint_timeseries(building_data: pd.DataFrame, freq: str) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Generates heating, cooling, and shoulder setpoint timeseries for each building

    Args:
        building_data (pd.DataFrame): Building metadata
        freq (str): Timeseries frequency

    Returns:
        Dict[int, Dict[str, pd.DataFrame]]: Nested dictionary keyed by building ID,
            containing 'heating', 'cooling', and 'shoulder' timeseries DataFrames
    """
    full_timeseries_dataset = {}

    for i in range(len(building_data)):
        building_id = building_data.index[i]
        print(
            f"Generating setpoint timeseries for building {i + 1}/{len(building_data)}...          ",
            end="\r",
        )

        heating_sp, cooling_sp, shoulder_sp = get_base_setpoints(building_id, building_data)

        heating_ts = generate_building_setpoint_timeseries("heating", building_id, building_data, freq, heating_sp)
        cooling_ts = generate_building_setpoint_timeseries("cooling", building_id, building_data, freq, cooling_sp)
        shoulder_ts = generate_building_setpoint_timeseries("shoulder", building_id, building_data, freq, shoulder_sp)

        heating_ts.set_index("timestamp", inplace=True)
        cooling_ts.set_index("timestamp", inplace=True)
        shoulder_ts.set_index("timestamp", inplace=True)

        full_timeseries_dataset[building_id] = {
            "heating": heating_ts,
            "cooling": cooling_ts,
            "shoulder": shoulder_ts,
        }

    return full_timeseries_dataset


def generate_setpoint_timeseries(
    combine: bool = True,
    ts_resolution: str = "15min",
    test_mode: bool = False) -> Dict[int, pd.DataFrame]:
    
    """Main function to generate setpoint timeseries for heating, cooling, and shoulder for each building.
    When combine=True, loads each building's season schedule and uses it to select the appropriate
    mode (heating/cooling/shoulder) per timestamp, producing a single combined timeseries per building.

    Season schedule values: 1 = heating (winter), 0 = shoulder, -1 = cooling (summer)

    Args:
        combine (bool): Whether to combine heating, cooling, and shoulder setpoint timeseries
                        using per-building season schedules
        ts_resolution (str): Timeseries resolution
        test_mode (bool): If True, only processes the first 5 buildings and skips saving output files

    Returns:
        Dict[int, pd.DataFrame]: Dictionary of DataFrames, where each key is the building ID
                                 and the value is the combined setpoint timeseries with columns:
                                 setpoint_temp, min_comfort_temp, max_comfort_temp (all in °C)
    """

    current_path = Path(__file__).resolve()
    parent_path = current_path.parents[1]
    input_path = parent_path / "in"
    save_path = parent_path / "out" / "setpoint_timeseries"
    schedule_path = parent_path / "out" / "schedules"

    # Scan schedule folder for available building IDs
    available_bldg_ids = [
        int(f.stem.replace("schedule_", ""))
        for f in sorted(schedule_path.glob("schedule_*.parquet"))
    ]

    # In test mode, randomly pick 5 available buildings
    if test_mode:
        rng = np.random.default_rng(seed=42)
        available_bldg_ids = list(rng.choice(available_bldg_ids, size=5, replace=False))

    print(f"Found {len(available_bldg_ids)} buildings with schedules")

    if not test_mode:
        if not os.path.exists(save_path):
            os.makedirs(save_path)    
    
    metadata_path = input_path / "TX_upgrade0.parquet"
    building_data = load_and_preprocess_parquet(metadata_path, available_bldg_ids)

    print("Generating setpoint timeseries...")
    timeseries_dict = generate_full_setpoint_timeseries(building_data, ts_resolution)
    print("Done")

    if combine:
        print("Combining timeseries using per-building season schedules...")

        # Map schedule values to mode keys
        schedule_to_mode = {
            1: "heating",
            0: "shoulder",
            -1: "cooling",
        }

        full_timeseries_dict = {}

        for j, building_id in enumerate(timeseries_dict):
            print(
                f"Processing building {j}/{len(timeseries_dict)}...          ",
                end="\r",
            )

            # Load the per-building season schedule
            season_schedule = pd.read_parquet(
                schedule_path / f"schedule_{building_id}.parquet",
                engine="fastparquet"
            )
            season_schedule["timestamp"] = pd.to_datetime(season_schedule["timestamp"])
            season_schedule.set_index("timestamp", inplace=True)

            # Reindex mode timeseries to match the schedule's index
            # (handles any edge differences, e.g. schedule running to 2019-01-01 00:00:00)
            mode_timeseries = {
                mode_name: ts.reindex(season_schedule.index)
                for mode_name, ts in timeseries_dict[building_id].items()
            }

            # Build the combined timeseries by selecting values based on season
            combined = pd.DataFrame(index=season_schedule.index)
            for mode_value, mode_name in schedule_to_mode.items():
                mask = season_schedule["schedule"] == mode_value
                combined.loc[mask, "setpoint_temp"] = mode_timeseries[mode_name].loc[mask, "setpoint"]
                combined.loc[mask, "min_comfort_temp"] = mode_timeseries[mode_name].loc[mask, "min_comfort_temp"]
                combined.loc[mask, "max_comfort_temp"] = mode_timeseries[mode_name].loc[mask, "max_comfort_temp"]

            # Save to parquet (skip in test mode)
            if not test_mode:
                save_filename = save_path / f"setpoint_{building_id}.parquet"
                combined.to_parquet(save_filename, engine="fastparquet")

            full_timeseries_dict[building_id] = combined

        return full_timeseries_dict
    else:
        return timeseries_dict


def test_plot_setpoints(timeseries_dict: Dict[int, pd.DataFrame], num_buildings: int = 3):
    """Plots the setpoint and comfort temperature bands for the first N buildings,
    and prints the percentage of indoor temp datapoints that fall within the comfort envelope.

    Args:
        timeseries_dict (Dict[int, pd.DataFrame]): Combined timeseries dictionary from generate_setpoint_timeseries()
        num_buildings (int): Number of buildings to plot (default 5)
    """
    current_path = Path(__file__).resolve()
    parent_path = current_path.parents[1]
    consumption_path = parent_path / "out" / "consumption_files"

    building_ids = list(timeseries_dict.keys())[:num_buildings]

    fig, axes = plt.subplots(num_buildings, 1, figsize=(14, 4 * num_buildings), sharex=True)
    if num_buildings == 1:
        axes = [axes]

    for ax, building_id in zip(axes, building_ids):
        df = timeseries_dict[building_id]

        # Load indoor temperature from consumption file to calculate comfort stats
        consumption_file = consumption_path / f"{building_id}-0.parquet"
        consumption_data = pd.read_parquet(consumption_file, engine="fastparquet")
        consumption_data["timestamp"] = pd.to_datetime(consumption_data["timestamp"])
        consumption_data.set_index("timestamp", inplace=True)
        indoor_temp = consumption_data["out.indoor_temperature.conditioned_space..c"]

        # Align indoor temp to the envelope's index
        indoor_temp = indoor_temp.reindex(df.index)

        # Calculate and print percentage of indoor temp datapoints within the comfort envelope
        valid = indoor_temp.dropna()
        envelope = df.loc[valid.index]
        within = ((valid >= envelope["min_comfort_temp"]) & (valid <= envelope["max_comfort_temp"])).sum()
        pct_within = (within / len(valid)) * 100
        print(f"Building {building_id}: {pct_within:.1f}% of indoor temp datapoints within comfort envelope")

        # Plot setpoint and comfort band
        ax.plot(df.index, df["setpoint_temp"], color="black", linewidth=1, label="Setpoint")
        ax.fill_between(df.index, df["min_comfort_temp"], df["max_comfort_temp"],
                        color="steelblue", alpha=0.3, label="Comfort band")
        ax.plot(df.index, df["min_comfort_temp"], color="steelblue", linewidth=0.7, linestyle="--")
        ax.plot(df.index, df["max_comfort_temp"], color="steelblue", linewidth=0.7, linestyle="--")

        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"Building {building_id}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    # Format x-axis to show months
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlabel("Month")

    plt.tight_layout()
    plt.savefig("test_setpoints_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    test_mode_flag = True

    result = generate_setpoint_timeseries(test_mode=test_mode_flag)

    if test_mode_flag:
        test_plot_setpoints(result)