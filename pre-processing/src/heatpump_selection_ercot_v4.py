from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEEP_CAPACITY_COLS = [
    "Minimum Capacity 5°F",
    "Rated Capacity 5°F",
    "Maximum Capacity 5°F",
    "Minimum Capacity 17°F",
    "Rated Capacity 17°F",
    "Maximum Capacity 17°F",
    "Minimum Capacity 47°F",
    "Rated Capacity 47°F",
    "Maximum Capacity 47°F",
    "Minimum Capacity 82°F",
    "Rated Capacity 82°F",
    "Maximum Capacity 82°F",
    "Minimum Capacity 95°F",
    "Rated Capacity 95°F",
    "Maximum Capacity 95°F",
]

NEEP_COP_COLS = [
    "COP at Min. Capacity 5°F",
    "COP at Rated Capacity 5°F",
    "COP at Max. Capacity 5°F",
    "COP at Min. Capacity 17°F",
    "COP at Rated Capacity 17°F",
    "COP at Max. Capacity 17°F",
    "COP at Min. Capacity 47°F",
    "COP at Rated Capacity 47°F",
    "COP at Max. Capacity 47°F",
    "COP at Min. Capacity 82°F",
    "COP at Rated Capacity 82°F",
    "COP at Max. Capacity 82°F",
    "COP at Min. Capacity 95°F",
    "COP at Rated Capacity 95°F",
    "COP at Max. Capacity 95°F",
]

NEEP_REQUIRED_COLS = [
    "Brand Name",
    "Status",
    "HSPF (Region IV)",
] + NEEP_CAPACITY_COLS + NEEP_COP_COLS

METADATA_COLS = [
    "bldg_id",
    "in.sqft..ft2",
    "in.hvac_has_ducts",
    "in.hvac_cooling_efficiency",
    "in.hvac_cooling_type",
    "in.hvac_heating_efficiency",
    "in.hvac_heating_type",
    "in.hvac_heating_type_and_fuel",
]

CONSUMPTION_COLS = [
    "out.load.cooling.energy_delivered..kbtu",
    "out.load.heating.energy_delivered..kbtu",
    "out.indoor_operative_temperature.conditioned_space..c",
    "out.outdoor_air_drybulb_temp..c",
]


# ---------------------------------------------------------------------------
# NEEP loading
# ---------------------------------------------------------------------------

def load_neep_heat_pumps(neep_filepath: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Load and filter the NEEP database for eligible Mitsubishi heat pumps.

    Filters:
      - 'Brand Name' must contain 'mitsubishi' (case-insensitive)
      - 'Status' must equal 'Live' (case-insensitive)
      - 'HSPF (Region IV)' must be populated
      - At least one capacity column must be populated (not all empty)

    Parameters
    ----------
    neep_filepath : Path
        Path to the NEEP database file (CSV or Excel).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe of eligible heat pump units.
    """
    suffix = neep_filepath.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(neep_filepath, low_memory=False)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(neep_filepath)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .csv or .xlsx/.xls.")

    missing = [col for col in NEEP_REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"The following required columns are missing from the NEEP database: {missing}")

    df = df[df["Brand Name"].str.contains("mitsubishi", case=False, na=False)]
    df = df[df["Status"].str.strip().str.lower() == "live"]
    df = df.dropna(subset=["HSPF (Region IV)"])
    df = df.dropna(subset=NEEP_CAPACITY_COLS, how="all")
    df = df.reset_index(drop=True)
    df = clean_neep_data(df)

    # Exclude units with any COP value exceeding 6 (data quality filter)
    cop_rated_cols = [c for c in NEEP_COP_COLS if "Rated" in c and c in df.columns]
    cop_mask = (df[cop_rated_cols] > 6).any(axis=1)
    n_excluded = cop_mask.sum()
    df = df[~cop_mask].reset_index(drop=True)

    if verbose:
        print(f"NEEP filter: {len(df)} eligible Mitsubishi Live units found ({n_excluded} excluded for COP > 6).")

    return df


# ---------------------------------------------------------------------------
# NEEP data cleaning
# ---------------------------------------------------------------------------

NEEP_TEMP_TIERS = ["5°F", "17°F", "47°F", "82°F", "95°F"]

def clean_neep_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each temperature tier, fill missing Rated Capacity and COP at Rated Capacity
    with the arithmetic mean of the corresponding Min and Max columns.

    Applied after Mitsubishi/Live filtering so all downstream code can use
    the rated columns directly without null checks.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered NEEP dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with imputed rated capacity and COP columns.
    """
    df = df.copy()

    for tier in NEEP_TEMP_TIERS:
        cap_rated = f"Rated Capacity {tier}"
        cap_min   = f"Minimum Capacity {tier}"
        cap_max   = f"Maximum Capacity {tier}"
        if all(c in df.columns for c in [cap_rated, cap_min, cap_max]):
            mask = df[cap_rated].isna()
            df.loc[mask, cap_rated] = (df.loc[mask, cap_min] + df.loc[mask, cap_max]) / 2

        cop_rated = f"COP at Rated Capacity {tier}"
        cop_min   = f"COP at Min. Capacity {tier}"
        cop_max   = f"COP at Max. Capacity {tier}"
        if all(c in df.columns for c in [cop_rated, cop_min, cop_max]):
            mask = df[cop_rated].isna()
            df.loc[mask, cop_rated] = (df.loc[mask, cop_min] + df.loc[mask, cop_max]) / 2

    return df


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(metadata_filepath: Path) -> pd.DataFrame:
    """
    Load the full building metadata parquet file once.

    Parameters
    ----------
    metadata_filepath : Path
        Path to the metadata parquet file.

    Returns
    -------
    pd.DataFrame
        Full metadata dataframe indexed by bldg_id.
    """
    df = pd.read_parquet(metadata_filepath, engine="fastparquet", columns=METADATA_COLS)
    df = df.set_index("bldg_id")
    return df


def get_building_metadata(metadata_df: pd.DataFrame, building_id: int) -> dict:
    """
    Extract metadata for a single building from the pre-loaded metadata dataframe.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Full metadata dataframe as returned by load_metadata().
    building_id : int
        The building ID to look up.

    Returns
    -------
    dict
        Dictionary of building characteristics for the specified building.
    """
    if building_id not in metadata_df.index:
        raise ValueError(f"Building ID {building_id} not found in metadata.")

    row = metadata_df.loc[building_id]

    return {
        "bldg_id": building_id,
        "conditioned_area_sqft": float(row["in.sqft..ft2"]),
        "has_ductwork": row["in.hvac_has_ducts"].strip().lower() == "yes",
        "hvac_cooling_efficiency": row["in.hvac_cooling_efficiency"],
        "hvac_cooling_type": row["in.hvac_cooling_type"],
        "hvac_heating_efficiency": row["in.hvac_heating_efficiency"],
        "hvac_heating_type": row["in.hvac_heating_type"],
        "hvac_heating_type_and_fuel": row["in.hvac_heating_type_and_fuel"],
    }


# ---------------------------------------------------------------------------
# Consumption loading
# ---------------------------------------------------------------------------

def load_building_consumption(consumption_dir: Path, building_id: int) -> pd.DataFrame:
    """
    Load the consumption data for a single building.

    Parameters
    ----------
    consumption_dir : Path
        Path to the folder containing per-building consumption parquet files.
    building_id : int
        The building ID to load.

    Returns
    -------
    pd.DataFrame
        Consumption dataframe with only the relevant columns.
    """
    filepath = consumption_dir / f"{building_id}-0.parquet"

    if not filepath.exists():
        raise FileNotFoundError(f"Consumption file not found for building {building_id}: {filepath}")

    df = pd.read_parquet(filepath, engine="fastparquet", columns=CONSUMPTION_COLS)

    return df


# ---------------------------------------------------------------------------
# Capacity selection helper
# ---------------------------------------------------------------------------

def select_capacity(row: pd.Series, outdoor_temp_f: float, ccASHP_df: pd.DataFrame = None) -> pd.Series:
    """
    Select the appropriate capacity rating for a single heat pump unit
    based on the outdoor temperature at peak heating load.

    Parameters
    ----------
    row : pd.Series
        A row from the NEEP dataframe.
    outdoor_temp_f : float
        Outdoor temperature at peak heating load in Fahrenheit.

    Returns
    -------
    pd.Series
        Series with 'capacity_btu' and 'ccASHP' values.
    """
    ccASHP = False

    if outdoor_temp_f > 47:
        capacity = row["Rated Capacity 47°F"]

    elif outdoor_temp_f > 17:
        capacity = row["Rated Capacity 17°F"]

    elif outdoor_temp_f >= 5:
        capacity = row["Rated Capacity 5°F"]

    else:
        # Below 5°F — only consider units in ccASHP_df
        ccASHP = True
        if ccASHP_df is not None and row.name not in ccASHP_df.index:
            # This unit has no optional low temp data — exclude it
            capacity = float("nan")
        else:
            opt_temp = row["Optional Low Temperature Data Outdoor Dry Bulb (°F)"]
            opt_min  = row["Optional Low Temperature Data Minimum Capacity X°F"]
            opt_max  = row["Optional Low Temperature Data Maximum Capacity X°F"]

            if outdoor_temp_f >= opt_temp:
                # Between opt_temp and 5°F — use average of min/max optional
                capacity = (opt_min + opt_max) / 2 if not pd.isna(opt_max) else opt_min
            else:
                # Below opt_temp — use minimum optional capacity
                capacity = opt_min

    return pd.Series({"capacity_btu": capacity, "ccASHP": ccASHP})


# ---------------------------------------------------------------------------
# Cooling capacity selection helper
# ---------------------------------------------------------------------------

def select_cooling_capacity(row: pd.Series, outdoor_temp_f: float) -> float:
    """
    Select the appropriate cooling capacity rating for a single heat pump unit
    based on the outdoor temperature at peak cooling load.

    Logic:
      > 95°F : Minimum Capacity 95°F
      82-95°F: Rated Capacity 95°F (or average of Min/Max if rated is null)
      < 82°F : Rated Capacity 82°F (or average of Min/Max if rated is null)

    Parameters
    ----------
    row : pd.Series
        A row from the NEEP dataframe.
    outdoor_temp_f : float
        Outdoor temperature at peak cooling load in Fahrenheit.

    Returns
    -------
    float
        Selected cooling capacity in BTU/hr.
    """
    if outdoor_temp_f > 95:
        return row["Minimum Capacity 95°F"]

    elif outdoor_temp_f >= 82:
        return row["Rated Capacity 95°F"]

    else:
        return row["Rated Capacity 82°F"]


# ---------------------------------------------------------------------------
# Capacity filtering
# ---------------------------------------------------------------------------

def get_capacity_at_design_temp(
    neep_df: pd.DataFrame,
    consumption_df: pd.DataFrame,
    ccASHP_df: pd.DataFrame = None,
    conditioned_area_sqft: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For each available heat pump, determine the relevant capacity rating based on
    the outdoor temperature at the building's peak heating load, then filter out
    units that cannot meet that load.

    Parameters
    ----------
    neep_df : pd.DataFrame
        Filtered NEEP dataframe of eligible heat pumps.
    consumption_df : pd.DataFrame
        Building consumption dataframe as returned by load_building_consumption().

    Returns
    -------
    pd.DataFrame
        Copy of neep_df with 'capacity_btu', 'ccASHP', and 'sufficient_capacity' columns.
    """
    heating_series = consumption_df["out.load.heating.energy_delivered..kbtu"]
    if heating_series.isna().all() or heating_series.max() == 0:
        # No heating load data — fall back to rule of thumb: 30 BTU/hr per sqft
        peak_load_btu_hr = conditioned_area_sqft * 30
        outdoor_temp_f = consumption_df["out.outdoor_air_drybulb_temp..c"].min() * 1.8 + 32
        print("No heating load found, rule of thumb used to calculate heating capacity requirement.")
    else:
        peak_idx = heating_series.idxmax()
        peak_load_kbtu = consumption_df.loc[peak_idx, "out.load.heating.energy_delivered..kbtu"]
        peak_load_btu_hr = peak_load_kbtu * 4000
        outdoor_temp_c = consumption_df.loc[peak_idx, "out.outdoor_air_drybulb_temp..c"]
        outdoor_temp_f = outdoor_temp_c * 1.8 + 32

    # --- Cooling peak load ---
    cooling_series = consumption_df["out.load.cooling.energy_delivered..kbtu"]
    if cooling_series.isna().all() or cooling_series.max() == 0:
        peak_cooling_btu_hr = conditioned_area_sqft * 30
        cooling_temp_f = consumption_df["out.outdoor_air_drybulb_temp..c"].max() * 1.8 + 32
        if verbose:
            print("No cooling load found, rule of thumb used to calculate cooling capacity requirement.")
    else:
        cool_idx = cooling_series.idxmax()
        peak_cooling_btu_hr = consumption_df.loc[cool_idx, "out.load.cooling.energy_delivered..kbtu"] * 4000
        cooling_temp_f = consumption_df.loc[cool_idx, "out.outdoor_air_drybulb_temp..c"] * 1.8 + 32

    if verbose:
        print(f"Peak heating load: {peak_load_btu_hr:,.0f} BTU/hr at outdoor temp: {outdoor_temp_f:.1f}°F")
        print(f"Peak cooling load: {peak_cooling_btu_hr:,.0f} BTU/hr at outdoor temp: {cooling_temp_f:.1f}°F")

    result = neep_df.copy()
    result[["capacity_btu", "ccASHP"]] = result.apply(
        lambda row: select_capacity(row, outdoor_temp_f, ccASHP_df), axis=1
    )

    result["cooling_capacity_btu"] = result.apply(
        lambda row: select_cooling_capacity(row, cooling_temp_f), axis=1
    )

    result["sufficient_capacity"] = (
        (result["capacity_btu"] > peak_load_btu_hr) &
        (result["cooling_capacity_btu"] > peak_cooling_btu_hr)
    )

    n_sufficient = result["sufficient_capacity"].sum()
    if verbose:
        print(f"{n_sufficient} of {len(result)} heat pump(s) have sufficient heating and cooling capacity.")

    return result, peak_load_btu_hr, outdoor_temp_f, peak_cooling_btu_hr, cooling_temp_f


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def eff_adjustment(capacity_btu: float, hspf: float, conditioned_area_sqft: float) -> float:
    """
    Calculate the efficiency premium on appliance cost.
    9% of total base ApplianceCost per 1 HSPF above/below a central HSPF of 10.
    """
    base_appliance_cost = 933.88 + 0.1480 * capacity_btu + 1.1126 * conditioned_area_sqft
    return base_appliance_cost * 0.09 * (hspf - 10)


def estimate_total_cost(
    capacity_btu: float,
    hspf: float,
    conditioned_area_sqft: float,
    has_ductwork: bool,
) -> float:
    """
    Estimate the total installed cost of a whole-home ASHP electrification
    using the OLS regression model from the thesis.

    Parameters
    ----------
    capacity_btu : float
        Nameplate heating capacity of the ASHP in BTU/hr.
    hspf : float
        Heating Seasonal Performance Factor (Region IV) of the ASHP.
    conditioned_area_sqft : float
        Conditioned floor area of the building in square feet.
    has_ductwork : bool
        True if the building already has ductwork, False if not.

    Returns
    -------
    float
        Estimated total installed cost in dollars.
    """
    base_appliance_cost = 933.88 + 0.1480 * capacity_btu + 1.1126 * conditioned_area_sqft
    appliance_cost = base_appliance_cost + eff_adjustment(capacity_btu, hspf, conditioned_area_sqft)
    labor_cost = 2286 + 0.1179 * capacity_btu + 0.6556 * conditioned_area_sqft
    ductwork_cost = 0.0 if has_ductwork else 0.883 * conditioned_area_sqft
    scaled_misc_cost = 74.797 + 1.1028 * conditioned_area_sqft
    flat_misc_cost = 180.0
    total_cost = appliance_cost + labor_cost + ductwork_cost + scaled_misc_cost + flat_misc_cost

    return {
        "cost_appliance": appliance_cost,
        "cost_labor": labor_cost,
        "cost_ductwork": ductwork_cost,
        "cost_scaled_misc": scaled_misc_cost,
        "cost_flat_misc": flat_misc_cost,
        "cost_total": total_cost,
    }


# ---------------------------------------------------------------------------
# Cost menu
# ---------------------------------------------------------------------------

def build_cost_menu(
    neep_df: pd.DataFrame,
    conditioned_area_sqft: float,
    has_ductwork: bool,
) -> pd.DataFrame:
    """
    Estimate installed cost components for each eligible heat pump option.

    Parameters
    ----------
    neep_df : pd.DataFrame
        Filtered NEEP dataframe with capacity_btu and ccASHP columns.
    conditioned_area_sqft : float
        Conditioned floor area of the building in square feet.
    has_ductwork : bool
        True if the building already has ductwork.

    Returns
    -------
    pd.DataFrame
        neep_df with added cost component columns.
    """
    costs = neep_df.apply(
        lambda row: pd.Series(estimate_total_cost(
            capacity_btu=row["capacity_btu"],
            hspf=row["HSPF (Region IV)"],
            conditioned_area_sqft=conditioned_area_sqft,
            has_ductwork=has_ductwork,
        )),
        axis=1,
    )
    return pd.concat([neep_df, costs], axis=1)


# ---------------------------------------------------------------------------
# Multi-unit matching
# ---------------------------------------------------------------------------

def find_multi_unit_options(
    neep_df: pd.DataFrame,
    peak_load_btu_hr: float,
    peak_cooling_btu_hr: float,
) -> tuple[pd.DataFrame, int] | tuple[None, None]:
    """
    Try to find heat pump combinations that together meet both peak heating and
    cooling loads, by installing N identical units of the same model.

    For each N in [2, 3, 4]:
      - Each unit must individually meet peak_load_btu_hr / N (heating)
      - Each unit must individually meet peak_cooling_btu_hr / N (cooling)

    Parameters
    ----------
    neep_df : pd.DataFrame
        Full NEEP dataframe with capacity_btu and cooling_capacity_btu columns.
    peak_load_btu_hr : float
        Required peak heating load in BTU/hr.
    peak_cooling_btu_hr : float
        Required peak cooling load in BTU/hr.

    Returns
    -------
    tuple of (filtered DataFrame, number of units) if a match is found,
    or (None, None) if no combination works up to 4 units.
    """
    for n in [2, 3, 4]:
        heating_per_unit = peak_load_btu_hr / n
        cooling_per_unit = peak_cooling_btu_hr / n
        eligible = neep_df[
            (neep_df["capacity_btu"] >= heating_per_unit) &
            (neep_df["cooling_capacity_btu"] >= cooling_per_unit)
        ].copy()
        if not eligible.empty:
            eligible["sufficient_capacity"] = True
            return eligible, n

    return None, None


# ---------------------------------------------------------------------------
# Heat pump metrics
# ---------------------------------------------------------------------------

def populate_heatpump_metrics(
    neep_df: pd.DataFrame,
    neep_idx: int,
    building_id: int,
    peak_load_btu_hr: float,
    outdoor_temp_f_val: float,
    peak_cooling_btu_hr: float,
    cooling_temp_f_val: float,
    cost_total: float,
    consumption_df: pd.DataFrame = None,
) -> dict:
    """
    Populate the performance metrics for the selected heat pump for a building.

    Parameters
    ----------
    neep_df : pd.DataFrame
        Full NEEP dataframe with capacity and COP columns.
    neep_idx : int
        Index of the selected heat pump in neep_df.
    building_id : int
        Building ID.
    peak_load_btu_hr : float
        Peak heating load in BTU/hr.
    outdoor_temp_f_val : float
        Outdoor temperature at peak heating load in Fahrenheit.
    cost_total : float
        Total installed cost in dollars.

    Returns
    -------
    dict
        Dictionary of heat pump metrics for the building.
    """
    BTU_TO_KW = 1 / 3412.142
    DESIGN_TEMP_C = 8.33  # 47°F in °C

    row = neep_df.loc[neep_idx]

    heating_capacity_kw = row["Rated Capacity 47°F"] * BTU_TO_KW
    cooling_capacity_kw = row["Rated Capacity 95°F"] * BTU_TO_KW
    cop_heating = row["COP at Rated Capacity 47°F"]
    cop_cooling = row["COP at Rated Capacity 95°F"]

    # --- Heating capacity loss (kW/°C) ---
    # Three points: 5°F, 17°F, 47°F converted to °C, delta relative to rated at 47°F
    temps_f = np.array([5.0, 17.0, 47.0])
    temps_c = (temps_f - 32) / 1.8
    caps = np.array([
        row["Rated Capacity 5°F"],
        row["Rated Capacity 17°F"],
        row["Rated Capacity 47°F"],
    ]) * BTU_TO_KW
    deltas = caps - heating_capacity_kw  # delta relative to 47°F, zero at 47°F
    delta_temps = temps_c - DESIGN_TEMP_C  # zero at 47°F
    # Fit slope through origin (no intercept)
    heating_capacity_loss = float(np.linalg.lstsq(delta_temps.reshape(-1, 1), deltas, rcond=None)[0][0])

    # --- Cooling capacity loss (kW/°C) ---
    # Three points: 47°F, 82°F, 95°F converted to °C, delta relative to rated at 47°F
    cool_temps_f = np.array([47.0, 82.0, 95.0])
    cool_temps_c = (cool_temps_f - 32) / 1.8
    cool_caps = np.array([
        row["Rated Capacity 47°F"],
        row["Rated Capacity 82°F"],
        row["Rated Capacity 95°F"],
    ]) * BTU_TO_KW
    cool_deltas = cool_caps - heating_capacity_kw  # delta relative to 47°F, zero at 47°F
    cool_delta_temps = cool_temps_c - DESIGN_TEMP_C  # zero at 47°F
    cooling_capacity_loss = float(np.linalg.lstsq(cool_delta_temps.reshape(-1, 1), cool_deltas, rcond=None)[0][0])

    # --- COP heating loss (--/°C) ---
    # Points: 5°F, 17°F, 47°F — delta relative to COP at 47°F
    cop_h_vals = np.array([
        row["COP at Rated Capacity 5°F"],
        row["COP at Rated Capacity 17°F"],
        cop_heating,
    ])
    cop_h_deltas = cop_h_vals - cop_heating
    cop_h_delta_temps = temps_c - DESIGN_TEMP_C
    cop_loss_heating = float(np.linalg.lstsq(cop_h_delta_temps.reshape(-1, 1), cop_h_deltas, rcond=None)[0][0])

    # --- COP cooling loss (--/°C) ---
    # Points: 47°F, 82°F, 95°F — delta relative to COP at 47°F
    cop_c_vals = np.array([
        cop_heating,
        row["COP at Rated Capacity 82°F"],
        row["COP at Rated Capacity 95°F"],
    ])
    cop_c_deltas = cop_c_vals - cop_heating
    cop_c_delta_temps = cool_temps_c - DESIGN_TEMP_C
    cop_loss_cooling = float(np.linalg.lstsq(cop_c_delta_temps.reshape(-1, 1), cop_c_deltas, rcond=None)[0][0])

    # --- Outdoor temperature range ---
    if consumption_df is not None:
        temp_series = consumption_df["out.outdoor_air_drybulb_temp..c"] * 1.8 + 32
        lowest_outdoor_temp_f = round(temp_series.min(), 1)
        highest_outdoor_temp_f = round(temp_series.max(), 1)
    else:
        lowest_outdoor_temp_f = None
        highest_outdoor_temp_f = None

    return {
        "bldg_id": building_id,
        "lowest_outdoor_temp_f": lowest_outdoor_temp_f,
        "highest_outdoor_temp_f": highest_outdoor_temp_f,
        "peak_load_btu_hr_heating": peak_load_btu_hr,
        "peak_load_temp_f_heating": round(outdoor_temp_f_val, 1),
        "peak_load_btu_hr_cooling": peak_cooling_btu_hr,
        "peak_load_temp_f_cooling": round(cooling_temp_f_val, 1),
        "heating_capacity_kw": heating_capacity_kw,
        "cooling_capacity_kw": cooling_capacity_kw,
        "cop_heating": cop_heating,
        "cop_cooling": cop_cooling,
        "design_temperature_c": DESIGN_TEMP_C,
        "heating_capacity_loss": heating_capacity_loss,
        "cooling_capacity_loss": cooling_capacity_loss,
        "cop_loss_heating": cop_loss_heating,
        "cop_loss_cooling": cop_loss_cooling,
        "capital_cost": cost_total,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_heatpump_scatter(
    ax,
    neep_df: pd.DataFrame,
    heating_req: float,
    cooling_req: float,
    title: str,
) -> None:
    """Helper to draw a single heat pump scatter plot on an axis."""
    hspf_vals = neep_df["HSPF (Region IV)"]
    scatter = ax.scatter(
        neep_df["Rated Capacity 47°F"] / 1000,
        neep_df["cost_total"],
        c=hspf_vals,
        cmap="viridis",
        vmin=hspf_vals.min(),
        vmax=hspf_vals.max(),
        alpha=0.8,
        edgecolors="none",
    )
    plt.colorbar(scatter, ax=ax).set_label("HSPF (Region IV)")
    ax.set_title(title)
    ax.set_xlabel("Rated Capacity at 47°F [kBtu/hr]")
    ax.set_ylabel("Total Cost [$]")
    ax.axvline(
        x=heating_req / 1000,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Heating req. ({heating_req / 1000:,.1f} kBtu/hr)",
    )
    ax.axvline(
        x=cooling_req / 1000,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Cooling req. ({cooling_req / 1000:,.1f} kBtu/hr)",
    )
    ax.legend()


def plot_heatpump_options(
    neep_df: pd.DataFrame,
    building_id: int,
    peak_load_btu_hr: float,
    peak_cooling_btu_hr: float,
    number_hp: int,
    output_dir: Path,
) -> None:
    """
    Save two plots for a building:
      1. Single-unit view: full building load requirements vs each unit's capacity.
      2. Per-unit view: load divided by number_hp, showing which units are viable.

    Parameters
    ----------
    neep_df : pd.DataFrame
        Heat pump dataframe with cost_total, capacity_btu, and HSPF columns.
    building_id : int
        Building ID used in the plot title and filename.
    peak_load_btu_hr : float
        Peak heating load in BTU/hr.
    peak_cooling_btu_hr : float
        Peak cooling load in BTU/hr.
    number_hp : int
        Number of units selected for this building.
    output_dir : Path
        Directory where the plot PNGs will be saved.
    """
    # --- Plot 1: single-unit full load ---
    fig, ax = plt.subplots(figsize=(10, 6))
    _make_heatpump_scatter(
        ax, neep_df,
        heating_req=peak_load_btu_hr,
        cooling_req=peak_cooling_btu_hr,
        title=f"Building {building_id} — Single-unit load (1 unit)",
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{building_id}_heatpump_options_single.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: per-unit load (divided by number_hp) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    _make_heatpump_scatter(
        ax, neep_df,
        heating_req=peak_load_btu_hr / number_hp,
        cooling_req=peak_cooling_btu_hr / number_hp,
        title=f"Building {building_id} — Per-unit load ({number_hp} units)",
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{building_id}_heatpump_options_per_unit.png", dpi=150)
    plt.close(fig)
    print(f"Saved plots for building {building_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Define all file and folder paths here ---
    BASE_DIR = Path(__file__).parents[1]
    neep_filepath = BASE_DIR / "in" / "neep_database.csv"
    metadata_filepath = BASE_DIR / "in" / "TX_upgrade0.parquet"
    ercot_map_filepath = BASE_DIR / "out" / "ercot_substation_nrel_map.parquet"
    consumption_dir = BASE_DIR / "out" / "consumption_files"

    # --- Flags ---
    plot_flag = False
    save_parquet = False
    debug = False  # If True, only process the first 10 buildings

    # --- Load shared data once ---
    neep_base = load_neep_heat_pumps(neep_filepath)
    metadata_df = load_metadata(metadata_filepath)

    # Load ERCOT substation map and get valid building IDs
    ercot_map_df = pd.read_parquet(ercot_map_filepath, engine="fastparquet", columns=["bldg_id"])
    ercot_bldg_ids = set(ercot_map_df["bldg_id"].astype(int).tolist())
    print(f"ERCOT map contains {len(ercot_bldg_ids)} buildings to process.")
    n_neep_total = len(neep_base)

    # --- Pre-filter ccASHP units (those with optional low temperature data) ---
    ccASHP_df = neep_base.dropna(subset=[
        "Optional Low Temperature Data Outdoor Dry Bulb (°F)",
        "Optional Low Temperature Data Minimum Capacity X°F",
        "Optional Low Temperature Data Maximum Capacity X°F",
    ]).copy()
    print(f"ccASHP units with optional low temp data: {len(ccASHP_df)}")

    heatpump_dir = BASE_DIR / "out" / "heatpump_files"
    heatpump_dir.mkdir(parents=True, exist_ok=True)

    # Save the filtered NEEP heat pump list with explicit index column
    neep_export = neep_base.copy()
    neep_export.insert(0, "hp_index", neep_export.index)
    neep_export.to_csv(heatpump_dir / "available_heatpumps.csv", index=False)
    print(f"Saved {len(neep_export)} available heat pumps to available_heatpumps.csv")

    # --- Loop over all buildings in consumption_files ---
    consumption_files = sorted(consumption_dir.glob("*-0.parquet"))
    consumption_files = [f for f in consumption_files if int(f.stem.split("-")[0]) in ercot_bldg_ids]
    print(f"Found {len(consumption_files)} buildings to process (filtered to ERCOT map).")

    zero_hp_match = []
    summary_rows = []

    for filepath in (consumption_files[:10] if debug else consumption_files):
        building_id = int(filepath.stem.split("-")[0])

        try:
            building = get_building_metadata(metadata_df, building_id)
            consumption_df = load_building_consumption(consumption_dir, building_id)

            neep_df, peak_load_btu_hr, outdoor_temp_f_val, peak_cooling_btu_hr, cooling_temp_f_val = get_capacity_at_design_temp(neep_base, consumption_df, ccASHP_df=ccASHP_df, conditioned_area_sqft=building["conditioned_area_sqft"], verbose=False)
            n_sufficient = neep_df["sufficient_capacity"].sum()
            print(f"{n_sufficient} out of {n_neep_total} heat pumps match the capacity requirements for building {building_id}")

            if n_sufficient == 0:
                # Try splitting load across 2, 3, or 4 identical units
                neep_df, number_hp = find_multi_unit_options(neep_df, peak_load_btu_hr, peak_cooling_btu_hr)

                if neep_df is None:
                    # No combination works up to 4 units
                    zero_hp_match.append({
                        "bldg_id": building_id,
                        "peak_load_btu_hr": peak_load_btu_hr,
                        "outdoor_temp_f": round(outdoor_temp_f_val, 1),
                    })
                    print(f"Building {building_id}: no matching heat pump even with up to 4 units.")
                else:
                    print(f"Building {building_id}: matched with {number_hp} identical units.")
                    neep_df = build_cost_menu(neep_df, building["conditioned_area_sqft"], building["has_ductwork"])
                    neep_df["number_hp"] = number_hp

                    if save_parquet:
                        output_path = heatpump_dir / f"{building_id}_heatpump_options.parquet"
                        neep_df.to_parquet(output_path, engine="fastparquet", index=False)

                    cheapest = neep_df[neep_df["sufficient_capacity"]].nsmallest(1, "cost_total").iloc[0]
                    metrics = populate_heatpump_metrics(neep_df, cheapest.name, building_id, peak_load_btu_hr, outdoor_temp_f_val, peak_cooling_btu_hr, cooling_temp_f_val, cheapest["cost_total"] * number_hp, consumption_df)
                    metrics["neep_idx"] = cheapest.name
                    metrics["count"] = number_hp
                    summary_rows.append(metrics)

                    if plot_flag:
                        neep_full = build_cost_menu(
                            get_capacity_at_design_temp(neep_base, consumption_df, ccASHP_df=ccASHP_df, conditioned_area_sqft=building["conditioned_area_sqft"], verbose=False)[0],
                            building["conditioned_area_sqft"], building["has_ductwork"]
                        )
                        plot_heatpump_options(neep_full, building_id, peak_load_btu_hr, peak_cooling_btu_hr, number_hp, heatpump_dir)
            else:
                neep_df["number_hp"] = 1
                neep_df = build_cost_menu(neep_df, building["conditioned_area_sqft"], building["has_ductwork"])

                if save_parquet:
                    output_path = heatpump_dir / f"{building_id}_heatpump_options.parquet"
                    neep_df.to_parquet(output_path, engine="fastparquet", index=False)

                cheapest = neep_df[neep_df["sufficient_capacity"]].nsmallest(1, "cost_total").iloc[0]
                metrics = populate_heatpump_metrics(neep_df, cheapest.name, building_id, peak_load_btu_hr, outdoor_temp_f_val, peak_cooling_btu_hr, cooling_temp_f_val, cheapest["cost_total"], consumption_df)
                metrics["neep_idx"] = cheapest.name
                metrics["count"] = 1
                summary_rows.append(metrics)

                if plot_flag:
                    plot_heatpump_options(neep_df, building_id, peak_load_btu_hr, peak_cooling_btu_hr, 1, heatpump_dir)

        except Exception as e:
            print(f"Error processing building {building_id}: {e}")

    if zero_hp_match:
        zero_hp_df = pd.DataFrame(zero_hp_match)
        zero_hp_path = heatpump_dir / "zero_hp_match_buildings.csv"
        zero_hp_df.to_csv(zero_hp_path, index=False)
        print(f"{len(zero_hp_match)} buildings with no matching heat pumps saved to {zero_hp_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = heatpump_dir / "building_hp_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Building HP summary saved to {summary_path}")