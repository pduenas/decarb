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

# Conversion factor
KW_TO_BTU = 3412.142


# ---------------------------------------------------------------------------
# NEEP loading
# ---------------------------------------------------------------------------

NEEP_TEMP_TIERS = ["5°F", "17°F", "47°F", "82°F", "95°F"]


def clean_neep_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each temperature tier, fill missing Rated Capacity and COP at Rated
    Capacity with the arithmetic mean of the corresponding Min and Max columns.

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


def load_neep_heat_pumps(neep_filepath: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Load and filter the NEEP database for eligible Mitsubishi heat pumps, then
    derive the generic linear HP parameters used by the rest of this pipeline.

    Filters:
      - 'Brand Name' must contain 'mitsubishi' (case-insensitive)
      - 'Status' must equal 'Live' (case-insensitive)
      - 'HSPF (Region IV)' must be populated
      - At least one capacity column must be populated (not all empty)
      - Any unit with a rated COP > 6 at any temperature tier is excluded

    Derived columns added to match the generic HP schema:
      ty      : model identifier (Brand Name + model columns joined)
      pHVmx   : rated heating capacity at 47°F in kW
      pACmx   : rated cooling capacity at 47°F in kW  (reference temp, not 95°F)
      pHVeff  : COP at rated capacity at 47°F
      pACeff  : COP at rated capacity at 47°F  (reference temp, not 95°F)
      temp    : reference temperature in °C (47°F = 8.33°C)
      pHVmx_  : heating capacity slope (kW/°C), fitted through 5°F/17°F/47°F
      pACmx_  : cooling capacity loss magnitude (kW/°C), negated so positive,
                fitted through 47°F/82°F/95°F
      pHVeff_ : heating COP slope (per °C), fitted through 5°F/17°F/47°F
      pACeff_ : cooling COP slope (per °C), fitted through 47°F/82°F/95°F
      pHSPF   : HSPF (Region IV) from NEEP

    Parameters
    ----------
    neep_filepath : Path
        Path to the NEEP database file (CSV or Excel).
    verbose : bool
        If True, print loading summary.

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per eligible unit and all generic HP columns
        populated, ready for use with the rest of this pipeline.
    """
    BTU_TO_KW = 1 / KW_TO_BTU
    DESIGN_TEMP_C = (47.0 - 32) / 1.8  # 8.33°C

    suffix = neep_filepath.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(neep_filepath, low_memory=False)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(neep_filepath)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .csv or .xlsx/.xls.")

    missing = [col for col in NEEP_REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in NEEP database: {missing}")

    df = df[df["Brand Name"].str.contains("mitsubishi", case=False, na=False)]
    df = df[df["Status"].str.strip().str.lower() == "live"]
    df = df.dropna(subset=["HSPF (Region IV)"])
    df = df.dropna(subset=NEEP_CAPACITY_COLS, how="all")
    df = df.reset_index(drop=True)
    df = clean_neep_data(df)

    cop_rated_cols = [c for c in NEEP_COP_COLS if "Rated" in c and c in df.columns]
    cop_mask = (df[cop_rated_cols] > 6).any(axis=1)
    n_excluded = cop_mask.sum()
    df = df[~cop_mask].reset_index(drop=True)

    # --- Derive generic HP parameters ---
    # All slopes are fitted relative to 47°F (DESIGN_TEMP_C) as the zero-delta reference.
    h_temps_f = np.array([5.0, 17.0, 47.0])
    h_temps_c = (h_temps_f - 32) / 1.8
    h_delta_c = h_temps_c - DESIGN_TEMP_C

    c_temps_f = np.array([47.0, 82.0, 95.0])
    c_temps_c = (c_temps_f - 32) / 1.8
    c_delta_c = c_temps_c - DESIGN_TEMP_C

    records = []
    for _, row in df.iterrows():
        # Base capacities and COPs at 47°F — consistent reference for all slopes
        pHVmx  = row["Rated Capacity 47°F"] * BTU_TO_KW
        pACmx  = row["Rated Capacity 47°F"] * BTU_TO_KW
        pHVeff = row["COP at Rated Capacity 47°F"]
        pACeff = row["COP at Rated Capacity 47°F"]

        # Heating capacity slope — as fraction of reference capacity per °C
        h_caps_kw = np.array([
            row["Rated Capacity 5°F"],
            row["Rated Capacity 17°F"],
            row["Rated Capacity 47°F"],
        ]) * BTU_TO_KW
        pHVmx_abs = float(np.linalg.lstsq(
            h_delta_c.reshape(-1, 1), h_caps_kw - pHVmx, rcond=None
        )[0][0])
        pHVmx_ = pHVmx_abs / pHVmx

        # Cooling capacity slope — as fraction of reference capacity per °C,
        # negated so the stored value is a positive magnitude
        c_caps_kw = np.array([
            row["Rated Capacity 47°F"],
            row["Rated Capacity 82°F"],
            row["Rated Capacity 95°F"],
        ]) * BTU_TO_KW
        pACmx_abs = float(np.linalg.lstsq(
            c_delta_c.reshape(-1, 1), c_caps_kw - pACmx, rcond=None
        )[0][0])
        pACmx_ = -(pACmx_abs / pACmx)

        # Heating COP slope — as fraction of reference COP per °C
        h_cops = np.array([
            row["COP at Rated Capacity 5°F"],
            row["COP at Rated Capacity 17°F"],
            pHVeff,
        ])
        pHVeff_abs = float(np.linalg.lstsq(
            h_delta_c.reshape(-1, 1), h_cops - pHVeff, rcond=None
        )[0][0])
        pHVeff_ = pHVeff_abs / pHVeff

        # Cooling COP slope — as fraction of reference COP per °C
        c_cops = np.array([
            pACeff,
            row["COP at Rated Capacity 82°F"],
            row["COP at Rated Capacity 95°F"],
        ])
        pACeff_abs = float(np.linalg.lstsq(
            c_delta_c.reshape(-1, 1), c_cops - pACeff, rcond=None
        )[0][0])
        pACeff_ = pACeff_abs / pACeff

        # Human-readable type name
        name_parts = []
        for col in ["Brand Name", "Model Number", "outdoor_model_number"]:
            if col in row.index and pd.notna(row[col]):
                name_parts.append(str(row[col]).strip())
        ty = " | ".join(name_parts) if name_parts else f"unit_{row.name}"

        records.append({
            "ty":      ty,
            "pHVmx":   pHVmx,
            "pACmx":   pACmx,
            "pHVeff":  pHVeff,
            "pACeff":  pACeff,
            "temp":    DESIGN_TEMP_C,
            "pHVmx_":  pHVmx_,
            "pACmx_":  pACmx_,
            "pHVeff_": pHVeff_,
            "pACeff_": pACeff_,
            "pHSPF":   row["HSPF (Region IV)"],
        })

    result = pd.DataFrame(records).reset_index(drop=True)

    if verbose:
        print(f"NEEP: {len(result)} eligible Mitsubishi Live units loaded "
              f"({n_excluded} excluded for COP > 6).")

    return result


# ---------------------------------------------------------------------------
# Capacity at outdoor temperature
# ---------------------------------------------------------------------------

def capacity_at_temp(row: pd.Series, outdoor_temp_c: float, mode: str) -> float:
    """
    Estimate heating or cooling capacity (kW) at a given outdoor temperature
    using fractional slope (fraction of reference capacity per °C).

    Parameters
    ----------
    row : pd.Series
        A row from the HP dataframe (must have pHVmx/pACmx, pHVmx_/pACmx_, temp).
    outdoor_temp_c : float
        Outdoor temperature in °C.
    mode : str
        'heating' or 'cooling'.

    Returns
    -------
    float
        Estimated capacity in kW (clamped to 0).
    """
    delta = outdoor_temp_c - row["temp"]
    if mode == "heating":
        cap = row["pHVmx"] * (1 + row["pHVmx_"] * delta)
    else:
        # pACmx_ is stored as a positive magnitude fraction; cooling capacity
        # decreases as temperature rises above the reference
        cap = row["pACmx"] * (1 - row["pACmx_"] * delta)
    return max(cap, 0.0)


def cop_at_temp(row: pd.Series, outdoor_temp_c: float, mode: str) -> float:
    """
    Estimate heating or cooling COP at a given outdoor temperature
    using fractional slope (fraction of reference COP per °C).

    Parameters
    ----------
    row : pd.Series
        A row from the HP dataframe.
    outdoor_temp_c : float
        Outdoor temperature in °C.
    mode : str
        'heating' or 'cooling'.

    Returns
    -------
    float
        Estimated COP (clamped to 1.0 minimum).
    """
    delta = outdoor_temp_c - row["temp"]
    if mode == "heating":
        cop = row["pHVeff"] * (1 + row["pHVeff_"] * delta)
    else:
        cop = row["pACeff"] * (1 + row["pACeff_"] * delta)
    return max(cop, 1.0)


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
# Sizing load helpers
# ---------------------------------------------------------------------------

def _absolute_peak_load_and_temp(
    load_series: pd.Series,
    temp_series: pd.Series,
) -> tuple[float, float]:
    """
    Return the absolute peak load (BTU/hr) from nonzero values and its
    directly paired outdoor temperature (°F).

    Parameters
    ----------
    load_series : pd.Series
        Load time series in kBTU per 15-min interval.
    temp_series : pd.Series
        Outdoor air temperature in °C, index-aligned with load_series.

    Returns
    -------
    tuple of (peak_load_btu_hr, outdoor_temp_f)
    """
    nonzero = load_series[load_series > 0]
    idx = nonzero.idxmax()
    return load_series.loc[idx] * 4000, temp_series.loc[idx] * 1.8 + 32


def _sizing_load_and_temp(
    load_series: pd.Series,
    temp_series: pd.Series,
    percentile: float,
) -> tuple[float, float]:
    """
    Derive a sizing load (BTU/hr) and the paired outdoor temperature (°F)
    from a load time series.

    Steps:
      1. Filter to nonzero timesteps.
      2. Compute the given percentile of those nonzero loads.
      3. If that threshold is zero (degenerate), fall back to
         percentile/100 * peak load over the full series.
      4. Find the timestep whose load is closest to the threshold.
      5. Return that load converted to BTU/hr and its paired outdoor temp in °F.

    Parameters
    ----------
    load_series : pd.Series
        Load time series in kBTU per 15-min interval.
    temp_series : pd.Series
        Outdoor air temperature in °C, index-aligned with load_series.
    percentile : float
        Percentile to apply to nonzero values (e.g. 95.0).

    Returns
    -------
    tuple of (sizing_load_btu_hr, outdoor_temp_f)
    """
    nonzero = load_series[load_series > 0]
    threshold = nonzero.quantile(percentile / 100.0)

    if threshold == 0:
        threshold = load_series.max() * (percentile / 100.0)

    idx = (load_series - threshold).abs().idxmin()
    return load_series.loc[idx] * 4000, temp_series.loc[idx] * 1.8 + 32


# ---------------------------------------------------------------------------
# Peak load determination
# ---------------------------------------------------------------------------

def get_peak_loads(
    consumption_df: pd.DataFrame,
    conditioned_area_sqft: float,
    percentile: float = 95.0,
    verbose: bool = True,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Determine sizing and absolute peak loads for heating and cooling, along
    with their paired outdoor temperatures.

    Sizing loads use the given percentile of nonzero values (used for HP
    selection). Absolute peaks are the true maximum nonzero values (stored
    in the summary for reference).

    Parameters
    ----------
    consumption_df : pd.DataFrame
        Building consumption dataframe.
    conditioned_area_sqft : float
        Conditioned floor area in sqft (used for fallback rule-of-thumb).
    percentile : float
        Percentile to use for sizing (default 95.0).

    Returns
    -------
    tuple of (
        sizing_load_btu_hr_heating, sizing_temp_f_heating,
        sizing_load_btu_hr_cooling, sizing_temp_f_cooling,
        abs_peak_btu_hr_heating, abs_peak_temp_f_heating,
        abs_peak_btu_hr_cooling, abs_peak_temp_f_cooling,
    )
    """
    temp_series = consumption_df["out.outdoor_air_drybulb_temp..c"]

    # --- Heating ---
    heating_series = consumption_df["out.load.heating.energy_delivered..kbtu"]
    if heating_series.isna().all() or heating_series.max() == 0:
        fallback = conditioned_area_sqft * 30
        fallback_temp = temp_series.min() * 1.8 + 32
        sizing_load_heating, sizing_temp_heating = fallback, fallback_temp
        abs_peak_heating, abs_peak_temp_heating = fallback, fallback_temp
        if verbose:
            print("No heating load found; using rule-of-thumb (30 BTU/hr per sqft).")
    else:
        sizing_load_heating, sizing_temp_heating = _sizing_load_and_temp(
            heating_series, temp_series, percentile,
        )
        abs_peak_heating, abs_peak_temp_heating = _absolute_peak_load_and_temp(
            heating_series, temp_series,
        )

    # --- Cooling ---
    cooling_series = consumption_df["out.load.cooling.energy_delivered..kbtu"]
    if cooling_series.isna().all() or cooling_series.max() == 0:
        fallback = conditioned_area_sqft * 30
        fallback_temp = temp_series.max() * 1.8 + 32
        sizing_load_cooling, sizing_temp_cooling = fallback, fallback_temp
        abs_peak_cooling, abs_peak_temp_cooling = fallback, fallback_temp
        if verbose:
            print("No cooling load found; using rule-of-thumb (30 BTU/hr per sqft).")
    else:
        sizing_load_cooling, sizing_temp_cooling = _sizing_load_and_temp(
            cooling_series, temp_series, percentile,
        )
        abs_peak_cooling, abs_peak_temp_cooling = _absolute_peak_load_and_temp(
            cooling_series, temp_series,
        )

    if verbose:
        print(f"{percentile}th-pct heating load: {sizing_load_heating:,.0f} BTU/hr at {sizing_temp_heating:.1f}°F")
        print(f"{percentile}th-pct cooling load: {sizing_load_cooling:,.0f} BTU/hr at {sizing_temp_cooling:.1f}°F")
        print(f"Absolute peak heating:  {abs_peak_heating:,.0f} BTU/hr at {abs_peak_temp_heating:.1f}°F")
        print(f"Absolute peak cooling:  {abs_peak_cooling:,.0f} BTU/hr at {abs_peak_temp_cooling:.1f}°F")

    return (
        sizing_load_heating, sizing_temp_heating,
        sizing_load_cooling, sizing_temp_cooling,
        abs_peak_heating, abs_peak_temp_heating,
        abs_peak_cooling, abs_peak_temp_cooling,
    )


# ---------------------------------------------------------------------------
# Capacity filtering
# ---------------------------------------------------------------------------

def get_capacity_at_design_temp(
    hp_df: pd.DataFrame,
    consumption_df: pd.DataFrame,
    conditioned_area_sqft: float = 0.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, float, float, float, float, float, float, float, float]:
    """
    For each HP unit, compute capacity at the sizing load outdoor temperatures
    and flag those that meet both heating and cooling requirements.

    Parameters
    ----------
    hp_df : pd.DataFrame
        HP characteristics dataframe (generic schema with pHVmx, pACmx, etc.).
    consumption_df : pd.DataFrame
        Building consumption dataframe.
    conditioned_area_sqft : float
        Conditioned floor area in sqft (used for fallback).

    Returns
    -------
    tuple of (
        result_df,
        sizing_load_btu_hr_heating, sizing_temp_f_heating,
        sizing_load_btu_hr_cooling, sizing_temp_f_cooling,
        abs_peak_btu_hr_heating, abs_peak_temp_f_heating,
        abs_peak_btu_hr_cooling, abs_peak_temp_f_cooling,
    )
    """
    (
        sizing_load_heating, sizing_temp_heating,
        sizing_load_cooling, sizing_temp_cooling,
        abs_peak_heating, abs_peak_temp_heating,
        abs_peak_cooling, abs_peak_temp_cooling,
    ) = get_peak_loads(consumption_df, conditioned_area_sqft, verbose=verbose)

    outdoor_temp_c_heating = (sizing_temp_heating - 32) / 1.8
    outdoor_temp_c_cooling = (sizing_temp_cooling - 32) / 1.8

    result = hp_df.copy()

    result["capacity_kw_heating"] = result.apply(
        lambda row: capacity_at_temp(row, outdoor_temp_c_heating, "heating"), axis=1
    )
    result["capacity_kw_cooling"] = result.apply(
        lambda row: capacity_at_temp(row, outdoor_temp_c_cooling, "cooling"), axis=1
    )

    result["capacity_btu_heating"] = result["capacity_kw_heating"] * KW_TO_BTU
    result["capacity_btu_cooling"] = result["capacity_kw_cooling"] * KW_TO_BTU

    result["sufficient_capacity"] = (
        (result["capacity_btu_heating"] >= sizing_load_heating) &
        (result["capacity_btu_cooling"] >= sizing_load_cooling)
    )

    n_sufficient = result["sufficient_capacity"].sum()
    if verbose:
        print(f"{n_sufficient} of {len(result)} HP unit(s) have sufficient heating and cooling capacity.")

    return (
        result,
        sizing_load_heating, sizing_temp_heating,
        sizing_load_cooling, sizing_temp_cooling,
        abs_peak_heating, abs_peak_temp_heating,
        abs_peak_cooling, abs_peak_temp_cooling,
    )


# ---------------------------------------------------------------------------
# Multi-unit matching
# ---------------------------------------------------------------------------

def find_multi_unit_options(
    hp_df: pd.DataFrame,
    peak_load_btu_hr: float,
    peak_cooling_btu_hr: float,
    max_units: int = 10,
) -> tuple[pd.DataFrame, int] | tuple[None, None]:
    """
    Try to find HP combinations that together meet both sizing loads by installing
    N identical units of the same type, incrementing N until a match is found.

    Parameters
    ----------
    hp_df : pd.DataFrame
        HP dataframe with capacity_btu_heating and capacity_btu_cooling columns.
    peak_load_btu_hr : float
        Required sizing heating load in BTU/hr.
    peak_cooling_btu_hr : float
        Required sizing cooling load in BTU/hr.
    max_units : int
        Maximum number of identical units to try (default 10).

    Returns
    -------
    tuple of (filtered DataFrame, number of units) or (None, None).
    """
    for n in range(2, max_units + 1):
        heating_per_unit = peak_load_btu_hr / n
        cooling_per_unit = peak_cooling_btu_hr / n
        eligible = hp_df[
            (hp_df["capacity_btu_heating"] >= heating_per_unit) &
            (hp_df["capacity_btu_cooling"] >= cooling_per_unit)
        ].copy()
        if not eligible.empty:
            eligible["sufficient_capacity"] = True
            return eligible, n

    return None, None


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def _extract_hspf(row: pd.Series) -> float:
    """Read the HSPF value from the pHSPF column. Falls back to 10.0 if missing."""
    val = row.get("pHSPF", None)
    if pd.isna(val):
        return 10.0
    return float(val)


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
) -> dict:
    """
    Estimate the total installed cost of a whole-home ASHP electrification
    using the OLS regression model.

    Parameters
    ----------
    capacity_btu : float
        Nameplate heating capacity of the ASHP in BTU/hr.
    hspf : float
        Heating Seasonal Performance Factor (Region IV) of the ASHP.
    conditioned_area_sqft : float
        Conditioned floor area of the building in square feet.
    has_ductwork : bool
        True if the building already has ductwork.

    Returns
    -------
    dict
        Cost component breakdown including cost_total.
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


def build_cost_menu(
    hp_df: pd.DataFrame,
    conditioned_area_sqft: float,
    has_ductwork: bool,
) -> pd.DataFrame:
    """
    Estimate installed cost components for each eligible HP unit.

    Uses pHVmx (heating capacity at reference temp) converted to BTU/hr as the
    nameplate capacity, and pHSPF as the HSPF rating.

    Parameters
    ----------
    hp_df : pd.DataFrame
        HP dataframe with pHVmx and pHSPF columns.
    conditioned_area_sqft : float
        Conditioned floor area in sqft.
    has_ductwork : bool
        True if the building already has ductwork.

    Returns
    -------
    pd.DataFrame
        hp_df with added cost component columns.
    """
    def _cost_row(row):
        capacity_btu = row["pHVmx"] * KW_TO_BTU
        hspf = _extract_hspf(row)
        return pd.Series(estimate_total_cost(capacity_btu, hspf, conditioned_area_sqft, has_ductwork))

    costs = hp_df.apply(_cost_row, axis=1)
    return pd.concat([hp_df, costs], axis=1)


# ---------------------------------------------------------------------------
# Heat pump metrics
# ---------------------------------------------------------------------------

def populate_heatpump_metrics(
    hp_df: pd.DataFrame,
    hp_idx: int,
    building_id: int,
    abs_peak_btu_hr_heating: float,
    abs_peak_temp_f_heating: float,
    abs_peak_btu_hr_cooling: float,
    abs_peak_temp_f_cooling: float,
    sizing_capacity_kw_heating: float,
    sizing_capacity_kw_cooling: float,
    cost_total: float,
    consumption_df: pd.DataFrame = None,
) -> dict:
    """
    Populate performance metrics for the selected HP unit for a building.

    Parameters
    ----------
    hp_df : pd.DataFrame
        HP characteristics dataframe.
    hp_idx : int
        Index of the selected HP unit in hp_df.
    building_id : int
        Building ID.
    abs_peak_btu_hr_heating : float
        Absolute peak heating load (max nonzero) in BTU/hr.
    abs_peak_temp_f_heating : float
        Outdoor temperature paired with absolute peak heating load (°F).
    abs_peak_btu_hr_cooling : float
        Absolute peak cooling load (max nonzero) in BTU/hr.
    abs_peak_temp_f_cooling : float
        Outdoor temperature paired with absolute peak cooling load (°F).
    sizing_capacity_kw_heating : float
        HP heating capacity (kW) evaluated at the 95th-pct sizing temperature.
    sizing_capacity_kw_cooling : float
        HP cooling capacity (kW) evaluated at the 95th-pct sizing temperature.
    cost_total : float
        Total installed cost in dollars.
    consumption_df : pd.DataFrame, optional
        Used to derive observed outdoor temperature range.

    Returns
    -------
    dict
        Dictionary of heat pump metrics for the building.
    """
    row = hp_df.loc[hp_idx]

    cop_heating = row["pHVeff"]
    cop_cooling = row["pACeff"]
    design_temp_c = float(row["temp"])

    heating_capacity_loss = row["pHVmx_"]
    cooling_capacity_loss = row["pACmx_"]
    cop_loss_heating = row["pHVeff_"]
    cop_loss_cooling = row["pACeff_"]

    if consumption_df is not None:
        temp_series = consumption_df["out.outdoor_air_drybulb_temp..c"] * 1.8 + 32
        lowest_outdoor_temp_f = round(temp_series.min(), 1)
        highest_outdoor_temp_f = round(temp_series.max(), 1)
    else:
        lowest_outdoor_temp_f = None
        highest_outdoor_temp_f = None

    return {
        "bldg_id": building_id,
        "hp_index": hp_idx,
        "pHSPF": row["pHSPF"],
        "lowest_outdoor_temp_f": lowest_outdoor_temp_f,
        "highest_outdoor_temp_f": highest_outdoor_temp_f,
        "peak_load_btu_hr_heating": abs_peak_btu_hr_heating,
        "peak_load_temp_f_heating": round(abs_peak_temp_f_heating, 1),
        "peak_load_btu_hr_cooling": abs_peak_btu_hr_cooling,
        "peak_load_temp_f_cooling": round(abs_peak_temp_f_cooling, 1),
        "heating_capacity_kw": sizing_capacity_kw_heating,
        "cooling_capacity_kw": sizing_capacity_kw_cooling,
        "cop_heating": cop_heating,
        "cop_cooling": cop_cooling,
        "design_temperature_c": design_temp_c,
        "heating_capacity_loss": heating_capacity_loss,
        "cooling_capacity_loss": cooling_capacity_loss,
        "cop_loss_heating": cop_loss_heating,
        "cop_loss_cooling": cop_loss_cooling,
        "capital_cost": cost_total,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_hp_heating_subplot(
    ax,
    hp_df: pd.DataFrame,
    heating_req: float,
    outdoor_temp_f_heating: float,
    hspf_min: float,
    hspf_max: float,
) -> None:
    """
    Draw the heating subplot: x-axis is each HP's capacity at the building's
    heating sizing temperature, y-axis is total installed cost, colored by HSPF.
    """
    hspf_vals = hp_df["pHSPF"]
    scatter = ax.scatter(
        hp_df["capacity_kw_heating"],
        hp_df["cost_total"],
        c=hspf_vals,
        cmap="viridis",
        vmin=hspf_min,
        vmax=hspf_max,
        alpha=0.8,
        edgecolors="none",
        s=100,
    )
    plt.colorbar(scatter, ax=ax).set_label("HSPF (Region IV)")
    ax.set_title(f"Heating  (sized at {outdoor_temp_f_heating:.1f}°F)")
    ax.set_xlabel(f"Heating Capacity at {outdoor_temp_f_heating:.1f}°F [kW]")
    ax.set_ylabel("Total Cost [$]")
    ax.axvline(
        x=heating_req / KW_TO_BTU,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Heating req. ({heating_req / KW_TO_BTU:,.1f} kW)",
    )
    ax.legend()


def _make_hp_cooling_subplot(
    ax,
    hp_df: pd.DataFrame,
    cooling_req: float,
    outdoor_temp_f_cooling: float,
    hspf_min: float,
    hspf_max: float,
) -> None:
    """
    Draw the cooling subplot: x-axis is each HP's capacity at the building's
    cooling sizing temperature, y-axis is total installed cost, colored by HSPF.
    """
    hspf_vals = hp_df["pHSPF"]
    scatter = ax.scatter(
        hp_df["capacity_kw_cooling"],
        hp_df["cost_total"],
        c=hspf_vals,
        cmap="viridis",
        vmin=hspf_min,
        vmax=hspf_max,
        alpha=0.8,
        edgecolors="none",
        s=100,
    )
    plt.colorbar(scatter, ax=ax).set_label("HSPF (Region IV)")
    ax.set_title(f"Cooling  (sized at {outdoor_temp_f_cooling:.1f}°F)")
    ax.set_xlabel(f"Cooling Capacity at {outdoor_temp_f_cooling:.1f}°F [kW]")
    ax.set_ylabel("Total Cost [$]")
    ax.axvline(
        x=cooling_req / KW_TO_BTU,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Cooling req. ({cooling_req / KW_TO_BTU:,.1f} kW)",
    )
    ax.legend()


def _save_hp_figure(
    hp_df: pd.DataFrame,
    heating_req: float,
    cooling_req: float,
    outdoor_temp_f_heating: float,
    outdoor_temp_f_cooling: float,
    suptitle: str,
    filepath: Path,
) -> None:
    """Create a figure with two side-by-side subplots (heating + cooling) and save it."""
    hspf_min = hp_df["pHSPF"].min()
    hspf_max = hp_df["pHSPF"].max()

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(18, 6))
    _make_hp_heating_subplot(ax_h, hp_df, heating_req, outdoor_temp_f_heating, hspf_min, hspf_max)
    _make_hp_cooling_subplot(ax_c, hp_df, cooling_req, outdoor_temp_f_cooling, hspf_min, hspf_max)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_heatpump_options(
    hp_df: pd.DataFrame,
    building_id: int,
    peak_load_btu_hr: float,
    peak_cooling_btu_hr: float,
    outdoor_temp_f_heating: float,
    outdoor_temp_f_cooling: float,
    number_hp: int,
    output_dir: Path,
) -> None:
    """
    Save two plots for a building, each with heating and cooling subplots:
      1. Single-unit view: full sizing load requirements vs each HP unit's
         capacity at the building's design temperatures.
      2. Per-unit view: load divided by number_hp, same x-axes.

    Parameters
    ----------
    hp_df : pd.DataFrame
        HP dataframe with cost_total, capacity_kw_heating, capacity_kw_cooling.
    building_id : int
        Building ID used in the plot title and filename.
    peak_load_btu_hr : float
        Sizing heating load in BTU/hr.
    peak_cooling_btu_hr : float
        Sizing cooling load in BTU/hr.
    outdoor_temp_f_heating : float
        Outdoor temperature used for heating sizing (°F).
    outdoor_temp_f_cooling : float
        Outdoor temperature used for cooling sizing (°F).
    number_hp : int
        Number of units selected for this building.
    output_dir : Path
        Directory where the plot PNGs will be saved.
    """
    _save_hp_figure(
        hp_df,
        heating_req=peak_load_btu_hr,
        cooling_req=peak_cooling_btu_hr,
        outdoor_temp_f_heating=outdoor_temp_f_heating,
        outdoor_temp_f_cooling=outdoor_temp_f_cooling,
        suptitle=f"Building {building_id} — Single-unit load (1 unit)",
        filepath=output_dir / f"{building_id}_hp_options_single.png",
    )
    _save_hp_figure(
        hp_df,
        heating_req=peak_load_btu_hr / number_hp,
        cooling_req=peak_cooling_btu_hr / number_hp,
        outdoor_temp_f_heating=outdoor_temp_f_heating,
        outdoor_temp_f_cooling=outdoor_temp_f_cooling,
        suptitle=f"Building {building_id} — Per-unit load ({number_hp} units)",
        filepath=output_dir / f"{building_id}_hp_options_per_unit.png",
    )
    print(f"Saved plots for building {building_id}")


# ---------------------------------------------------------------------------
# Summary histogram
# ---------------------------------------------------------------------------

def plot_hspf_allocation_histogram(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot and save a histogram of buildings allocated to each HSPF heat pump type.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary dataframe containing a 'pHSPF' column with numeric HSPF values.
    output_dir : Path
        Directory where the histogram PNG will be saved.
    """
    if "pHSPF" not in summary_df.columns:
        print("Histogram skipped: 'pHSPF' column not found in summary.")
        return

    hspf_series = summary_df["pHSPF"].dropna()
    counts = hspf_series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="white", width=0.6)
    ax.set_xlabel("HSPF (Region IV)")
    ax.set_ylabel("Number of Buildings")
    ax.set_title("Buildings Allocated by Heat Pump HSPF Rating")
    for x, y in zip(counts.index.astype(str), counts.values):
        ax.text(x, y + 0.05, str(int(y)), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path = output_dir / "hspf_allocation_histogram.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved HSPF allocation histogram to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import concurrent.futures


def process_building(
    filepath: Path,
    hp_base: pd.DataFrame,
    metadata_df: pd.DataFrame,
    consumption_dir: Path,
    heatpump_dir: Path,
    save_parquet: bool,
    plot_flag: bool,
) -> tuple[dict | None, dict | None]:
    """
    Process a single building and return (metrics_dict, zero_match_dict).
    One of the two will always be None.
    """
    building_id = int(filepath.stem.split("-")[0])
    try:
        building = get_building_metadata(metadata_df, building_id)
        consumption_df = load_building_consumption(consumption_dir, building_id)

        (
            hp_df,
            sizing_load_heating, sizing_temp_heating,
            sizing_load_cooling, sizing_temp_cooling,
            abs_peak_heating, abs_peak_temp_heating,
            abs_peak_cooling, abs_peak_temp_cooling,
        ) = get_capacity_at_design_temp(
            hp_base, consumption_df,
            conditioned_area_sqft=building["conditioned_area_sqft"],
            verbose=False,
        )

        n_sufficient = hp_df["sufficient_capacity"].sum()
        print(f"{n_sufficient} of {len(hp_base)} HP unit(s) meet capacity requirements for building {building_id}")

        if n_sufficient == 0:
            hp_df_multi, number_hp = find_multi_unit_options(hp_df, sizing_load_heating, sizing_load_cooling)

            if hp_df_multi is None:
                print(f"Building {building_id}: no matching HP unit even with up to 10 units.")
                return None, {
                    "bldg_id": building_id,
                    "peak_load_btu_hr": abs_peak_heating,
                    "outdoor_temp_f": round(abs_peak_temp_heating, 1),
                }
            else:
                print(f"Building {building_id}: matched with {number_hp} identical units.")
                hp_df_multi = build_cost_menu(hp_df_multi, building["conditioned_area_sqft"], building["has_ductwork"])
                hp_df_multi["number_hp"] = number_hp

                if save_parquet:
                    hp_df_multi.to_parquet(
                        heatpump_dir / f"{building_id}_hp_options.parquet",
                        engine="fastparquet", index=False,
                    )

                cheapest = hp_df_multi[hp_df_multi["sufficient_capacity"]].nsmallest(1, "cost_total").iloc[0]
                metrics = populate_heatpump_metrics(
                    hp_df_multi, cheapest.name, building_id,
                    abs_peak_heating, abs_peak_temp_heating,
                    abs_peak_cooling, abs_peak_temp_cooling,
                    cheapest["capacity_kw_heating"], cheapest["capacity_kw_cooling"],
                    cheapest["cost_total"] * number_hp, consumption_df,
                )
                metrics["hp_idx"] = cheapest.name
                metrics["count"] = number_hp

                if plot_flag:
                    hp_full = build_cost_menu(hp_df, building["conditioned_area_sqft"], building["has_ductwork"])
                    plot_heatpump_options(
                        hp_full, building_id,
                        sizing_load_heating, sizing_load_cooling,
                        sizing_temp_heating, sizing_temp_cooling,
                        number_hp, heatpump_dir,
                    )

                return metrics, None

        else:
            hp_df["number_hp"] = 1
            hp_df = build_cost_menu(hp_df, building["conditioned_area_sqft"], building["has_ductwork"])

            if save_parquet:
                hp_df.to_parquet(
                    heatpump_dir / f"{building_id}_hp_options.parquet",
                    engine="fastparquet", index=False,
                )

            cheapest = hp_df[hp_df["sufficient_capacity"]].nsmallest(1, "cost_total").iloc[0]
            metrics = populate_heatpump_metrics(
                hp_df, cheapest.name, building_id,
                abs_peak_heating, abs_peak_temp_heating,
                abs_peak_cooling, abs_peak_temp_cooling,
                cheapest["capacity_kw_heating"], cheapest["capacity_kw_cooling"],
                cheapest["cost_total"], consumption_df,
            )
            metrics["hp_idx"] = cheapest.name
            metrics["count"] = 1

            if plot_flag:
                plot_heatpump_options(
                    hp_df, building_id,
                    sizing_load_heating, sizing_load_cooling,
                    sizing_temp_heating, sizing_temp_cooling,
                    1, heatpump_dir,
                )

            return metrics, None

    except Exception as e:
        print(f"Error processing building {building_id}: {e}")
        return None, None


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parents[1]
    neep_filepath = BASE_DIR / "in" / "neep_database.csv"
    metadata_filepath = BASE_DIR / "in" / "TX_upgrade0.parquet"
    ercot_map_filepath = BASE_DIR / "out" / "ercot_substation_nrel_map.parquet"
    consumption_dir = BASE_DIR / "out" / "consumption_files"

    plot_flag = False
    save_parquet = False
    debug = False   # If True, only process the first 10 buildings
    overwrite_flag = True  # If False, skip buildings already in the summary file
    n_workers = 20

    hp_base = load_neep_heat_pumps(neep_filepath)
    metadata_df = load_metadata(metadata_filepath)

    ercot_map_df = pd.read_parquet(ercot_map_filepath, engine="fastparquet", columns=["bldg_id"])
    ercot_bldg_ids = set(ercot_map_df["bldg_id"].astype(int).tolist())
    print(f"ERCOT map contains {len(ercot_bldg_ids)} buildings to process.")

    heatpump_dir = BASE_DIR / "out" / "heatpump_files"
    heatpump_dir.mkdir(parents=True, exist_ok=True)

    hp_export = hp_base.copy()
    hp_export.insert(hp_export.columns.get_loc("ty") + 1, "hp_index", hp_export.index)
    hp_export.to_csv(heatpump_dir / "available_heatpumps.csv", index=False)
    print(f"Saved {len(hp_base)} available HP unit(s) to available_heatpumps.csv")

    consumption_files = sorted(consumption_dir.glob("*-0.parquet"))
    consumption_files = [f for f in consumption_files if int(f.stem.split("-")[0]) in ercot_bldg_ids]
    print(f"Found {len(consumption_files)} buildings to process (filtered to ERCOT map).")

    # --- Determine which buildings to process ---
    suffix = "_debug" if debug else ""
    summary_path = heatpump_dir / f"building_hp_summary{suffix}.csv"
    zero_hp_path = heatpump_dir / f"zero_hp_match_buildings{suffix}.csv"

    if not overwrite_flag and summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        already_processed = set(existing_summary["bldg_id"].astype(int).tolist())
        summary_rows = existing_summary.to_dict("records")
        print(f"Resuming: {len(already_processed)} buildings already in summary, skipping them.")
    else:
        already_processed = set()
        summary_rows = []

    files_to_process = [
        f for f in (consumption_files[:10] if debug else consumption_files)
        if int(f.stem.split("-")[0]) not in already_processed
    ]
    print(f"Processing {len(files_to_process)} buildings with {n_workers} workers.")

    zero_hp_match = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_building,
                fp, hp_base, metadata_df, consumption_dir,
                heatpump_dir, save_parquet, plot_flag,
            ): fp
            for fp in files_to_process
        }
        for future in concurrent.futures.as_completed(futures):
            metrics, zero_match = future.result()
            if metrics is not None:
                summary_rows.append(metrics)
            if zero_match is not None:
                zero_hp_match.append(zero_match)

    if zero_hp_match:
        zero_hp_df = pd.DataFrame(zero_hp_match).sort_values("bldg_id").reset_index(drop=True)
        zero_hp_df.to_csv(zero_hp_path, index=False)
        print(f"{len(zero_hp_match)} buildings with no matching HP unit saved to {zero_hp_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("bldg_id").reset_index(drop=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"Building HP summary saved to {summary_path}")
        plot_hspf_allocation_histogram(summary_df, heatpump_dir)