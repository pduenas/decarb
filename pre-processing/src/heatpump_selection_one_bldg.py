from pathlib import Path
import pandas as pd
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
]

NEEP_REQUIRED_COLS = [
    "Brand Name",
    "Status",
    "HSPF (Region IV)",
] + NEEP_CAPACITY_COLS

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

def load_neep_heat_pumps(neep_filepath: Path) -> pd.DataFrame:
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

    print(f"NEEP filter: {len(df)} eligible Mitsubishi Live units found.")

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

def select_capacity(row: pd.Series, outdoor_temp_f: float) -> pd.Series:
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
        rated = row["Rated Capacity 47°F"]
        capacity = rated if not pd.isna(rated) else (row["Minimum Capacity 47°F"] + row["Maximum Capacity 47°F"]) / 2

    elif outdoor_temp_f > 17:
        rated = row["Rated Capacity 17°F"]
        capacity = rated if not pd.isna(rated) else (row["Minimum Capacity 17°F"] + row["Maximum Capacity 17°F"]) / 2

    elif outdoor_temp_f >= 5:
        rated = row["Rated Capacity 5°F"]
        capacity = rated if not pd.isna(rated) else (row["Minimum Capacity 5°F"] + row["Maximum Capacity 5°F"]) / 2

    else:
        capacity = row["Minimum Capacity 5°F"]
        ccASHP = True

    return pd.Series({"capacity_btu": capacity, "ccASHP": ccASHP})


# ---------------------------------------------------------------------------
# Capacity filtering
# ---------------------------------------------------------------------------

def get_capacity_at_design_temp(
    neep_df: pd.DataFrame,
    consumption_df: pd.DataFrame,
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
    peak_idx = consumption_df["out.load.heating.energy_delivered..kbtu"].idxmax()
    peak_load_kbtu = consumption_df.loc[peak_idx, "out.load.heating.energy_delivered..kbtu"]
    peak_load_btu_hr = peak_load_kbtu * 4000
    outdoor_temp_c = consumption_df.loc[peak_idx, "out.outdoor_air_drybulb_temp..c"]
    outdoor_temp_f = outdoor_temp_c * 1.8 + 32

    print(f"Peak heating load: {peak_load_btu_hr:,.0f} BTU/hr at outdoor temp: {outdoor_temp_f:.1f}°F")

    result = neep_df.copy()
    result[["capacity_btu", "ccASHP"]] = result.apply(
        lambda row: select_capacity(row, outdoor_temp_f), axis=1
    )

    result["sufficient_capacity"] = result["capacity_btu"] > peak_load_btu_hr

    n_sufficient = result["sufficient_capacity"].sum()
    print(f"{n_sufficient} of {len(result)} heat pump(s) have sufficient capacity for peak load of {peak_load_btu_hr:,.0f} BTU/hr.")

    return result, peak_load_btu_hr


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
# Plotting
# ---------------------------------------------------------------------------

def plot_heatpump_options(
    neep_df: pd.DataFrame,
    building_id: int,
    peak_load_btu_hr: float,
    output_dir: Path,
) -> None:
    """
    Plot total cost vs capacity for all eligible heat pumps, color-coded by HSPF,
    with a red dashed line at the building peak heating load.

    Parameters
    ----------
    neep_df : pd.DataFrame
        Heat pump dataframe with cost_total, capacity_btu, and HSPF columns.
    building_id : int
        Building ID used in the plot title and filename.
    peak_load_btu_hr : float
        Peak heating load in BTU/hr, shown as a vertical reference line.
    output_dir : Path
        Directory where the plot PNG will be saved.
    """
    hspf_vals = neep_df["HSPF (Region IV)"]
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        neep_df["capacity_btu"] / 1000,
        neep_df["cost_total"],
        c=hspf_vals,
        cmap="viridis",
        vmin=hspf_vals.min(),
        vmax=hspf_vals.max(),
        alpha=0.8,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("HSPF (Region IV)")
    ax.set_title(f"Heat pump options for Building {building_id}")
    ax.set_xlabel("Capacity [kBtu/hr]")
    ax.set_ylabel("Total Cost [$]")
    ax.axvline(
        x=peak_load_btu_hr / 1000,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Required capacity ({peak_load_btu_hr / 1000:,.1f} kBtu/hr)",
    )
    ax.legend()
    plt.tight_layout()
    plot_path = output_dir / f"{building_id}_heatpump_options.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Define all file and folder paths here ---
    BASE_DIR = Path(__file__).parents[1]
    neep_filepath = BASE_DIR / "in" / "neep_database.csv"
    metadata_filepath = BASE_DIR / "in" / "TX_upgrade0.parquet"
    consumption_dir = BASE_DIR / "out" / "consumption_files"

    # --- Building ID ---
    building_id = 77

    # --- Flags ---
    plot_flag = True

    # --- Run ---
    neep_df = load_neep_heat_pumps(neep_filepath)
    metadata_df = load_metadata(metadata_filepath)
    building = get_building_metadata(metadata_df, building_id)
    consumption_df = load_building_consumption(consumption_dir, building_id)
    n_neep_total = len(neep_df)
    neep_df, peak_load_btu_hr = get_capacity_at_design_temp(neep_df, consumption_df)

    neep_df = build_cost_menu(neep_df, building["conditioned_area_sqft"], building["has_ductwork"])

    cost_cols = ["Brand Name", "Outdoor Unit Model", "HSPF (Region IV)", "capacity_btu", "ccASHP", "cost_appliance", "cost_labor", "cost_ductwork", "cost_scaled_misc", "cost_flat_misc", "cost_total"]

    print("\nBuilding metadata:")
    print(building)
    print("\nCost menu (first 5 rows):")
    print(neep_df[cost_cols].head())

    # --- Save output ---
    heatpump_dir = BASE_DIR / "out" / "heatpump_files"
    heatpump_dir.mkdir(parents=True, exist_ok=True)
    output_path = heatpump_dir / f"{building_id}_heatpump_options.parquet"
    neep_df.to_parquet(output_path, engine="fastparquet", index=False)
    print(f"\nSaved heat pump options to {output_path}")

    if plot_flag:
        plot_heatpump_options(neep_df, building_id, peak_load_btu_hr, heatpump_dir)