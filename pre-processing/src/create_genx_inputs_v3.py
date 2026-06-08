from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np
import pvlib


# =============================================================================
# FILE PATHS
# =============================================================================

def get_paths(dispatch_mode: str, network_mode: str) -> dict:
    """Build all input/output paths for the case."""
    base = Path(__file__).parents[1]  # Z:\ercot_project

    in_dir = base / "in"
    scenario_dir = in_dir / "Texas7k_20210804_Plus2023" / "Texas7k_Scenario_Data"

    case_name = f"elec_0_flat_{dispatch_mode}_{network_mode}"
    out_dir = base / "out" / "genx_cases" / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output subdirectories (GenX v0.4 folder structure)
    resources_dir = out_dir / "resources"
    system_dir = out_dir / "system"
    resources_dir.mkdir(exist_ok=True)
    system_dir.mkdir(exist_ok=True)

    return {
        # Inputs
        "gen_tech_file":        scenario_dir / "Texas-7k_GenUCParameters_Nov2021.xlsx",
        "hourly_production":    scenario_dir / "TX7kMW2020.csv",
        "cost_curves":          scenario_dir / "Texas7k_CostCurves_Nov2021.xlsx",
        "eia860_plants":        in_dir / "eia8602024" / "2___Plant_Y2024.xlsx",
        "f923_generation":      in_dir / "f923_2024" / "EIA923_Schedules_2_3_4_5_M_12_2024_Final.xlsx",
        "snl_data":             in_dir / "sp_global_dataset_ercot.csv",
        # Outputs
        "out_thermal":          resources_dir / "Thermal.csv",
        "out_vre":              resources_dir / "Vre.csv",
        "out_storage":          resources_dir / "Storage.csv",
        "out_fuels":            system_dir / "Fuels_data.csv",
        "out_variability":      system_dir / "Generators_variability.csv",
        "out_demand":           system_dir / "Demand_data.csv",
        "out_network":          system_dir / "Network.csv",
        # Cache
        "nrel_cache":           out_dir / "nrel_profiles_cache.csv",
    }


# =============================================================================
# CONSTANTS
# =============================================================================

# Mapping from source Fuel Type Code -> GenX fuel name (Fuels_data.csv column)
FUEL_MAP = {
    "SUB":  "Coal",
    "BIT":  "Coal",
    "NG":   "NG",
    "NUC":  "Nuclear",
    "WAT":  "Hydro",
    "WND":  "Wind",
    "SUN":  "Solar",
    "PC":   "PC",
    "OTH":  "NG",       # Assume NG as conservative default
}

# Fuel prices (EIA 2020 averages, $/MMBtu) — flat across all hours
FUEL_PRICES = {
    "NG":      2.39,
    "Coal":    1.92,
    "PC":      1.50,
    "Nuclear": 0.71,
    "Hydro":   0.00,
    "Wind":    0.00,
    "Solar":   0.00,
}

# CO2 emission rates (metric tons CO2/MMBtu, EPA)
# Stored as Time_Index=0 row in Fuels_data.csv
CO2_RATES = {
    "NG":      0.05306,
    "Coal":    0.09535,
    "PC":      0.10241,
    "Nuclear": 0.00000,
    "Hydro":   0.00000,
    "Wind":    0.00000,
    "Solar":   0.00000,
}

# Fuel types treated as VRE (need variability profiles)
VRE_FUELS = {"WND", "SUN"}

# NREL ATB 2023 fixed O&M by unit type ($/MW-yr)
# Source: NREL Annual Technology Baseline 2023
# Used as fallback when SNL fixed O&M is missing
NREL_FIXED_OM = {
    "ST":   28000,   # Gas/oil steam turbine
    "GT":    7000,   # Gas combustion turbine
    "CA":   15800,   # Combined cycle (aggregate)
    "CT":   15800,   # Combined cycle gas turbine
    "IC":   10000,   # Internal combustion
    "NUC": 130000,   # Nuclear
    # Fuel-type fallbacks
    "SUB":  40500,   # Coal steam
    "BIT":  40500,   # Coal steam
    "NG":   15800,   # Natural gas (generic CC assumption)
    "WAT":  44000,   # Hydro
    "PC":   28000,   # Petroleum coke steam
    "OTH":  15800,   # Other
}

# Fuel types treated as thermal/dispatchable
THERMAL_FUELS = {"SUB", "BIT", "NG", "NUC", "WAT", "PC", "OTH"}

# EIA 2020 average heat rates by unit type (MMBtu/MWh)
# Source: EIA Electric Power Annual 2020, Table 8.1
# Used as fallback when heat rate is missing or zero in source data
EIA_HEAT_RATES = {
    "ST":   10.41,   # Steam turbine (coal/gas/oil)
    "GT":   11.31,   # Gas turbine / combustion turbine
    "CA":    7.65,   # Combined cycle steam portion
    "CT":    7.65,   # Combined cycle gas turbine portion
    "IC":   10.00,   # Internal combustion
    "NUC":  10.46,   # Nuclear (heat rate is conventional, not thermodynamic)
    # Fuel-type based fallbacks
    "SUB":  10.41,   # Subbituminous coal steam
    "BIT":  10.41,   # Bituminous coal steam
    "NG":   10.00,   # Natural gas (generic)
    "WAT":   0.00,   # Hydro (no heat rate)
    "PC":   11.50,   # Petroleum coke
    "OTH":  10.00,   # Other
}

# EIA 2020 average variable O&M by unit type ($/MWh)
# Source: EIA, NREL ATB 2020
EIA_VAR_OM = {
    "ST":   4.00,    # Steam turbine
    "GT":   5.00,    # Gas turbine
    "CA":   4.50,    # Combined cycle
    "CT":   4.50,    # Combined cycle
    "IC":   5.00,    # Internal combustion
    "NUC":  2.00,    # Nuclear
    "SUB":  4.00,
    "BIT":  4.00,
    "NG":   4.00,
    "WAT":  0.00,
    "PC":   4.00,
    "OTH":  4.00,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_gen_tech(path: Path) -> pd.DataFrame:
    """
    Load generator technical parameters.
    Skips the 'Source' metadata row (row index 1 in Excel).
    """
    df = pd.read_excel(path, header=0)
    # Drop the source citation row if present
    df = df[df["BusNum"].apply(lambda x: str(x).isnumeric())].copy()
    df["BusNum"] = df["BusNum"].astype(int)
    df["GenMWMax"] = df["GenMWMax"].astype(float)
    return df


def load_cost_curves(path: Path) -> pd.DataFrame:
    """
    Load ERCOT bid/cost curve data.
    Contains startup costs and TPO (Tranched Price Offer) for Min_Power derivation.
    """
    df = pd.read_excel(path, header=0)
    df = df[df["BusNum"].apply(lambda x: str(x).isnumeric())].copy()
    df["BusNum"] = df["BusNum"].astype(int)
    df["GenMWMax"] = df["GenMWMax"].astype(float)
    return df


def load_hourly_production(path: Path) -> pd.DataFrame:
    """
    Load PowerWorld OPF hourly dispatch output (8760 rows).
    Auto-detects the real header row by finding the row containing 'Date'.
    Skips the PWOPFTimePoint metadata row at the top.
    """
    # Read first few rows to find the real header
    df_raw = pd.read_csv(path, header=None, low_memory=False, nrows=5)
    header_row = None
    for i, row in df_raw.iterrows():
        if "Date" in row.values or "Total MW Load" in row.values:
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not find header row in OPF file. "
                         "Expected a row containing 'Date' or 'Total MW Load'.")

    print(f"  OPF header found at row index: {header_row}")
    df = pd.read_csv(path, header=header_row, low_memory=False)

    # Drop any non-numeric rows (e.g. stray metadata)
    df = df[pd.to_numeric(df["Total MW Load"], errors="coerce").notna()].copy()
    df["Total MW Load"] = pd.to_numeric(df["Total MW Load"])
    df = df.reset_index(drop=True)
    return df


# =============================================================================
# HELPERS
# =============================================================================

def make_resource_name(row: pd.Series) -> str:
    """Create a unique resource identifier from plant name + EIA generator ID."""
    plant = str(row["EIA-860 Plant Name"]).strip().replace(" ", "_")
    gen_id = str(row["EIA Generator ID"]).strip()
    return f"{plant}_{gen_id}"


def derive_min_power(gen_row: pd.Series, cost_df: pd.DataFrame) -> float:
    """
    Derive minimum generation fraction from TPO MW1 (first offer tranche).
    Falls back to technology-based defaults if not found.
    """
    defaults = {
        "ST":  0.40,   # Steam turbine
        "GT":  0.10,   # Gas turbine / combustion turbine
        "CA":  0.30,   # Combined cycle (steam portion)
        "CT":  0.30,   # Combined cycle (gas portion)
        "NUC": 0.95,   # Nuclear (near must-run)
    }

    match = cost_df[
        (cost_df["BusNum"] == gen_row["BusNum"]) &
        (cost_df["GenID"] == gen_row["GenID"])
    ]

    if not match.empty and "Submitted TPO-MW1" in match.columns:
        tpo_mw1 = match.iloc[0]["Submitted TPO-MW1"]
        cap = gen_row["GenMWMax"]
        if pd.notna(tpo_mw1) and float(tpo_mw1) > 0 and cap > 0:
            return min(float(tpo_mw1) / cap, 1.0)

    # Fall back to technology default
    unit_type = str(gen_row.get("Unit Type Code", "")).strip().upper()
    return defaults.get(unit_type, 0.20)


def get_startup_costs(gen_row: pd.Series, cost_df: pd.DataFrame) -> dict:
    """Extract cold/warm/hot startup costs from cost curves file."""
    match = cost_df[
        (cost_df["BusNum"] == gen_row["BusNum"]) &
        (cost_df["GenID"] == gen_row["GenID"])
    ]

    if not match.empty:
        row = match.iloc[0]
        return {
            "cold": float(row.get("Start Up Cold Offer", 0) or 0),
            "warm": float(row.get("Start Up Inter Offer", 0) or 0),
            "hot":  float(row.get("Start Up Hot Offer", 0) or 0),
        }
    return {"cold": 0.0, "warm": 0.0, "hot": 0.0}


# =============================================================================
# THERMAL.CSV
# =============================================================================

# Module-level F923 heat rate lookup table (set in main before building thermal)
_F923_HEAT_RATES: pd.DataFrame = None

# Module-level SNL data lookup table (set in main before building thermal)
_SNL_DATA: pd.DataFrame = None


def load_snl_data(path: Path) -> pd.DataFrame:
    """
    Load S&P Global / SNL Energy ERCOT power plant dataset.
    Joins to generator data via EIA_SITE_CODE = EIA-860 Plant Code.
    Key columns used:
      - HEAT_RATE_WHOLE_ANNUAL            -> Heat_Rate_MMBTU_per_MWh
      - NON_FUEL_NON_ALLOW_VARIABLE_O_AND_M_COST_PER_MWH_WHOLE_GSC -> Var_OM_Cost_per_MWh
      - FIXED_O_AND_M_COST_PER_KW_YR_WHOLE_GSC -> Fixed_OM_Cost_per_MWyr (x1000)
    Returns DataFrame indexed by EIA_SITE_CODE (int).
    """
    print("  Loading S&P Global SNL dataset...")
    df = pd.read_csv(path, dtype={"EIA_SITE_CODE": str}, low_memory=False, thousands=",")

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Convert EIA site code to int
    df["EIA_SITE_CODE"] = pd.to_numeric(df["EIA_SITE_CODE"], errors="coerce")
    df = df.dropna(subset=["EIA_SITE_CODE"])
    df["EIA_SITE_CODE"] = df["EIA_SITE_CODE"].astype(int)

    # Convert numeric columns
    numeric_cols = [
        "HEAT_RATE_WHOLE_ANNUAL",
        "NON_FUEL_NON_ALLOW_VARIABLE_O_AND_M_COST_PER_MWH_WHOLE_GSC",
        "FIXED_O_AND_M_COST_PER_KW_YR_WHOLE_GSC",
        "FUEL_COST_PER_MWH_WHOLE_GSC",
        "CO2_EMISSIONS_RATES",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert fixed O&M: SNL $/kW-yr -> GenX $/MW-yr (* 1000)
    if "FIXED_O_AND_M_COST_PER_KW_YR_WHOLE_GSC" in df.columns:
        df["fixed_om_per_mw_yr"] = df["FIXED_O_AND_M_COST_PER_KW_YR_WHOLE_GSC"] * 1000

    # Aggregate to plant level if multiple rows per plant
    # Use mean for cost/rate columns
    # Add unit type for fallback grouping
    if "TECH_TYPE" in df.columns:
        df["unit_type"] = df["TECH_TYPE"].str.strip().str.upper()

    agg_cols = {
        "NON_FUEL_NON_ALLOW_VARIABLE_O_AND_M_COST_PER_MWH_WHOLE_GSC": "mean",
        "fixed_om_per_mw_yr": "mean",
        "FUEL_COST_PER_MWH_WHOLE_GSC": "mean",
        "CO2_EMISSIONS_RATES": "mean",
        "unit_type": "first",
    }
    agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns or k == "fixed_om_per_mw_yr"}

    # Only aggregate columns that exist
    valid_agg = {}
    for col, func in agg_cols.items():
        if col in df.columns:
            valid_agg[col] = func

    if valid_agg:
        snl = df.groupby("EIA_SITE_CODE").agg(valid_agg)
    else:
        snl = df.set_index("EIA_SITE_CODE")

    print(f"  Loaded {len(snl)} plant records from SNL")
    return snl


def get_snl_value(plant_code: int, col: str, default=None):
    """Look up a value from the SNL dataset by plant code."""
    if _SNL_DATA is None or plant_code not in _SNL_DATA.index:
        return default
    val = _SNL_DATA.loc[plant_code, col] if col in _SNL_DATA.columns else default
    if pd.isna(val):
        return default
    return float(val)


def _get_heat_rate(row: pd.Series) -> float:
    """
    Get heat rate for a generator row using this priority:
      1. Source data (gen tech file) if non-zero
      2. EIA Form 923 plant-specific actual heat rate
      3. EIA technology-type average
    """
    hr = float(row.get("Heat Rate", 0) or 0)
    if hr > 0:
        return hr

    # Try EIA 923 plant-specific heat rate
    plant_code = row.get("EIA-860 Plant Code")
    if _F923_HEAT_RATES is not None and pd.notna(plant_code):
        f923_hr = get_f923_heat_rate(int(plant_code), _F923_HEAT_RATES)
        if f923_hr is not None:
            return f923_hr

    # Fallback by unit type then fuel type
    unit_type = str(row.get("Unit Type Code", "")).strip().upper()
    if unit_type in EIA_HEAT_RATES:
        return EIA_HEAT_RATES[unit_type]
    fuel_code = str(row.get("Fuel Type Code", "")).strip().upper()
    return EIA_HEAT_RATES.get(fuel_code, 10.0)


def _get_var_om(row: pd.Series) -> float:
    """
    Get variable O&M for a generator row. Priority:
      1. Source data if non-zero
      2. SNL plant-specific
      3. EIA averages by unit/fuel type
    """
    vom = float(row.get("Variable O&M ($/MWh)", 0) or 0)
    if vom > 0:
        return vom
    # Try SNL
    plant_code = row.get("EIA-860 Plant Code")
    if _SNL_DATA is not None and pd.notna(plant_code):
        snl_vom = get_snl_value(
            int(plant_code),
            "NON_FUEL_NON_ALLOW_VARIABLE_O_AND_M_COST_PER_MWH_WHOLE_GSC"
        )
        if snl_vom and snl_vom > 0:
            return snl_vom
    # EIA averages
    unit_type = str(row.get("Unit Type Code", "")).strip().upper()
    if unit_type in EIA_VAR_OM:
        return EIA_VAR_OM[unit_type]
    fuel_code = str(row.get("Fuel Type Code", "")).strip().upper()
    return EIA_VAR_OM.get(fuel_code, 4.0)


def _get_fixed_om(row: pd.Series) -> float:
    """
    Get fixed O&M ($/MW-yr) for a generator row. Priority:
      1. SNL plant-specific (98% coverage for ERCOT)
      2. SNL average for same unit type (from populated plants)
      3. NREL ATB average by unit type
      4. NREL ATB average by fuel type
    """
    plant_code = row.get("EIA-860 Plant Code")

    # 1. SNL plant-specific
    if _SNL_DATA is not None and pd.notna(plant_code):
        snl_fom = get_snl_value(int(plant_code), "fixed_om_per_mw_yr")
        if snl_fom and snl_fom > 0:
            return snl_fom

    # 2. SNL average for same unit type
    unit_type = str(row.get("Unit Type Code", "")).strip().upper()
    if _SNL_DATA is not None and "unit_type" in _SNL_DATA.columns:
        same_type = _SNL_DATA[
            (_SNL_DATA["unit_type"] == unit_type) &
            (_SNL_DATA["fixed_om_per_mw_yr"] > 0)
        ]["fixed_om_per_mw_yr"]
        if not same_type.empty:
            return float(same_type.mean())

    # 3. NREL ATB by unit type
    if unit_type in NREL_FIXED_OM:
        return NREL_FIXED_OM[unit_type]

    # 4. NREL ATB by fuel type
    fuel_code = str(row.get("Fuel Type Code", "")).strip().upper()
    return float(NREL_FIXED_OM.get(fuel_code, 15800))


def build_thermal(
    gen_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    network_mode: str,
    dispatch_mode: str,
) -> pd.DataFrame:
    """
    Build Thermal.csv for all dispatchable generators.
    Zone is set to BusNum for 7kbus mode, or 1 for single-bus mode.
    Model=1 for ED (linear dispatch), Model=2 for UC (unit commitment).
    """
    model_flag = 1 if dispatch_mode == "ed" else 2

    thermal_df = gen_df[
        gen_df["Fuel Type Code"].isin(THERMAL_FUELS)
    ].copy()

    records = []
    for _, row in thermal_df.iterrows():
        cap = float(row["GenMWMax"])
        ramp_rate = float(row.get("Ramprate (MW/hr)", cap))
        ramp_pct = min(ramp_rate / cap, 1.0) if cap > 0 else 1.0
        fuel_code = str(row.get("Fuel Type Code", "NG")).strip()
        fuel_name = FUEL_MAP.get(fuel_code, "NG")
        startup = get_startup_costs(row, cost_df)
        min_power = derive_min_power(row, cost_df)
        zone = 1 if network_mode == "1bus" else int(row["BusNum"])

        records.append({
            "Resource":               make_resource_name(row),
            "Zone":                   zone,
            "Model":                  model_flag,
            "New_Build":              0,
            "Can_Retire":             0,
            "Existing_Cap_MW":        cap,
            "Max_Cap_MW":             -1,
            "Min_Cap_MW":             0,
            "Inv_Cost_per_MWyr":      0,
            "Fixed_OM_Cost_per_MWyr": _get_fixed_om(row),
            "Var_OM_Cost_per_MWh":    _get_var_om(row),
            "Heat_Rate_MMBTU_per_MWh":_get_heat_rate(row),
            "Fuel":                   fuel_name,
            "Cap_Size":               cap,
            "Start_Cost_per_MW":      float(row.get("Fixed Startup Cost ($/MW of capacity/start)", 0) or 0),
            "Start_Fuel_MMBTU_per_MW":float(row.get("Turn On Fuel Cost (MMBtu/MW)", 0) or 0),
            "Up_Time":                float(row.get("Minimum Up Time (hrs)", 1) or 1),
            "Down_Time":              float(row.get("Minimum Down Time (hrs)", 1) or 1),
            "Ramp_Up_Percentage":     ramp_pct,
            "Ramp_Dn_Percentage":     ramp_pct,
            "Min_Power":              min_power,
            "Reg_Max":                0,
            "Rsv_Max":                0,
            "Reg_Cost":               0,
            "Rsv_Cost":               0,
            "region":                 "ERCOT",
            "cluster":                str(row.get("Unit Type Code", "")).strip(),
        })

    return pd.DataFrame(records)


# =============================================================================
# VRE.CSV
# =============================================================================

def build_vre(
    gen_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    network_mode: str,
) -> pd.DataFrame:
    """
    Build Vre.csv for wind and solar generators.
    Identifies VRE bus columns in the OPF output and computes capacity factors.
    """
    vre_gen_df = gen_df[gen_df["Fuel Type Code"].isin(VRE_FUELS)].copy()

    records = []
    for _, row in vre_gen_df.iterrows():
        cap = float(row["GenMWMax"])
        fuel_code = str(row.get("Fuel Type Code", "")).strip()
        fuel_name = FUEL_MAP.get(fuel_code, "Wind")
        zone = 1 if network_mode == "1bus" else int(row["BusNum"])

        records.append({
            "Resource":               make_resource_name(row),
            "Zone":                   zone,
            "New_Build":              0,
            "Can_Retire":             0,
            "Existing_Cap_MW":        cap,
            "Max_Cap_MW":             -1,
            "Min_Cap_MW":             0,
            "Inv_Cost_per_MWyr":      0,
            "Fixed_OM_Cost_per_MWyr": _get_fixed_om(row),
            "Var_OM_Cost_per_MWh":    float(row.get("Variable O&M ($/MWh)", 0) or 0),
            "Fuel":                   fuel_name,
            "Cap_Size":               cap,
            "Reg_Max":                0,
            "Rsv_Max":                0,
            "Reg_Cost":               0,
            "Rsv_Cost":               0,
            "region":                 "ERCOT",
            "cluster":                str(row.get("Unit Type Code", "")).strip(),
        })

    return pd.DataFrame(records)


# =============================================================================
# EIA-860 PLANT COORDINATES
# =============================================================================

def load_eia860_plants(path: Path) -> pd.DataFrame:
    """
    Load EIA-860 plant-level data to get lat/long by Plant Code.
    The 2___Plant_Y2024.xlsx file has a Plant sheet with:
      Plant Code, Plant Name, Latitude, Longitude, State
    Skips the first row which is a disclaimer/metadata row.
    """
    for sheet in ["Plant", "Plants", "2___Plant_Y2024"]:
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=1)
            df.columns = [str(c).strip() for c in df.columns]
            if "Plant Code" in df.columns and "Latitude" in df.columns:
                df = df[["Plant Code", "Plant Name", "State",
                          "Latitude", "Longitude"]].dropna(subset=["Plant Code"])
                df["Plant Code"] = pd.to_numeric(df["Plant Code"], errors="coerce")
                df = df.dropna(subset=["Plant Code"])
                df["Plant Code"] = df["Plant Code"].astype(int)
                print(f"  EIA-860 plants loaded from sheet {sheet!r}: {len(df)} plants")
                return df.set_index("Plant Code")
        except Exception:
            continue
    raise ValueError(f"Could not find plant sheet with Plant Code and Latitude in {path}")


def load_f923_heat_rates(path: Path) -> pd.DataFrame:
    """
    Load EIA Form 923 Page 1 Generation and Fuel Data.
    Calculates annual heat rate per plant as:
      Heat Rate = Elec Fuel Consumption MMBtu / Net Generation MWh
    Returns DataFrame indexed by Plant Id with columns:
      heat_rate_mmbtu_per_mwh, elec_mmbtu, net_gen_mwh, fuel_type, prime_mover
    Only includes plants with positive net generation.
    """
    print("  Loading EIA Form 923 heat rates...")
    df = pd.read_excel(
        path,
        sheet_name="Page 1 Generation and Fuel Data",
        header=5,          # row 6 (0-indexed = 5) is the header
        dtype={"Plant Id": str},
    )

    # Normalize column names
    df.columns = [str(c).strip().replace("\n", " ").replace("  ", " ")
                  for c in df.columns]

    # Key columns
    elec_mmbtu_col = "Elec Fuel Consumption MMBtu"
    netgen_col     = "Net Generation (Megawatthours)"
    plant_col      = "Plant Id"
    fuel_col       = "Reported Fuel Type Code"
    mover_col      = "Reported Prime Mover"

    # Keep only rows with valid plant ID and positive generation
    df = df[df[plant_col].notna()].copy()
    df[plant_col] = pd.to_numeric(df[plant_col], errors="coerce")
    df = df.dropna(subset=[plant_col])
    df[plant_col] = df[plant_col].astype(int)

    for col in [elec_mmbtu_col, netgen_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Aggregate to plant level (sum across fuel types/months already done)
    agg = df.groupby(plant_col).agg(
        elec_mmbtu   =(elec_mmbtu_col, "sum"),
        net_gen_mwh  =(netgen_col,     "sum"),
        fuel_type    =(fuel_col,        "first"),
        prime_mover  =(mover_col,       "first"),
    ).reset_index()

    # Calculate heat rate — only where net generation > 0
    agg = agg[agg["net_gen_mwh"] > 0].copy()
    agg["heat_rate_mmbtu_per_mwh"] = agg["elec_mmbtu"] / agg["net_gen_mwh"]

    # Sanity check — filter out implausible values (< 3 or > 30 MMBtu/MWh)
    n_before = len(agg)
    agg = agg[(agg["heat_rate_mmbtu_per_mwh"] >= 3) &
              (agg["heat_rate_mmbtu_per_mwh"] <= 30)]
    n_after = len(agg)
    if n_before > n_after:
        print(f"  Filtered {n_before - n_after} plants with implausible heat rates")

    agg = agg.set_index(plant_col)
    print(f"  Loaded {len(agg)} plant-level heat rates from EIA 923")
    return agg


def get_f923_heat_rate(plant_code: int, f923_df: pd.DataFrame,
                       fallback: float = None) -> float:
    """
    Look up plant-specific heat rate from EIA 923.
    Returns fallback value if plant not found.
    """
    if plant_code in f923_df.index:
        return float(f923_df.loc[plant_code, "heat_rate_mmbtu_per_mwh"])
    return fallback


def get_plant_coords(eia_plant_code: int, eia860_df: pd.DataFrame) -> tuple:
    """Return (latitude, longitude) for a given EIA-860 Plant Code."""
    if eia_plant_code in eia860_df.index:
        row = eia860_df.loc[eia_plant_code]
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        if pd.notna(lat) and pd.notna(lon):
            return lat, lon
    return None, None


# =============================================================================
# GENERATORS_VARIABILITY.CSV
# =============================================================================

def build_variability(
    gen_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    eia860_df: pd.DataFrame,
    nrel_profiles: dict = None,
) -> pd.DataFrame:
    """
    Build Generators_variability.csv for VRE generators.

    Priority order for capacity factor profiles:
      1. OPF hourly dispatch output (matched by bus/gen ID)
      2. NREL API profiles (NSRDB solar or Wind Toolkit wind)
      3. Same-fuel-type donor profile from OPF
      4. CF=0 with warning (only if no other source available)
    """
    vre_gen_df = gen_df[gen_df["Fuel Type Code"].isin(VRE_FUELS)].copy()

    # --- Pass 1: match what we can directly from OPF ---
    variability = {}
    fuel_type_map = {}  # resource_name -> fuel type code
    matched_by_fuel = {}  # fuel_type_code -> list of matched CF arrays

    for _, row in vre_gen_df.iterrows():
        bus_num = int(row["BusNum"])
        gen_id = int(row["GenID"])
        cap = float(row["GenMWMax"])
        resource_name = make_resource_name(row)
        fuel_code = str(row.get("Fuel Type Code", "")).strip()
        fuel_type_map[resource_name] = fuel_code

        # Try multiple suffixes
        col_name = None
        for suffix in [gen_id, 5, 1]:
            candidate = f"Bus {bus_num} #{suffix} MW"
            if candidate in hourly_df.columns:
                col_name = candidate
                break

        if col_name and cap > 0:
            cf = (pd.to_numeric(hourly_df[col_name], errors="coerce")
                    .fillna(0) / cap).clip(0, 1).values
            variability[resource_name] = cf
            matched_by_fuel.setdefault(fuel_code, []).append(cf)
        else:
            variability[resource_name] = None  # placeholder for pass 2

    # --- Pass 2: fill unmatched using NREL profiles, then same-fuel donor ---
    n_hours = len(hourly_df)
    nrel_profiles = nrel_profiles or {}
    from_nrel = []
    from_donor = []
    still_zero = []

    for resource_name, cf in variability.items():
        if cf is None:
            fuel_code = fuel_type_map[resource_name]

            # Priority 1: NREL profile for this specific plant
            if resource_name in nrel_profiles:
                variability[resource_name] = nrel_profiles[resource_name]
                from_nrel.append(resource_name)

            # Priority 2: same-fuel OPF donor
            elif matched_by_fuel.get(fuel_code):
                variability[resource_name] = matched_by_fuel[fuel_code][0]
                from_donor.append(resource_name)

            # Priority 3: same-fuel NREL donor (another plant of same type)
            elif any(fuel_type_map.get(k) == fuel_code for k in nrel_profiles):
                donor_cf = next(
                    v for k, v in nrel_profiles.items()
                    if fuel_type_map.get(k) == fuel_code
                )
                variability[resource_name] = donor_cf
                from_donor.append(resource_name)

            # No source found
            else:
                variability[resource_name] = np.zeros(n_hours)
                still_zero.append(resource_name)

    # Summary
    matched_opf = len(vre_gen_df) - len(from_nrel) - len(from_donor) - len(still_zero)
    print(f"  VRE capacity factors: {matched_opf} from OPF, "
          f"{len(from_nrel)} from NREL, "
          f"{len(from_donor)} from same-fuel donor, "
          f"{len(still_zero)} CF=0")
    for resource_name in still_zero:
        print(f"  WARNING: No profile found for {resource_name} -- CF set to 0.")

    variability_df = pd.DataFrame(variability)
    variability_df.index = range(1, len(variability_df) + 1)
    variability_df.index.name = "Time_Index"
    return variability_df


# =============================================================================
# STORAGE.CSV (empty template)
# =============================================================================

def build_storage() -> pd.DataFrame:
    """Return empty Storage.csv with correct headers."""
    cols = [
        "Resource", "Zone", "New_Build", "Can_Retire",
        "Existing_Cap_MW", "Existing_Cap_MWh",
        "Max_Cap_MW", "Min_Cap_MW", "Max_Cap_MWh", "Min_Cap_MWh",
        "Inv_Cost_per_MWyr", "Inv_Cost_per_MWhyr",
        "Fixed_OM_Cost_per_MWyr", "Fixed_OM_Cost_per_MWhyr",
        "Var_OM_Cost_per_MWh", "Var_OM_Cost_per_MWh_charge",
        "Eff_Up", "Eff_Down", "Self_Disch",
        "Min_Duration", "Max_Duration",
        "Reg_Max", "Rsv_Max", "Reg_Cost", "Rsv_Cost",
        "region", "cluster",
    ]
    return pd.DataFrame(columns=cols)


# =============================================================================
# FUELS_DATA.CSV
# =============================================================================

def build_fuels(n_hours: int = 8760) -> pd.DataFrame:
    """
    Build Fuels_data.csv.
    Row 0 (Time_Index=0): CO2 content in tons/MMBtu per fuel.
    Rows 1-8760: flat hourly fuel price in $/MMBtu.
    """
    fuels = list(FUEL_PRICES.keys())

    rows = []

    # Row 0: CO2 rates
    co2_row = {"Time_Index": 0}
    for fuel in fuels:
        co2_row[fuel] = CO2_RATES.get(fuel, 0.0)
    rows.append(co2_row)

    # Rows 1–8760: flat fuel prices
    for t in range(1, n_hours + 1):
        price_row = {"Time_Index": t}
        for fuel in fuels:
            price_row[fuel] = FUEL_PRICES.get(fuel, 0.0)
        rows.append(price_row)

    return pd.DataFrame(rows).set_index("Time_Index")


# =============================================================================
# DEMAND_DATA.CSV
# =============================================================================

def build_demand(hourly_df: pd.DataFrame, network_mode: str) -> pd.DataFrame:
    """
    Build Demand_data.csv in GenX v0.4 format.

    GenX expects a specific layout:
      - Row 1 contains metadata (Voll, Demand_Segment, curtailment params,
        Rep_Periods, Timesteps_per_Rep_Period, Sub_Weights) plus the first
        hour's demand.
      - Rows 2–8760 leave the metadata columns blank and only fill
        Time_Index and Demand_MW_z1.
    """
    if "Total MW Load" not in hourly_df.columns:
        raise ValueError("'Total MW Load' column not found in hourly production file.")

    load = pd.to_numeric(hourly_df["Total MW Load"], errors="coerce").fillna(0).values[:8760]
    n_hours = len(load)

    cols = [
        "Voll", "Demand_Segment", "Cost_of_Demand_Curtailment_per_MW",
        "Max_Demand_Curtailment", "$/MWh",
        "Rep_Periods", "Timesteps_per_Rep_Period", "Sub_Weights",
        "Time_Index", "Demand_MW_z1",
    ]

    # Build rows — metadata only in first row, blank thereafter
    rows = []
    for t in range(n_hours):
        if t == 0:
            rows.append([50000, 1, 1, 1, 2000, 1, n_hours, n_hours, t + 1, load[t]])
        else:
            rows.append(["", "", "", "", "", "", "", "", t + 1, load[t]])

    demand_df = pd.DataFrame(rows, columns=cols)
    return demand_df



# =============================================================================
# NETWORK.CSV
# =============================================================================

def build_network(network_mode: str, gen_df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder — network data is handled by a separate script."""
    pass


# =============================================================================
# NREL API — SOLAR (NSRDB) AND WIND CAPACITY FACTORS
# =============================================================================

NSRDB_URL       = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.csv"
NSRDB_GOES_URL  = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"
WIND_URL   = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-download.csv"
BCHRRR_URL = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/bc-hrrr-download.csv"

# Generic utility-scale PV assumptions
SOLAR_TILT       = 25       # degrees
SOLAR_AZIMUTH    = 180      # south-facing
SOLAR_EFFICIENCY = 0.18     # panel efficiency
SOLAR_ILF        = 1.1      # DC/AC ratio (inverter loading factor)

# Generic IEC Class II wind turbine power curve (wind speed m/s -> CF)
# Simplified piecewise: cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s
def wind_speed_to_cf(ws: float) -> float:
    """Simple cubic power curve: CF = (ws/rated)^3, clipped to [0,1]."""
    if ws < 3.0 or ws > 25.0:
        return 0.0
    elif ws >= 12.0:
        return 1.0
    else:
        return min((ws / 12.0) ** 3, 1.0)


def fetch_nsrdb_solar_cf(lat: float, lon: float, year: int,
                          api_key: str, email: str,
                          tilt: float = SOLAR_TILT,
                          azimuth: float = SOLAR_AZIMUTH) -> np.ndarray:
    """
    Fetch hourly solar capacity factors from NREL NSRDB PSM3 for a given
    lat/lon and year. Uses pvlib to convert GHI/DNI/DHI + temperature
    into DC power output, normalized to installed capacity.
    Returns an array of 8760 capacity factors (0-1).
    """
    params = {
        "wkt":          f"POINT({lon} {lat})",
        "names":        str(year),
        "leap_day":     "false",
        "interval":     "60",
        "utc":          "true",
        "full_name":    "GenX+User",
        "email":        email,
        "affiliation":  "research",
        "mailing_list": "false",
        "reason":       "energy+modeling",
        "api_key":      api_key,
        "attributes":   "ghi,dhi,dni,wind_speed,air_temperature",
    }

    # Try GOES Aggregated v4 first (1998-present), then PSM3 v2.2 as fallback
    # Try multiple years for each endpoint in case a specific year is unavailable
    attempts = (
        [(NSRDB_GOES_URL, y) for y in [2023, 2022, 2021, 2020, 2019, 2018]] +
        [(NSRDB_URL,      y) for y in [2023, 2022, 2021, 2020, 2019, 2018]]
    )
    resp = None
    for url, try_year in attempts:
        params["names"] = str(try_year)
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            break
    resp.raise_for_status()

    # NSRDB returns 2 header rows then data
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), skiprows=2)
    df.index = pd.date_range(f"1/1/{year}", periods=len(df), freq="h", tz="UTC")

    # Use pvlib to compute plane-of-array irradiance and cell temperature
    location = pvlib.location.Location(latitude=lat, longitude=lon, tz="UTC")
    solar_position = location.get_solarposition(df.index)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=df["DNI"],
        ghi=df["GHI"],
        dhi=df["DHI"],
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
    )

    cell_temp = pvlib.temperature.sapm_cell(
        poa_global=poa["poa_global"],
        temp_air=df["Temperature"],
        wind_speed=df["Wind Speed"],
        a=-3.47, b=-0.0594, deltaT=3,
    )

    # DC power using simple linear model: P = G * efficiency * (1 - temp_coeff*(T-25))
    temp_coeff = 0.0045
    dc_power = poa["poa_global"] * SOLAR_EFFICIENCY * (1 - temp_coeff * (cell_temp - 25))
    dc_power = dc_power.clip(lower=0)

    # Normalize: rated DC power = 1000 W/m2 * efficiency * area
    # CF = dc_power / (1000 * efficiency) / ILF  (DC/AC)
    rated_dc = 1000 * SOLAR_EFFICIENCY
    cf = (dc_power / rated_dc / SOLAR_ILF).clip(0, 1).values[:8760]
    return cf


# Wind Toolkit latest available year (API returns 400 for years beyond this)
WIND_MAX_YEAR = 2014  # WTK covers 2007-2014 only

def fetch_wind_cf(lat: float, lon: float, year: int,
                  api_key: str, email: str,
                  hub_height: int = 100) -> np.ndarray:
    """
    Fetch hourly wind capacity factors from NREL for a given lat/lon and year.
    Tries in order:
      1. BC-HRRR (2015-2023) — bias-corrected NOAA HRRR data
      2. WTK (2007-2014) — original Wind Integration National Dataset
    Converts wind speed at hub height to CF using a simplified cubic power curve.
    Returns an array of 8760 capacity factors (0-1).
    """
    from io import StringIO

    params = {
        "wkt":          f"POINT({lon} {lat})",
        "leap_day":     "false",
        "interval":     "60",
        "utc":          "true",
        "full_name":    "GenX+User",
        "email":        email,
        "affiliation":  "research",
        "mailing_list": "false",
        "reason":       "energy+modeling",
        "api_key":      api_key,
        "attributes":   f"windspeed_{hub_height}m",
    }

    # Try BC-HRRR first (2015-2023), then WTK (2007-2014)
    attempts = (
        [(BCHRRR_URL, y) for y in [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015]] +
        [(WIND_URL,   y) for y in [2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]]
    )

    resp = None
    for url, try_year in attempts:
        params["names"] = str(try_year)
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            break

    resp.raise_for_status()

    # WTK CSV has metadata rows before the actual header
    # Detect header row by finding the row that contains 'Year' or 'wind'
    lines = resp.text.split("\n")
    header_row = 0
    for i, line in enumerate(lines):
        if "Year" in line or "wind" in line.lower() or "speed" in line.lower():
            header_row = i
            break

    df = pd.read_csv(StringIO(resp.text), skiprows=header_row)
    ws_col = [c for c in df.columns if "wind" in c.lower() or "speed" in c.lower()]
    if not ws_col:
        raise ValueError(f"No wind speed column found. Columns: {list(df.columns)}")

    ws = pd.to_numeric(df[ws_col[0]], errors="coerce").fillna(0)
    cf = np.array([wind_speed_to_cf(float(v)) for v in ws])[:8760]
    return cf


def fetch_nrel_profiles(
    gen_df: pd.DataFrame,
    eia860_df: pd.DataFrame,
    api_key: str,
    email: str,
    year: int,
    rate_limit_sleep: float = 1.1,
) -> dict:
    """
    For all VRE generators, fetch hourly capacity factor profiles from NREL:
      - Solar (SUN) -> NSRDB PSM3
      - Wind  (WND) -> Wind Toolkit

    Returns a dict: {resource_name: np.ndarray of 8760 CFs}
    Respects NREL rate limit of 1 request/second.
    """
    vre_gen_df = gen_df[gen_df["Fuel Type Code"].isin(VRE_FUELS)].copy()
    profiles = {}
    n = len(vre_gen_df)

    for i, (_, row) in enumerate(vre_gen_df.iterrows()):
        resource_name = make_resource_name(row)
        fuel_code = str(row.get("Fuel Type Code", "")).strip()
        plant_code = int(row["EIA-860 Plant Code"])
        lat, lon = get_plant_coords(plant_code, eia860_df)

        if lat is None:
            print(f"  [{i+1}/{n}] SKIP {resource_name} — no coordinates")
            profiles[resource_name] = np.zeros(8760)
            continue

        print(f"  [{i+1}/{n}] Fetching {fuel_code} profile for "
              f"{resource_name} (lat={lat:.3f}, lon={lon:.3f})...", end=" ")

        try:
            if fuel_code == "SUN":
                cf = fetch_nsrdb_solar_cf(lat, lon, year, api_key, email)
            else:  # WND
                cf = fetch_wind_cf(lat, lon, year, api_key, email)
            profiles[resource_name] = cf
            print(f"OK (mean CF={cf.mean():.3f})")
        except Exception as e:
            print(f"FAILED: {e}")
            profiles[resource_name] = np.zeros(8760)

        # Respect rate limit
        time.sleep(rate_limit_sleep)

    return profiles


# =============================================================================
# THERMAL DATA VALIDATION & IMPUTATION
# =============================================================================

# Columns that must not be missing for GenX thermal no-commit to work
THERMAL_REQUIRED_COLS = [
    "Ramp_Up_Percentage",
    "Ramp_Dn_Percentage",
    "Min_Power",
    "Cap_Size",
    "Heat_Rate_MMBTU_per_MWh",
    "Var_OM_Cost_per_MWh",
    "Up_Time",
    "Down_Time",
]

def impute_thermal_missing(thermal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in required thermal columns.
    For any generator with missing values, find the most similar donor:
      1. Same EIA plant (same plant name prefix)
      2. Same unit type (cluster column)
      3. Same fuel type
      4. Any valid generator
    Copies missing values from the donor and logs what was done.
    """
    df = thermal_df.copy()
    issues_found = False

    for col in THERMAL_REQUIRED_COLS:
        if col not in df.columns:
            continue
        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue

        issues_found = True
        missing_resources = df[missing_mask]["Resource"].tolist()
        print(f"  WARNING: Missing values in '{col}' for {len(missing_resources)} generators:")

        for idx in df[missing_mask].index:
            row = df.loc[idx]
            resource = row["Resource"]
            unit_type = row.get("cluster", "")
            fuel = row.get("Fuel", "")

            # Extract plant name prefix (first part before underscore+number)
            plant_prefix = "_".join(resource.split("_")[:-1])

            # Find donor: prioritize same plant, then same unit type, then same fuel
            valid = df[~df[col].isna()]

            donor = None
            # 1. Same plant prefix
            same_plant = valid[valid["Resource"].str.startswith(plant_prefix)]
            if not same_plant.empty:
                donor = same_plant.iloc[0]
                reason = f"same plant ({plant_prefix})"
            # 2. Same unit type
            elif unit_type and not valid[valid["cluster"] == unit_type].empty:
                donor = valid[valid["cluster"] == unit_type].iloc[0]
                reason = f"same unit type ({unit_type})"
            # 3. Same fuel
            elif fuel and not valid[valid["Fuel"] == fuel].empty:
                donor = valid[valid["Fuel"] == fuel].iloc[0]
                reason = f"same fuel ({fuel})"
            # 4. Any valid generator
            elif not valid.empty:
                donor = valid.iloc[0]
                reason = "closest available generator"

            if donor is not None:
                df.at[idx, col] = donor[col]
                print(f"    {resource}: imputed {col}={donor[col]:.4f} "
                      f"from {donor['Resource']} [{reason}]")
            else:
                print(f"    {resource}: no donor found for {col} -- left as NaN")

    if not issues_found:
        print("  All required thermal columns present and complete.")

    return df

def validate_fuel_names(thermal_df: pd.DataFrame, vre_df: pd.DataFrame,
                        fuels_df: pd.DataFrame) -> None:
    """
    Validate that all fuel names in Thermal.csv and Vre.csv
    match column names in Fuels_data.csv. Warns on mismatches.
    """
    fuel_cols = set(fuels_df.columns)
    thermal_fuels = set(thermal_df["Fuel"].unique())
    vre_fuels = set(vre_df["Fuel"].unique()) if len(vre_df) > 0 else set()
    all_fuels = thermal_fuels | vre_fuels

    mismatches = all_fuels - fuel_cols
    if mismatches:
        print(f"  WARNING: Fuel name mismatches (not in Fuels_data.csv): {mismatches}")
        print(f"  Available fuel columns: {fuel_cols}")
        print("  GenX will treat these generators as having 0 fuel cost!")
    else:
        print("  All fuel names validated OK.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # FLAGS — change these to generate different cases
    # -------------------------------------------------------------------------
    DISPATCH_MODE = "uc"        # "ed" = economic dispatch, "uc" = unit commitment
    NETWORK_MODE  = "1bus"      # "1bus" = single zone,    "7kbus" = full network
    N_HOURS       = 8760        # Number of hours in study period

    # NREL API credentials
    NREL_API_KEY  = "HTw831IgqjLbaQ2666LjseHpulciK7Fcd0f5WRdt"
    NREL_EMAIL    = "your_email@example.com"  # replace with your email
    NREL_YEAR     = 2024        # resource data year to fetch from NREL
    USE_NREL      = True        # set False to skip NREL fetching and use CF=0
    OVERWRITE     = True       # set True to overwrite existing output files

    # --- File generation toggles (set False to skip) ---
    BUILD_THERMAL      = True
    BUILD_VRE          = True
    BUILD_STORAGE      = True
    BUILD_FUELS        = True
    BUILD_VARIABILITY  = True
    BUILD_DEMAND       = True

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    print(f"\n=== Preparing GenX inputs: {DISPATCH_MODE.upper()} | {NETWORK_MODE} ===\n")

    def write_csv(df, path, index=False, label=""):
        """Write CSV, skipping if file exists and OVERWRITE is False."""
        if path.exists() and not OVERWRITE:
            print(f"  SKIP (exists): {path.name}")
            return
        df.to_csv(path, index=index)
        suffix = f" ({label})" if label else ""
        print(f"  Written: {path.name}{suffix}")

    paths = get_paths(DISPATCH_MODE, NETWORK_MODE)

    # Load inputs (only what's needed for enabled build steps)
    import sys
    _current_module = sys.modules[__name__]

    print("Loading input files...")
    gen_df = cost_df = hourly_df = eia860_df = None

    if BUILD_THERMAL or BUILD_VRE or BUILD_VARIABILITY:
        gen_df = load_gen_tech(paths["gen_tech_file"])
        print(f"  Generators loaded:       {len(gen_df)}")

    if BUILD_THERMAL:
        cost_df = load_cost_curves(paths["cost_curves"])
        print(f"  Cost curve rows loaded:  {len(cost_df)}")

    if BUILD_VRE or BUILD_VARIABILITY or BUILD_DEMAND:
        hourly_df = load_hourly_production(paths["hourly_production"])
        print(f"  Hourly OPF rows loaded:  {len(hourly_df)}")

    if BUILD_VARIABILITY:
        eia860_df = load_eia860_plants(paths["eia860_plants"])

    if BUILD_THERMAL:
        _current_module._F923_HEAT_RATES = load_f923_heat_rates(paths["f923_generation"])
        _current_module._SNL_DATA = load_snl_data(paths["snl_data"])

    if BUILD_VRE and not BUILD_THERMAL:
        # VRE uses _get_fixed_om which needs SNL data
        _current_module._SNL_DATA = load_snl_data(paths["snl_data"])

    # Build and write Thermal.csv
    thermal_df = None
    if BUILD_THERMAL:
        print("\nBuilding Thermal.csv...")
        thermal_df = build_thermal(gen_df, cost_df, NETWORK_MODE, DISPATCH_MODE)
        print("  Validating and imputing missing values...")
        thermal_df = impute_thermal_missing(thermal_df)
        write_csv(thermal_df, paths["out_thermal"])
        print(f"  Thermal: {len(thermal_df)} generators")
    else:
        print("\nSkipping Thermal.csv")

    # Build and write Vre.csv
    vre_df = None
    if BUILD_VRE:
        print("\nBuilding Vre.csv...")
        vre_df = build_vre(gen_df, hourly_df, NETWORK_MODE)
        write_csv(vre_df, paths["out_vre"])
        print(f"  VRE: {len(vre_df)} generators")
    else:
        print("\nSkipping Vre.csv")

    # Build and write Storage.csv (empty)
    if BUILD_STORAGE:
        print("\nBuilding Storage.csv (empty)...")
        storage_df = build_storage()
        write_csv(storage_df, paths["out_storage"])
        print(f"  Storage: empty template")
    else:
        print("\nSkipping Storage.csv")

    # Build and write Fuels_data.csv
    if BUILD_FUELS:
        print("\nBuilding Fuels_data.csv...")
        fuels_df = build_fuels(N_HOURS)
        write_csv(fuels_df, paths["out_fuels"], index=True)
        if thermal_df is not None and vre_df is not None:
            print("  Validating fuel name consistency...")
            validate_fuel_names(thermal_df, vre_df, fuels_df)
    else:
        print("\nSkipping Fuels_data.csv")

    # Fetch NREL profiles and build Generators_variability.csv
    if BUILD_VARIABILITY:
        nrel_profiles = {}
        cache_path = paths["nrel_cache"]
        if cache_path.exists():
            print(f"\nLoading cached NREL profiles from {cache_path}...")
            cache_df = pd.read_csv(cache_path, index_col=0)
            nrel_profiles = {col: cache_df[col].values for col in cache_df.columns}
            print(f"  Loaded {len(nrel_profiles)} cached profiles.")
        elif USE_NREL:
            print("\nFetching NREL VRE profiles (this may take a few minutes)...")
            nrel_profiles = fetch_nrel_profiles(
                gen_df, eia860_df,
                api_key=NREL_API_KEY,
                email=NREL_EMAIL,
                year=NREL_YEAR,
            )
            # Save to cache so we don't need to re-fetch next run
            cache_df = pd.DataFrame(nrel_profiles)
            cache_df.to_csv(cache_path)
            print(f"  Cached {len(nrel_profiles)} profiles to {cache_path}")

        print("\nBuilding Generators_variability.csv...")
        var_df = build_variability(gen_df, hourly_df, eia860_df, nrel_profiles)
        write_csv(var_df, paths["out_variability"], index=True)
        print(f"  Variability: {len(var_df.columns)} VRE resources")
    else:
        print("\nSkipping Generators_variability.csv")

    # Build and write Demand_data.csv
    if BUILD_DEMAND:
        print("\nBuilding Demand_data.csv...")
        demand_df = build_demand(hourly_df, NETWORK_MODE)
        if demand_df is not None:
            write_csv(demand_df, paths["out_demand"], index=False)
    else:
        print("\nSkipping Demand_data.csv")

    # Network.csv — handled by separate script
    # print("\nBuilding Network.csv...")

    print(f"\n=== Done. Outputs in: {paths['out_thermal'].parents[1]} ===\n")