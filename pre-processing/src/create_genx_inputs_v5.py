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
        # 7kbus pre-built input files (in out/genx_inputs/)
        "demand_hourly_7k":     base / "out" / "genx_inputs" / "Demand_data_hourly.csv",
        "network_hourly_7k":    base / "out" / "genx_inputs" / "Network_data_hourly.csv",
        # 7kbus bus number remapping (saved alongside outputs)
        "bus_map":              out_dir / "bus_map.json",
        # Cache
        "nrel_cache":           out_dir / "nrel_profiles_cache.csv",
    }


# =============================================================================
# CONSTANTS
# =============================================================================

FUEL_MAP = {
    "SUB":  "Coal",
    "BIT":  "Coal",
    "NG":   "NG",
    "NUC":  "Nuclear",
    "WAT":  "Hydro",
    "WND":  "Wind",
    "SUN":  "Solar",
    "PC":   "PC",
    "OTH":  "NG",
}

FUEL_PRICES = {
    "NG":      2.39,
    "Coal":    1.92,
    "PC":      1.50,
    "Nuclear": 0.71,
    "Hydro":   0.00,
    "Wind":    0.00,
    "Solar":   0.00,
}

CO2_RATES = {
    "NG":      0.05306,
    "Coal":    0.09535,
    "PC":      0.10241,
    "Nuclear": 0.00000,
    "Hydro":   0.00000,
    "Wind":    0.00000,
    "Solar":   0.00000,
}

VRE_FUELS = {"WND", "SUN"}

NREL_FIXED_OM = {
    "ST":   28000,
    "GT":    7000,
    "CA":   15800,
    "CT":   15800,
    "IC":   10000,
    "NUC": 130000,
    "SUB":  40500,
    "BIT":  40500,
    "NG":   15800,
    "WAT":  44000,
    "PC":   28000,
    "OTH":  15800,
}

THERMAL_FUELS = {"SUB", "BIT", "NG", "NUC", "WAT", "PC", "OTH"}

EIA_HEAT_RATES = {
    "ST":   10.41,
    "GT":   11.31,
    "CA":    7.65,
    "CT":    7.65,
    "IC":   10.00,
    "NUC":  10.46,
    "SUB":  10.41,
    "BIT":  10.41,
    "NG":   10.00,
    "WAT":   0.00,
    "PC":   11.50,
    "OTH":  10.00,
}

EIA_VAR_OM = {
    "ST":   4.00,
    "GT":   5.00,
    "CA":   4.50,
    "CT":   4.50,
    "IC":   5.00,
    "NUC":  2.00,
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
    df = pd.read_excel(path, header=0)
    df = df[df["BusNum"].apply(lambda x: str(x).isnumeric())].copy()
    df["BusNum"] = df["BusNum"].astype(int)
    df["GenMWMax"] = df["GenMWMax"].astype(float)
    return df


def load_cost_curves(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=0)
    df = df[df["BusNum"].apply(lambda x: str(x).isnumeric())].copy()
    df["BusNum"] = df["BusNum"].astype(int)
    df["GenMWMax"] = df["GenMWMax"].astype(float)
    return df


def load_hourly_production(path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(path, header=None, low_memory=False, nrows=5)
    header_row = None
    for i, row in df_raw.iterrows():
        if "Date" in row.values or "Total MW Load" in row.values:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find header row in OPF file.")
    print(f"  OPF header found at row index: {header_row}")
    df = pd.read_csv(path, header=header_row, low_memory=False)
    df = df[pd.to_numeric(df["Total MW Load"], errors="coerce").notna()].copy()
    df["Total MW Load"] = pd.to_numeric(df["Total MW Load"])
    df = df.reset_index(drop=True)
    return df


# =============================================================================
# HELPERS
# =============================================================================

def make_resource_name(row: pd.Series) -> str:
    plant = str(row["EIA-860 Plant Name"]).strip().replace(" ", "_")
    gen_id = str(row["EIA Generator ID"]).strip()
    return f"{plant}_{gen_id}"


def derive_min_power(gen_row: pd.Series, cost_df: pd.DataFrame) -> float:
    defaults = {
        "ST":  0.40,
        "GT":  0.10,
        "CA":  0.30,
        "CT":  0.30,
        "NUC": 0.95,
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
    unit_type = str(gen_row.get("Unit Type Code", "")).strip().upper()
    return defaults.get(unit_type, 0.20)


def get_startup_costs(gen_row: pd.Series, cost_df: pd.DataFrame) -> dict:
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

_F923_HEAT_RATES: pd.DataFrame = None
_SNL_DATA: pd.DataFrame = None


def load_snl_data(path: Path) -> pd.DataFrame:
    print("  Loading S&P Global SNL dataset...")
    df = pd.read_csv(path, dtype={"EIA_SITE_CODE": str}, low_memory=False, thousands=",")
    df.columns = [c.strip() for c in df.columns]
    df["EIA_SITE_CODE"] = pd.to_numeric(df["EIA_SITE_CODE"], errors="coerce")
    df = df.dropna(subset=["EIA_SITE_CODE"])
    df["EIA_SITE_CODE"] = df["EIA_SITE_CODE"].astype(int)

    numeric_cols = [
        "NON_FUEL_NON_ALLOW_VARIABLE_O_AND_M_COST_PER_MWH_WHOLE_GSC",
        "FIXED_O_AND_M_COST_PER_KW_YR_WHOLE_GSC",
        "FUEL_COST_PER_MWH_WHOLE_GSC",
        "CO2_EMISSIONS_RATES",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "FIXED_O_AND_M_COST_PER_KW_YR_WHOLE_GSC" in df.columns:
        df["fixed_om_per_mw_yr"] = df["FIXED_O_AND_M_COST_PER_KW_YR_WHOLE_GSC"] * 1000

    if "TECH_TYPE" in df.columns:
        df["unit_type"] = df["TECH_TYPE"].str.strip().str.upper()

    agg_cols = {
        "NON_FUEL_NON_ALLOW_VARIABLE_O_AND_M_COST_PER_MWH_WHOLE_GSC": "mean",
        "fixed_om_per_mw_yr": "mean",
        "FUEL_COST_PER_MWH_WHOLE_GSC": "mean",
        "CO2_EMISSIONS_RATES": "mean",
        "unit_type": "first",
    }
    valid_agg = {k: v for k, v in agg_cols.items() if k in df.columns}
    snl = df.groupby("EIA_SITE_CODE").agg(valid_agg) if valid_agg else df.set_index("EIA_SITE_CODE")
    print(f"  Loaded {len(snl)} plant records from SNL")
    return snl


def get_snl_value(plant_code: int, col: str, default=None):
    if _SNL_DATA is None or plant_code not in _SNL_DATA.index:
        return default
    val = _SNL_DATA.loc[plant_code, col] if col in _SNL_DATA.columns else default
    if pd.isna(val):
        return default
    return float(val)


def _get_heat_rate(row: pd.Series) -> float:
    hr = float(row.get("Heat Rate", 0) or 0)
    if hr > 0:
        return hr
    plant_code = row.get("EIA-860 Plant Code")
    if _F923_HEAT_RATES is not None and pd.notna(plant_code):
        f923_hr = get_f923_heat_rate(int(plant_code), _F923_HEAT_RATES)
        if f923_hr is not None:
            return f923_hr
    unit_type = str(row.get("Unit Type Code", "")).strip().upper()
    if unit_type in EIA_HEAT_RATES:
        return EIA_HEAT_RATES[unit_type]
    fuel_code = str(row.get("Fuel Type Code", "")).strip().upper()
    return EIA_HEAT_RATES.get(fuel_code, 10.0)


def _get_var_om(row: pd.Series) -> float:
    vom = float(row.get("Variable O&M ($/MWh)", 0) or 0)
    if vom > 0:
        return vom
    plant_code = row.get("EIA-860 Plant Code")
    if _SNL_DATA is not None and pd.notna(plant_code):
        snl_vom = get_snl_value(
            int(plant_code),
            "NON_FUEL_NON_ALLOW_VARIABLE_O_AND_M_COST_PER_MWH_WHOLE_GSC"
        )
        if snl_vom and snl_vom > 0:
            return snl_vom
    unit_type = str(row.get("Unit Type Code", "")).strip().upper()
    if unit_type in EIA_VAR_OM:
        return EIA_VAR_OM[unit_type]
    fuel_code = str(row.get("Fuel Type Code", "")).strip().upper()
    return EIA_VAR_OM.get(fuel_code, 4.0)


def _get_fixed_om(row: pd.Series) -> float:
    plant_code = row.get("EIA-860 Plant Code")
    if _SNL_DATA is not None and pd.notna(plant_code):
        snl_fom = get_snl_value(int(plant_code), "fixed_om_per_mw_yr")
        if snl_fom and snl_fom > 0:
            return snl_fom
    unit_type = str(row.get("Unit Type Code", "")).strip().upper()
    if _SNL_DATA is not None and "unit_type" in _SNL_DATA.columns:
        same_type = _SNL_DATA[
            (_SNL_DATA["unit_type"] == unit_type) &
            (_SNL_DATA["fixed_om_per_mw_yr"] > 0)
        ]["fixed_om_per_mw_yr"]
        if not same_type.empty:
            return float(same_type.mean())
    if unit_type in NREL_FIXED_OM:
        return NREL_FIXED_OM[unit_type]
    fuel_code = str(row.get("Fuel Type Code", "")).strip().upper()
    return float(NREL_FIXED_OM.get(fuel_code, 15800))


def build_thermal(
    gen_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    network_mode: str,
    dispatch_mode: str,
    bus_map: dict = None,
) -> pd.DataFrame:
    model_flag = 1 if dispatch_mode == "ed" else 2
    thermal_df = gen_df[gen_df["Fuel Type Code"].isin(THERMAL_FUELS)].copy()
    records = []
    for _, row in thermal_df.iterrows():
        cap = float(row["GenMWMax"])
        ramp_rate = float(row.get("Ramprate (MW/hr)", cap))
        ramp_pct = min(ramp_rate / cap, 1.0) if cap > 0 else 1.0
        fuel_code = str(row.get("Fuel Type Code", "NG")).strip()
        fuel_name = FUEL_MAP.get(fuel_code, "NG")
        startup = get_startup_costs(row, cost_df)
        min_power = derive_min_power(row, cost_df)
        if network_mode == "1bus":
            zone = 1
        else:
            actual_bus = int(row["BusNum"])
            reverse_map = {v: k for k, v in (bus_map or {}).items()}
            zone = reverse_map.get(actual_bus, actual_bus)
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
    bus_map: dict = None,
) -> pd.DataFrame:
    vre_gen_df = gen_df[gen_df["Fuel Type Code"].isin(VRE_FUELS)].copy()
    records = []
    for _, row in vre_gen_df.iterrows():
        cap = float(row["GenMWMax"])
        fuel_code = str(row.get("Fuel Type Code", "")).strip()
        fuel_name = FUEL_MAP.get(fuel_code, "Wind")
        if network_mode == "1bus":
            zone = 1
        else:
            actual_bus = int(row["BusNum"])
            reverse_map = {v: k for k, v in (bus_map or {}).items()}
            zone = reverse_map.get(actual_bus, actual_bus)
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
    print("  Loading EIA Form 923 heat rates...")
    df = pd.read_excel(
        path,
        sheet_name="Page 1 Generation and Fuel Data",
        header=5,
        dtype={"Plant Id": str},
    )
    df.columns = [str(c).strip().replace("\n", " ").replace("  ", " ") for c in df.columns]

    elec_mmbtu_col = "Elec Fuel Consumption MMBtu"
    netgen_col     = "Net Generation (Megawatthours)"
    plant_col      = "Plant Id"
    fuel_col       = "Reported Fuel Type Code"
    mover_col      = "Reported Prime Mover"

    df = df[df[plant_col].notna()].copy()
    df[plant_col] = pd.to_numeric(df[plant_col], errors="coerce")
    df = df.dropna(subset=[plant_col])
    df[plant_col] = df[plant_col].astype(int)

    for col in [elec_mmbtu_col, netgen_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    agg = df.groupby(plant_col).agg(
        elec_mmbtu  =(elec_mmbtu_col, "sum"),
        net_gen_mwh =(netgen_col,     "sum"),
        fuel_type   =(fuel_col,        "first"),
        prime_mover =(mover_col,       "first"),
    ).reset_index()

    agg = agg[agg["net_gen_mwh"] > 0].copy()
    agg["heat_rate_mmbtu_per_mwh"] = agg["elec_mmbtu"] / agg["net_gen_mwh"]

    n_before = len(agg)
    agg = agg[(agg["heat_rate_mmbtu_per_mwh"] >= 3) & (agg["heat_rate_mmbtu_per_mwh"] <= 30)]
    if n_before > len(agg):
        print(f"  Filtered {n_before - len(agg)} plants with implausible heat rates")

    agg = agg.set_index(plant_col)
    print(f"  Loaded {len(agg)} plant-level heat rates from EIA 923")
    return agg


def get_f923_heat_rate(plant_code: int, f923_df: pd.DataFrame, fallback: float = None) -> float:
    if plant_code in f923_df.index:
        return float(f923_df.loc[plant_code, "heat_rate_mmbtu_per_mwh"])
    return fallback


def get_plant_coords(eia_plant_code: int, eia860_df: pd.DataFrame) -> tuple:
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
    vre_gen_df = gen_df[gen_df["Fuel Type Code"].isin(VRE_FUELS)].copy()

    variability = {}
    fuel_type_map = {}
    matched_by_fuel = {}

    for _, row in vre_gen_df.iterrows():
        bus_num = int(row["BusNum"])
        gen_id = int(row["GenID"])
        cap = float(row["GenMWMax"])
        resource_name = make_resource_name(row)
        fuel_code = str(row.get("Fuel Type Code", "")).strip()
        fuel_type_map[resource_name] = fuel_code

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
            variability[resource_name] = None

    n_hours = len(hourly_df)
    nrel_profiles = nrel_profiles or {}
    from_nrel = []
    from_donor = []
    still_zero = []

    for resource_name, cf in variability.items():
        if cf is None:
            fuel_code = fuel_type_map[resource_name]

            if resource_name in nrel_profiles:
                variability[resource_name] = nrel_profiles[resource_name]
                from_nrel.append(resource_name)
            elif matched_by_fuel.get(fuel_code):
                variability[resource_name] = matched_by_fuel[fuel_code][0]
                from_donor.append(resource_name)
            elif any(fuel_type_map.get(k) == fuel_code for k in nrel_profiles):
                donor_cf = next(
                    v for k, v in nrel_profiles.items()
                    if fuel_type_map.get(k) == fuel_code
                )
                variability[resource_name] = donor_cf
                from_donor.append(resource_name)
            else:
                variability[resource_name] = np.zeros(n_hours)
                still_zero.append(resource_name)

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
    fuels = list(FUEL_PRICES.keys())
    rows = []
    co2_row = {"Time_Index": 0}
    for fuel in fuels:
        co2_row[fuel] = CO2_RATES.get(fuel, 0.0)
    rows.append(co2_row)
    for t in range(1, n_hours + 1):
        price_row = {"Time_Index": t}
        for fuel in fuels:
            price_row[fuel] = FUEL_PRICES.get(fuel, 0.0)
        rows.append(price_row)
    return pd.DataFrame(rows).set_index("Time_Index")


# =============================================================================
# DEMAND_DATA.CSV
# =============================================================================

def build_bus_map(gen_df: pd.DataFrame,
                  demand_path: Path = None,
                  network_path: Path = None) -> dict:
    """
    Build a contiguous 1-indexed zone map from all unique bus numbers
    found across generator data, demand file, and network file.

    GenX requires zone numbers to be contiguous integers starting at 1.

    Returns:
        dict mapping genx_zone (int) -> actual_bus_number (int)
        e.g. {1: 101, 2: 205, 3: 412, ...}
    """
    bus_numbers = set()

    # From generator data
    if gen_df is not None:
        buses = pd.to_numeric(gen_df["BusNum"], errors="coerce").dropna().astype(int)
        bus_numbers.update(buses.tolist())

    # From demand file columns (Load_MW_z{bus})
    if demand_path is not None and demand_path.exists():
        df = pd.read_csv(demand_path, nrows=1)
        for col in df.columns:
            if col.startswith("Load_MW_z"):
                try:
                    bus_numbers.add(int(col.replace("Load_MW_z", "")))
                except ValueError:
                    pass

    # From network file zone columns
    if network_path is not None and network_path.exists():
        df = pd.read_csv(network_path, nrows=1)
        for col in df.columns:
            if col.startswith("z"):
                try:
                    bus_numbers.add(int(col[1:]))
                except ValueError:
                    pass

    sorted_buses = sorted(bus_numbers)
    bus_map = {i + 1: bus for i, bus in enumerate(sorted_buses)}
    print(f"  Bus map: {len(bus_map)} buses -> contiguous zones 1..{len(bus_map)}")
    return bus_map


def apply_bus_map(df: pd.DataFrame, bus_map: dict,
                  col_prefix: str = "Load_MW_z") -> pd.DataFrame:
    """
    Rename columns from actual bus numbers to contiguous GenX zone indices.
    Works for demand columns (Load_MW_z{bus}) and network columns (z{bus}).
    """
    reverse_map = {v: k for k, v in bus_map.items()}
    rename = {}
    for col in df.columns:
        if col.startswith(col_prefix):
            try:
                actual_bus = int(col[len(col_prefix):])
                if actual_bus in reverse_map:
                    rename[col] = f"{col_prefix}{reverse_map[actual_bus]}"
            except ValueError:
                pass
    return df.rename(columns=rename) if rename else df


def build_demand(hourly_df: pd.DataFrame, network_mode: str,
                 demand_hourly_path: Path = None,
                 bus_map: dict = None) -> pd.DataFrame:
    """
    Build Demand_data.csv.
    - 1bus: derives single-zone load from OPF Total MW Load column.
    - 7kbus: reads pre-built Demand_data_hourly.csv which retains the
             bus-to-zone mapping from the network script.
    """
    if network_mode == "7kbus":
        if demand_hourly_path is None or not demand_hourly_path.exists():
            raise FileNotFoundError(
                f"7kbus demand file not found: {demand_hourly_path}\n"
                "Expected: out/genx_inputs/Demand_data_hourly.csv"
            )
        print(f"  Reading 7kbus demand from: {demand_hourly_path.name}")
        df = pd.read_csv(demand_hourly_path)
        if bus_map is not None:
            df = apply_bus_map(df, bus_map, col_prefix="Load_MW_z")
        print(f"  Zones found: {[c for c in df.columns if c.startswith('Load_MW_z')]}")
        return df

    # 1bus: derive from OPF total load
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

    rows = []
    for t in range(n_hours):
        if t == 0:
            rows.append([50000, 1, 1, 1, 2000, 1, n_hours, n_hours, t + 1, load[t]])
        else:
            rows.append(["", "", "", "", "", "", "", "", t + 1, load[t]])

    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# NETWORK.CSV
# =============================================================================

def build_network(network_mode: str, network_hourly_path: Path = None,
                  bus_map: dict = None) -> pd.DataFrame:
    """
    Build Network.csv.
    - 1bus: no network file needed (GenX runs single-zone without it).
    - 7kbus: reads pre-built Network_data_hourly.csv which retains the
             bus naming scheme from the network script.
    """
    if network_mode == "1bus":
        return None  # GenX single-zone does not require Network.csv

    if network_hourly_path is None or not network_hourly_path.exists():
        raise FileNotFoundError(
            f"7kbus network file not found: {network_hourly_path}\n"
            "Expected: out/genx_inputs/Network_data_hourly.csv"
        )
    print(f"  Reading 7kbus network from: {network_hourly_path.name}")
    df = pd.read_csv(network_hourly_path)
    if bus_map is not None:
        df = apply_bus_map(df, bus_map, col_prefix="z")
        # Remap Network_zones column if present
        if "Network_zones" in df.columns:
            reverse_map = {v: k for k, v in bus_map.items()}
            def remap_zone_label(x):
                s = str(x).lstrip("z")
                return f"z{reverse_map[int(s)]}" if s.isdigit() and int(s) in reverse_map else x
            df["Network_zones"] = df["Network_zones"].apply(remap_zone_label)
    return df


# =============================================================================
# NREL API — SOLAR (NSRDB) AND WIND CAPACITY FACTORS
# =============================================================================

NSRDB_URL      = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.csv"
NSRDB_GOES_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"
WIND_URL       = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-download.csv"
BCHRRR_URL     = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/bc-hrrr-download.csv"

SOLAR_TILT       = 25
SOLAR_AZIMUTH    = 180
SOLAR_EFFICIENCY = 0.18
SOLAR_ILF        = 1.1


def wind_speed_to_cf(ws: float) -> float:
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

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), skiprows=2)
    df.index = pd.date_range(f"1/1/{year}", periods=len(df), freq="h", tz="UTC")

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

    temp_coeff = 0.0045
    dc_power = poa["poa_global"] * SOLAR_EFFICIENCY * (1 - temp_coeff * (cell_temp - 25))
    dc_power = dc_power.clip(lower=0)
    rated_dc = 1000 * SOLAR_EFFICIENCY
    cf = (dc_power / rated_dc / SOLAR_ILF).clip(0, 1).values[:8760]
    return cf


WIND_MAX_YEAR = 2014


def fetch_wind_cf(lat: float, lon: float, year: int,
                  api_key: str, email: str,
                  hub_height: int = 100) -> np.ndarray:
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
            else:
                cf = fetch_wind_cf(lat, lon, year, api_key, email)
            profiles[resource_name] = cf
            print(f"OK (mean CF={cf.mean():.3f})")
        except Exception as e:
            print(f"FAILED: {e}")
            profiles[resource_name] = np.zeros(8760)

        time.sleep(rate_limit_sleep)

    return profiles


# =============================================================================
# THERMAL DATA VALIDATION & IMPUTATION
# =============================================================================

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
    df = thermal_df.copy()
    issues_found = False

    for col in THERMAL_REQUIRED_COLS:
        if col not in df.columns:
            continue
        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue

        issues_found = True
        print(f"  WARNING: Missing values in '{col}' for {missing_mask.sum()} generators:")

        for idx in df[missing_mask].index:
            row = df.loc[idx]
            resource = row["Resource"]
            unit_type = row.get("cluster", "")
            fuel = row.get("Fuel", "")
            plant_prefix = "_".join(resource.split("_")[:-1])
            valid = df[~df[col].isna()]

            donor = None
            same_plant = valid[valid["Resource"].str.startswith(plant_prefix)]
            if not same_plant.empty:
                donor = same_plant.iloc[0]
                reason = f"same plant ({plant_prefix})"
            elif unit_type and not valid[valid["cluster"] == unit_type].empty:
                donor = valid[valid["cluster"] == unit_type].iloc[0]
                reason = f"same unit type ({unit_type})"
            elif fuel and not valid[valid["Fuel"] == fuel].empty:
                donor = valid[valid["Fuel"] == fuel].iloc[0]
                reason = f"same fuel ({fuel})"
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
    fuel_cols = set(fuels_df.columns)
    thermal_fuels = set(thermal_df["Fuel"].unique())
    vre_fuels = set(vre_df["Fuel"].unique()) if len(vre_df) > 0 else set()
    mismatches = (thermal_fuels | vre_fuels) - fuel_cols
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
    DISPATCH_MODE = "ed"        # "ed" = economic dispatch, "uc" = unit commitment
    NETWORK_MODE  = "7kbus"      # "1bus" = single zone,    "7kbus" = full network
    N_HOURS       = 8760

    NREL_API_KEY  = "HTw831IgqjLbaQ2666LjseHpulciK7Fcd0f5WRdt"
    NREL_EMAIL    = "onurtalu@mit.edu"
    NREL_YEAR     = 2024
    USE_NREL      = True
    OVERWRITE     = True

    # --- File generation toggles (set False to skip) ---
    BUILD_THERMAL      = True
    BUILD_VRE          = True
    BUILD_STORAGE      = True
    BUILD_FUELS        = True
    BUILD_VARIABILITY  = True
    BUILD_DEMAND       = True
    BUILD_NETWORK      = True

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

    import sys
    _current_module = sys.modules[__name__]

    print("Loading input files...")
    gen_df = cost_df = hourly_df = eia860_df = None

    if BUILD_THERMAL or BUILD_VRE or BUILD_VARIABILITY:
        gen_df = load_gen_tech(paths["gen_tech_file"])
        print(f"  Generators loaded:       {len(gen_df)}")

    # Build bus map for 7kbus mode (must happen before any build step)
    bus_map = None
    if NETWORK_MODE == "7kbus":
        print("\nBuilding bus number map (actual -> contiguous GenX zones)...")
        bus_map = build_bus_map(
            gen_df,
            demand_path=paths["demand_hourly_7k"],
            network_path=paths["network_hourly_7k"],
        )
        import json
        with open(paths["bus_map"], "w") as f:
            json.dump({str(k): v for k, v in bus_map.items()}, f, indent=2)
        print(f"  Bus map saved: GenX z1={bus_map[1]}, z2={bus_map[2]}, ...")
        print(f"  Full map written to: {paths['bus_map'].name}")

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
        _current_module._SNL_DATA = load_snl_data(paths["snl_data"])

    # Build and write Thermal.csv
    thermal_df = None
    if BUILD_THERMAL:
        print("\nBuilding Thermal.csv...")
        thermal_df = build_thermal(gen_df, cost_df, NETWORK_MODE, DISPATCH_MODE, bus_map=bus_map)
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
        vre_df = build_vre(gen_df, hourly_df, NETWORK_MODE, bus_map=bus_map)
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
        # Priority: (1) input folder cache, (2) output folder cache, (3) fetch from NREL
        input_cache_path = Path(__file__).parents[1] / "in" / "nrel_profiles_cache.csv"

        if input_cache_path.exists():
            print(f"\nLoading NREL profiles from input cache: {input_cache_path}...")
            cache_df = pd.read_csv(input_cache_path, index_col=0)
            nrel_profiles = {col: cache_df[col].values for col in cache_df.columns}
            print(f"  Loaded {len(nrel_profiles)} cached profiles.")
        elif cache_path.exists():
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
        demand_df = build_demand(
            hourly_df, NETWORK_MODE,
            demand_hourly_path=paths["demand_hourly_7k"],
            bus_map=bus_map,
        )
        if demand_df is not None:
            write_csv(demand_df, paths["out_demand"], index=False)
    else:
        print("\nSkipping Demand_data.csv")

    # Build and write Network.csv
    if BUILD_NETWORK:
        print("\nBuilding Network.csv...")
        network_df = build_network(
            NETWORK_MODE,
            network_hourly_path=paths["network_hourly_7k"],
            bus_map=bus_map,
        )
        if network_df is not None:
            write_csv(network_df, paths["out_network"])
        else:
            print("  Single-bus mode: no Network.csv needed.")
    else:
        print("\nSkipping Network.csv")

    print(f"\n=== Done. Outputs in: {paths['out_thermal'].parents[1]} ===\n")