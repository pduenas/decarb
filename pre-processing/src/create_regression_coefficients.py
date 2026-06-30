import time
import re
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.sparse import block_diag as sp_block_diag
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import spsolve
import json
from scipy.optimize import lsq_linear

# ============================================================
# Function 1: Create building timeseries
# ============================================================
def create_building_timeseries(
    zone, input_dir, output_dir, ts_config,
    save_outputs=False,
    skip_missing_weather=False
):
    """
    Builds per-building timeseries and returns a dict of DataFrames.
    """

    # Directories
    bldg_consumption_dir = output_dir / "consumption_files"
    bldg_id_dir = output_dir / zone / "bldg_ids"
    weather_dir = input_dir / "TX_weather_data"
    out_dir = output_dir / zone / "input_timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ts_dict = {}

    # ----------------------------
    # Load building IDs
    # ----------------------------
    bldg_id_file = bldg_id_dir / f"{zone}_bldg_ids_all.csv"
    if not bldg_id_file.exists():
        raise FileNotFoundError(f"Missing building ID file: {bldg_id_file}")

    bldg_df = pd.read_csv(bldg_id_file)

    # Force building ID column
    if "bldg_id" not in bldg_df.columns:
        if "nrel_climate_match" in bldg_df.columns:
            bldg_df = bldg_df.rename(columns={"nrel_climate_match": "bldg_id"})
        else:
            raise KeyError("No usable building ID column found.")

    # Safe integer IDs
    bldg_df["bldg_id"] = pd.to_numeric(bldg_df["bldg_id"], errors="coerce").astype("Int64")
    print(f"Loaded {len(bldg_df)} buildings from {bldg_id_file}")

    # ----------------------------
    # Load ResStock metadata for square footage
    # ----------------------------
    resstock_file = input_dir / "TX_upgrade0.parquet"
    if not resstock_file.exists():
        raise FileNotFoundError(f"Missing ResStock parquet file: {resstock_file}")

    resstock_df = pd.read_parquet(resstock_file, engine="fastparquet")
    
    # Keep only needed columns
    if "in.sqft..ft2" not in resstock_df.columns:
        raise KeyError("Missing 'in.sqft..ft2' column in ResStock data")
    
    resstock_df = resstock_df[["bldg_id", "in.sqft..ft2", "in.county"]].copy()
    resstock_df = resstock_df.rename(columns={
        "in.sqft..ft2": "sqft",
        "in.county": "county_code"
    })
    
    resstock_df["bldg_id"] = pd.to_numeric(
        resstock_df["bldg_id"], errors="coerce"
    ).astype("Int64")
    
    # Convert square feet to square meters (1 ft² = 0.092903 m²)
    resstock_df["sqm"] = resstock_df["sqft"] * 0.092903

    # ----------------------------
    # Merge building list with ResStock metadata
    # ----------------------------
    merged_bldg_df = pd.merge(
        bldg_df,
        resstock_df,
        on="bldg_id",
        how="left",
        validate="m:1"
    )

    # ----------------------------
    # Building loop
    # ----------------------------
    for _, row in merged_bldg_df.iterrows():

        bldg_id = row["bldg_id"]
        county_code = row["county_code"]
        sqm = row["sqm"]

        if pd.isna(bldg_id):
            continue
        
        if pd.isna(sqm):
            print(f"⚠ Missing square footage for building {bldg_id}")
            continue

        bldg_path = bldg_consumption_dir / f"{bldg_id}-0.parquet"
        if not bldg_path.exists():
            print(f"⚠ Missing consumption file for {bldg_id}")
            continue

        # Load building parquet
        try:
            df_bldg = pd.read_parquet(bldg_path, engine="fastparquet")
            if "timestamp" in df_bldg.columns:
                df_bldg["datetime"] = pd.to_datetime(df_bldg["timestamp"])
            else:
                print(f"❌ {bldg_path.name} missing 'timestamp'")
                continue

            df_cols = df_bldg.columns
            if ts_config["indoor_temp_key"] not in df_cols:
                print(f"❌ Missing indoor temp for {bldg_id}")
                continue

            # Replace missing load columns with 0
            if (
                ts_config["cooling_load_key"] not in df_cols or
                df_bldg[ts_config["cooling_load_key"]].isna().all()
            ):
                df_bldg[ts_config["cooling_load_key"]] = 0

            if (
                ts_config["heating_load_key"] not in df_cols or
                df_bldg[ts_config["heating_load_key"]].isna().all()
            ):
                df_bldg[ts_config["heating_load_key"]] = 0

            df_bldg = df_bldg[
                [
                    "datetime",
                    ts_config["indoor_temp_key"],
                    ts_config["cooling_load_key"],
                    ts_config["heating_load_key"],
                ]
            ].copy()

            df_bldg.rename(
                columns={
                    ts_config["indoor_temp_key"]: "indoor_temp_c",
                    ts_config["cooling_load_key"]: "cooling_load_kbtu",
                    ts_config["heating_load_key"]: "heating_load_kbtu",
                },
                inplace=True,
            )

            df_bldg["cooling_load_w"] = df_bldg["cooling_load_kbtu"] * 1172.284
            df_bldg["heating_load_w"] = df_bldg["heating_load_kbtu"] * 1172.284

        except Exception as e:
            print(f"❌ Error reading {bldg_id}-0.parquet: {e}")
            continue

        # ----------------------------
        # Weather lookup
        # ----------------------------
        df_weather = None
        if not pd.isna(county_code):
            weather_path = weather_dir / f"{county_code}_2018.csv"
            if weather_path.exists():
                try:
                    df_weather = pd.read_csv(weather_path)
                    
                    # Find outdoor temperature column
                    outdoor_col = next(
                        (
                            c
                            for c in df_weather.columns
                            if ts_config["outdoor_temp_key"]
                            .lower()
                            .replace(" ", "")
                            in c.lower().replace(" ", "")
                        ),
                        None,
                    )
                    
                    # Find solar radiation column
                    dnr_col = next(
                        (
                            c
                            for c in df_weather.columns
                            if ts_config["dnr_radiation_key"]
                            .lower()
                            .replace(" ", "")
                            in c.lower().replace(" ", "")
                        ),
                        None,
                    )
                    
                    # Find humidity column
                    if "outdoor_humidity_key" in ts_config:
                        humidity_col = next(
                            (
                                c
                                for c in df_weather.columns
                                if ts_config["outdoor_humidity_key"]
                                .lower()
                                .replace(" ", "")
                                in c.lower().replace(" ", "")
                            ),
                            None,
                        )
                    else:
                        humidity_col = next(
                            (
                                c
                                for c in df_weather.columns
                                if "humidity" in c.lower() or "rh" in c.lower()
                            ),
                            None,
                        )
                    
                    # Find wind speed column
                    if "wind_speed_key" in ts_config:
                        wind_col = next(
                            (
                                c
                                for c in df_weather.columns
                                if ts_config["wind_speed_key"]
                                .lower()
                                .replace(" ", "")
                                in c.lower().replace(" ", "")
                            ),
                            None,
                        )
                    else:
                        wind_col = next(
                            (
                                c
                                for c in df_weather.columns
                                if "wind" in c.lower() and "speed" in c.lower()
                            ),
                            None,
                        )

                    # Require at minimum outdoor temp and radiation
                    if outdoor_col is None or dnr_col is None:
                        df_weather = None
                    else:
                        df_weather["datetime"] = pd.to_datetime(df_weather["date_time"])
                        
                        rename_dict = {
                            dnr_col: "solar_radiation_wm2",
                            outdoor_col: "outdoor_temp_c",
                        }
                        
                        # Add humidity if available
                        if humidity_col is not None:
                            rename_dict[humidity_col] = "outdoor_humidity"
                        
                        # Add wind speed if available
                        if wind_col is not None:
                            rename_dict[wind_col] = "wind_speed"
                        
                        df_weather = df_weather.rename(columns=rename_dict)
                        
                        # **SCALE RADIATION BY BUILDING AREA**
                        df_weather["solar_radiation_w"] = df_weather["solar_radiation_wm2"] * sqm
                        
                        # Select columns (humidity and wind optional)
                        cols_to_keep = ["datetime", "solar_radiation_w", "outdoor_temp_c"]
                        if "outdoor_humidity" in df_weather.columns:
                            cols_to_keep.append("outdoor_humidity")
                        if "wind_speed" in df_weather.columns:
                            cols_to_keep.append("wind_speed")
                        
                        df_weather = df_weather[cols_to_keep]

                        df_weather.set_index("datetime", inplace=True)
                        df_weather = df_weather.reindex(df_bldg["datetime"]).interpolate(
                            "linear"
                        )
                        df_weather.reset_index(inplace=True)
                except Exception:
                    df_weather = None

        # Attach weather
        if df_weather is not None:
            merged = df_bldg.merge(df_weather, on="datetime", how="left")
        elif skip_missing_weather:
            merged = df_bldg.copy()
        else:
            continue

        all_ts_dict[int(bldg_id)] = merged.copy()

        if save_outputs:
            out_path = out_dir / f"{zone}_{bldg_id}_in_ts.csv"
            merged.to_csv(out_path, index=False)
            print(f"Saved {out_path.name}")

    print("🏁 Done creating input timeseries.")
    return all_ts_dict


# ============================================================
# Function 2: Flexible OLS Regression with Feature Sets
# ============================================================
def compute_regression_coefficients_old(
    zone, 
    all_ts_dict, 
    output_dir,
    feature_set="baseline",
    required_columns=None
):
    """
    Compute OLS regression with configurable feature sets.
    
    Regression form: T_in(t) = T_in(t-1) + k[1] * [T_out(t-1) - T_in(t-1)] + k[2] * Q_rad(t) + k[3] * (q_heating(t-1) - q_cooling(t-1))
    
    Args:
        zone: Zone identifier
        all_ts_dict: Dictionary of building timeseries
        output_dir: Output directory
        feature_set: Name identifier for this feature configuration
        required_columns: Dict mapping feature names to dataframe columns
    """
    results = []
    
    # Default feature set
    if required_columns is None:
        required_columns = {
            "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],  # T_out - T_in
            "radiation": "solar_radiation_w",
            "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        }

    for bldg_id, df in all_ts_dict.items():
        if df.shape[0] < 2:
            continue

        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)

        df_lag = df.shift(1)

        # Build X dataframe dynamically based on required_columns
        X_dict = {}
        skip_building = False
        
        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                # Handle computed features
                if feature_name == "heat_minus_cool_w":
                    # Use lagged heating and cooling
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
                elif feature_name == "temp_diff_lag":
                    # T_out(t-1) - T_in(t-1)
                    if col_spec[0] not in df_lag.columns or col_spec[1] not in df_lag.columns:
                        skip_building = True
                        break
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
            elif "lag" in feature_name:
                # Lagged variable
                col = col_spec
                if col not in df_lag.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df_lag[col]
            else:
                # Current time variable (like radiation)
                col = col_spec
                if col not in df.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df[col]
        
        if skip_building or not X_dict:
            continue
            
        X = pd.DataFrame(X_dict)
        
        # y = T_in(t) - T_in(t-1), so we're modeling the change
        y = df["indoor_temp_c"] - df_lag["indoor_temp_c"]

        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if X.shape[0] < 2:
            continue

        # NO CONSTANT - the model form doesn't include an intercept
        model = sm.OLS(y, X).fit()

        y_pred_delta = model.predict(X)
        mse = mean_squared_error(y, y_pred_delta)

        # Corrected R²: evaluate on T_in(t), not ΔT
        T_in_actual = df.loc[valid_idx, "indoor_temp_c"]
        T_in_lag    = df_lag.loc[valid_idx, "indoor_temp_c"]
        T_in_pred   = T_in_lag + y_pred_delta

        ss_res = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
        ss_tot = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
        r_squared_reconstructed = 1 - ss_res / ss_tot
        mse_reconstructed = mean_squared_error(T_in_actual.values, T_in_pred.values)
        
        # Build result dict dynamically - no k0 since no intercept
        result = {"bldg_id": bldg_id}
        
        # Add all k coefficients (starting from k1)
        for i, feature_name in enumerate(X.columns, start=1):
            result[f"k{i}"] = model.params[feature_name]
        
        # Then add all p-values
        for i, feature_name in enumerate(X.columns, start=1):
            result[f"p_k{i}"] = model.pvalues[feature_name]
        
        # Finally add metrics
        result["mse"] = mse_reconstructed
        result["r_squared"] = r_squared_reconstructed
        
        results.append(result)

    # Save with feature_set in filename
    out_csv = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    
    # Save column mapping for reference
    metadata = {
        "feature_set": feature_set,
        "features": list(required_columns.keys()),
        "model_form": "T_in(t) = T_in(t-1) + k1*[T_out(t-1)-T_in(t-1)] + k2*Q_rad(t) + k3*[Q_heat(t-1)-Q_cool(t-1)]",
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Save metadata alongside results
    metadata_file = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    results_df.to_csv(out_csv, index=False)
    print(f"💾 Saved {feature_set} regression coefficients: {out_csv}")


def compute_regression_coefficients(
    zone, 
    all_ts_dict, 
    output_dir,
    feature_set="baseline",
    required_columns=None
):
    """
    Compute OLS regression with configurable feature sets and positivity constraints.
    
    Regression form: T_in(t) = T_in(t-1) + k[1] * [T_out(t-1) - T_in(t-1)] + k[2] * Q_rad(t) + k[3] * (q_heating(t-1) - q_cooling(t-1))
    
    Constraints: k1, k2, k3 >= 0 (all coefficients must be non-negative)
    
    Args:
        zone: Zone identifier
        all_ts_dict: Dictionary of building timeseries
        output_dir: Output directory
        feature_set: Name identifier for this feature configuration
        required_columns: Dict mapping feature names to dataframe columns
    """
    results = []
    
    # Default feature set
    if required_columns is None:
        required_columns = {
            "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],  # T_out - T_in
            "radiation": "solar_radiation_w",
            "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        }

    for bldg_id, df in all_ts_dict.items():
        if df.shape[0] < 2:
            continue

        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)

        df_lag = df.shift(1)

        # Build X dataframe dynamically based on required_columns
        X_dict = {}
        skip_building = False
        
        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                # Handle computed features
                if feature_name == "heat_minus_cool_w":
                    # Use lagged heating and cooling
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
                elif feature_name == "temp_diff_lag":
                    # T_out(t-1) - T_in(t-1)
                    if col_spec[0] not in df_lag.columns or col_spec[1] not in df_lag.columns:
                        skip_building = True
                        break
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
            elif "lag" in feature_name:
                # Lagged variable
                col = col_spec
                if col not in df_lag.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df_lag[col]
            else:
                # Current time variable (like radiation)
                col = col_spec
                if col not in df.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df[col]
        
        if skip_building or not X_dict:
            continue
            
        X = pd.DataFrame(X_dict)
        
        # y = T_in(t) - T_in(t-1), so we're modeling the change
        y = df["indoor_temp_c"] - df_lag["indoor_temp_c"]

        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if X.shape[0] < 2:
            continue

        # Use constrained least squares with non-negativity constraints
        n_features = X.shape[1]
        lower_bounds = np.zeros(n_features)  # All coefficients >= 0
        upper_bounds = np.inf * np.ones(n_features)  # No upper bound
        
        result_lsq = lsq_linear(X.values, y.values, bounds=(lower_bounds, upper_bounds))
        coefficients = result_lsq.x
        
        # Compute predictions
        y_pred_delta = X.values @ coefficients

        # Corrected R²: evaluate on T_in(t), not ΔT
        T_in_actual = df.loc[valid_idx, "indoor_temp_c"]
        T_in_lag    = df_lag.loc[valid_idx, "indoor_temp_c"]
        T_in_pred   = T_in_lag + y_pred_delta

        ss_res = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
        ss_tot = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
        r_squared_reconstructed = 1 - ss_res / ss_tot
        mse_reconstructed = mean_squared_error(T_in_actual.values, T_in_pred.values)
        
        # Build result dict - save only bldg_id, k1, k2, k3, mse, r_squared
        result = {
            "bldg_id": bldg_id,
            "k1": coefficients[0],  # temp_diff_lag coefficient
            "k2": coefficients[1],  # radiation coefficient
            "k3": coefficients[2],  # heat_minus_cool_w coefficient
            "mse": mse_reconstructed,
            "r_squared": r_squared_reconstructed
        }
        
        results.append(result)

    # Save with feature_set in filename
    out_csv = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    
    # Save column mapping for reference
    metadata = {
        "feature_set": feature_set,
        "features": list(required_columns.keys()),
        "constraints": "All coefficients (k1, k2, k3) constrained to be non-negative (>= 0)",
        "model_form": "T_in(t) = T_in(t-1) + k1*[T_out(t-1)-T_in(t-1)] + k2*Q_rad(t) + k3*[Q_heat(t-1)-Q_cool(t-1)]",
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Save metadata alongside results
    metadata_file = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    results_df.to_csv(out_csv, index=False)
    print(f"💾 Saved {feature_set} regression coefficients: {out_csv}")
    
    
# ============================================================
# Function 3: Regularized Ridge Regression
# ============================================================
def compute_regularized_ridge_coefficients_sparse(
    zone, 
    all_ts_dict, 
    output_dir, 
    lambda_reg=1.0,
    feature_set="baseline",
    required_columns=None
):
    """
    Regularized ridge regression with configurable features.
    
    Regression form: T_in(t) = T_in(t-1) + k[1] * [T_out(t-1) - T_in(t-1)] + k[2] * Q_rad(t) + k[3] * (q_heating(t-1) - q_cooling(t-1))
    """
    
    if required_columns is None:
        required_columns = {
            "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],
            "radiation": "solar_radiation_w",
            "heat_minus_cool": ["heating_load_w", "cooling_load_w"]
        }

    building_ids = list(all_ts_dict.keys())
    X_list, y_list = [], []
    valid_building_ids = []

    # ----------------------------------
    # Build X and y for each building
    # ----------------------------------
    for bldg_id in building_ids:
        df = all_ts_dict[bldg_id].copy()

        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)

        # First check if all required base columns exist
        skip_building = False
        for feature_name, col_spec in required_columns.items():
            if not isinstance(col_spec, list):
                # Check if base column exists
                if col_spec not in df.columns:
                    skip_building = True
                    break
            else:
                # Check if all columns in list exist
                for col in col_spec:
                    if col not in df.columns:
                        skip_building = True
                        break
        
        if skip_building:
            continue

        # Create lagged variables
        for col in df.columns:
            if col != "datetime":
                df[f"{col}_lag"] = df[col].shift(1)
        
        df = df.dropna()

        if df.shape[0] == 0:
            continue
        
        T_in_actual_arr = df["indoor_temp_c"].values
        T_in_lag_arr    = df["indoor_temp_c_lag"].values

        # Build X dynamically - NO INTERCEPT
        X_dict = {}
        
        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                # Computed feature
                if "heat_minus_cool" in feature_name:
                    # Use lagged heating and cooling
                    X_dict[feature_name] = df[f"{col_spec[0]}_lag"] - df[f"{col_spec[1]}_lag"]
                elif "temp_diff" in feature_name:
                    # T_out(t-1) - T_in(t-1)
                    X_dict[feature_name] = df[f"{col_spec[0]}_lag"] - df[f"{col_spec[1]}_lag"]
            else:
                # Direct column reference
                if "lag" in feature_name:
                    col = col_spec + "_lag"
                else:
                    col = col_spec
                X_dict[feature_name] = df[col]
        
        X = pd.DataFrame(X_dict)
        
        # y = T_in(t) - T_in(t-1)
        y = df["indoor_temp_c"] - df["indoor_temp_c_lag"]

        X_list.append(csr_matrix(X.values))
        y_list.append(y.values)
        valid_building_ids.append(bldg_id)

    # ----------------------------------
    # SAFE FEATURE CHECK (prevents index errors)
    # ----------------------------------
    if len(X_list) == 0:
        print(f"⚠ No usable buildings for zone {zone}, feature set {feature_set}. Skipping ridge regression.")
        return

    n_buildings = len(X_list)
    n_features = X_list[0].shape[1]

    # ----------------------------------
    # Block diagonal X
    # ----------------------------------
    X_block = sp_block_diag(X_list, format="csr")
    y_block = np.concatenate(y_list)

    # ----------------------------------
    # Adjacent-difference Laplacian
    # ----------------------------------
    L_rows = []
    for i in range(n_buildings - 1):
        row = np.zeros(n_features * n_buildings)

        # Difference of all coefficients (no intercept to skip)
        row[i*n_features: (i+1)*n_features] = 1
        row[(i+1)*n_features: (i+2)*n_features] = -1

        L_rows.append(csr_matrix(row))

    L = vstack(L_rows, format="csr")

    # ----------------------------------
    # Ridge solve: (XᵀX + λ LᵀL) β = Xᵀy
    # ----------------------------------
    XtX = X_block.T @ X_block
    Xty = X_block.T @ y_block
    LtL = L.T @ L
    A = XtX + lambda_reg * LtL

    beta_reg = spsolve(A, Xty)

    # ----------------------------------
    # Extract coefficients
    # ----------------------------------
    results = []
    offset = 0

    for idx, bldg_id in enumerate(valid_building_ids):
        coeffs = beta_reg[offset: offset + n_features]
        offset += n_features

        X_i = X_list[idx]
        y_i = y_list[idx]
        mse = np.mean((y_i - X_i @ coeffs) ** 2)

        # Corrected R²: evaluate on T_in(t), not ΔT
        # df_i = all_ts_dict[bldg_id]
        T_in_actual = T_in_actual_arr  # already aligned to X_i
        T_in_lag    = T_in_lag_arr

        T_in_pred   = T_in_lag + (X_i @ coeffs)

        ss_res = np.sum((T_in_actual - T_in_pred) ** 2)
        ss_tot = np.sum((T_in_actual - T_in_actual.mean()) ** 2)
        r_squared_reconstructed = 1 - ss_res / ss_tot

        # Build result dict - no k0 since no intercept
        result = {"bldg_id": bldg_id}
        
        # Add all k coefficients (starting from k1)
        for i, feature_name in enumerate(required_columns.keys(), start=1):
            result[f"k{i}"] = coeffs[i-1]
        
        # Add mse and r_squared at the end
        result["mse"] = mse
        result["r_squared"] = r_squared_reconstructed
        # assert len(T_in_actual) == X_i.shape[0], f"Length mismatch for building {bldg_id}"
        
        results.append(result)

    out_csv = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_regularized_{feature_set}_lambda_{lambda_reg}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"💾 Saved regularized {feature_set} coefficients (λ={lambda_reg}): {out_csv}")

# ============================================================
# Function 3.5: Regularized Ridge Regression with Filter Width
# ============================================================
def compute_regularized_ridge_coefficients_sparse_fw(
    zone, 
    all_ts_dict, 
    output_dir, 
    lambda_reg=1.0,
    feature_set="baseline",
    required_columns=None,
    filter_width=1
):
    """
    Regularized ridge regression with configurable features and a filter width.

    Regression form: 
    T_in(t) = T_in(t-1) + k[1]*(T_out(t-1)-T_in(t-1)) + k[2]*Q_rad(t) + k[3]*(q_heating(t-1)-q_cooling(t-1))
    
    filter_width: number of neighboring buildings up/down to include in the Laplacian smoothing.
    """
    if required_columns is None:
        required_columns = {
            "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],
            "radiation": "solar_radiation_w",
            "heat_minus_cool": ["heating_load_w", "cooling_load_w"]
        }

    building_ids = list(all_ts_dict.keys())
    X_list, y_list = [], []
    valid_building_ids = []
    T_in_lag_list = []

    # ----------------------------------
    # Build X and y for each building
    # ----------------------------------
    for bldg_id in building_ids:
        df = all_ts_dict[bldg_id].copy()

        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)

        # Skip building if required columns missing
        skip_building = False
        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                for col in col_spec:
                    if col not in df.columns:
                        skip_building = True
                        break
            else:
                if col_spec not in df.columns:
                    skip_building = True
                    break
        if skip_building:
            continue

        # Create lagged variables
        for col in df.columns:
            if col != "datetime":
                df[f"{col}_lag"] = df[col].shift(1)
        df = df.dropna()
        if df.shape[0] == 0:
            continue

        # Build X
        X_dict = {}
        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                if "heat_minus_cool" in feature_name:
                    X_dict[feature_name] = df[f"{col_spec[0]}_lag"] - df[f"{col_spec[1]}_lag"]
                elif "temp_diff" in feature_name:
                    X_dict[feature_name] = df[f"{col_spec[0]}_lag"] - df[f"{col_spec[1]}_lag"]
            else:
                col = col_spec
                if "lag" in feature_name:
                    col += "_lag"
                X_dict[feature_name] = df[col]

        X = csr_matrix(pd.DataFrame(X_dict).values)
        y = df["indoor_temp_c"].values - df["indoor_temp_c_lag"].values

        X_list.append(X)
        y_list.append(y)
        T_in_lag_list.append(df["indoor_temp_c_lag"].values)
        valid_building_ids.append(bldg_id)

    if len(X_list) == 0:
        print(f"⚠ No usable buildings for zone {zone}, feature set {feature_set}. Skipping ridge regression.")
        return

    n_buildings = len(X_list)
    n_features = X_list[0].shape[1]

    # ----------------------------------
    # Block diagonal X
    # ----------------------------------
    X_block = sp_block_diag(X_list, format="csr")
    y_block = np.concatenate(y_list)

    # ----------------------------------
    # Laplacian smoothing with filter_width
    # ----------------------------------
    L_rows = []
    for i in range(n_buildings):
        for j in range(max(0, i - filter_width), min(n_buildings, i + filter_width + 1)):
            if i == j:
                continue
            row = np.zeros(n_features * n_buildings)
            row[i*n_features: (i+1)*n_features] = 1
            row[j*n_features: (j+1)*n_features] = -1
            L_rows.append(csr_matrix(row))
    if L_rows:
        L = vstack(L_rows, format="csr")
        LtL = L.T @ L
    else:
        LtL = 0

    # ----------------------------------
    # Ridge solve: (XᵀX + λ LᵀL) β = Xᵀy
    # ----------------------------------
    XtX = X_block.T @ X_block
    Xty = X_block.T @ y_block
    A = XtX + lambda_reg * LtL

    beta_reg = spsolve(A, Xty)

    # ----------------------------------
    # Extract coefficients per building
    # ----------------------------------
    results = []
    offset = 0
    for idx, bldg_id in enumerate(valid_building_ids):
        coeffs = beta_reg[offset: offset + n_features]
        offset += n_features

        X_i = X_list[idx]
        y_i = y_list[idx]
        T_in_lag_i = T_in_lag_list[idx][: X_i.shape[0]]  # <-- fix shape mismatch
        T_in_pred = T_in_lag_i + (X_i @ coeffs)
        mse = np.mean((y_i - X_i @ coeffs) ** 2)

        ss_res = np.sum((T_in_lag_list[idx][: X_i.shape[0]] + y_i - T_in_pred) ** 2)
        ss_tot = np.sum((T_in_lag_i + y_i - np.mean(T_in_lag_i + y_i)) ** 2)
        r_squared_reconstructed = 1 - ss_res / ss_tot

        result = {"bldg_id": bldg_id}
        for i, feature_name in enumerate(required_columns.keys(), start=1):
            result[f"k{i}"] = coeffs[i-1]
        result["mse"] = mse
        result["r_squared"] = r_squared_reconstructed
        results.append(result)

    out_csv = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_regularized_{feature_set}_lambda_{lambda_reg}_fw_{filter_width}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"💾 Saved regularized {feature_set} coefficients with filter width {filter_width}: {out_csv}")

def compute_neighbor_smoothed_ridge_existing_imports(
    zone,
    all_ts_dict,
    output_dir,
    lambda_reg=1.0,
    filter_width=1,
    feature_set="baseline"
):
    """
    Regularized regression with neighbor smoothing.
    Saves CSV with the same naming convention as your original function.
    """
    building_ids = list(all_ts_dict.keys())
    X_list, y_list, valid_building_ids = [], [], []
    T_in_lag_list = []

    # -------------------------------
    # Build X and y for each building
    # -------------------------------
    for bldg_id in building_ids:
        df = all_ts_dict[bldg_id].copy()
        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)

        # Create lagged variables
        for col in ["indoor_temp_c", "outdoor_temp_c", "solar_radiation_w", "heating_load_w", "cooling_load_w"]:
            df[f"{col}_lag"] = df[col].shift(1)
        df = df.dropna()
        if df.shape[0] == 0:
            continue

        # Build feature matrix X
        X = np.vstack([
            (df["outdoor_temp_c_lag"] - df["indoor_temp_c_lag"]).values,
            df["solar_radiation_w_lag"].values,
            (df["heating_load_w_lag"] - df["cooling_load_w_lag"]).values
        ]).T

        # Target vector y
        y = df["indoor_temp_c"].values - df["indoor_temp_c_lag"].values

        X_list.append(csr_matrix(X))
        y_list.append(y)
        T_in_lag_list.append(df["indoor_temp_c_lag"].values)
        valid_building_ids.append(bldg_id)

    n_buildings = len(X_list)
    if n_buildings == 0:
        print(f"⚠ No usable buildings for zone {zone}. Skipping.")
        return
    n_features = X_list[0].shape[1]

    # -------------------------------
    # Stack X block-diagonal
    # -------------------------------
    X_block = sp_block_diag(X_list, format="csr")
    y_block = np.concatenate(y_list)

    # -------------------------------
    # Neighbor smoothing term
    # -------------------------------
    L_rows = []
    for i in range(n_buildings):
        neighbors = list(range(max(0, i - filter_width), min(n_buildings, i + filter_width + 1)))
        neighbors.remove(i)
        if not neighbors:
            continue

        for j in neighbors:
            row = np.zeros(n_features * n_buildings)
            row[i*n_features: (i+1)*n_features] = 1
            row[j*n_features: (j+1)*n_features] = -1 / len(neighbors)
            L_rows.append(csr_matrix(row))

    if L_rows:
        L = vstack(L_rows, format="csr")
        LtL = L.T @ L
    else:
        LtL = 0

    # -------------------------------
    # Solve (X^T X + λ LtL) β = X^T y
    # -------------------------------
    XtX = X_block.T @ X_block
    Xty = X_block.T @ y_block
    A = XtX + lambda_reg * LtL
    beta_reg = spsolve(A, Xty)

    # -------------------------------
    # Extract per-building coefficients
    # -------------------------------
    results = []
    offset = 0
    for idx, bldg_id in enumerate(valid_building_ids):
        coeffs = beta_reg[offset: offset + n_features]
        offset += n_features

        X_i = X_list[idx].toarray()
        y_i = y_list[idx]
        T_in_lag_i = T_in_lag_list[idx][: X_i.shape[0]]

        T_in_pred = T_in_lag_i + X_i @ coeffs
        mse = np.mean((y_i - X_i @ coeffs) ** 2)
        ss_res = np.sum((T_in_lag_i + y_i - T_in_pred) ** 2)
        ss_tot = np.sum((T_in_lag_i + y_i - np.mean(T_in_lag_i + y_i)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        results.append({
            "bldg_id": bldg_id,
            "k1": coeffs[0],
            "k2": coeffs[1],
            "k3": coeffs[2],
            "mse": mse,
            "r_squared": r_squared
        })

    # -------------------------------
    # Save results (same naming as your original function)
    # -------------------------------
    out_csv = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_regularized_{feature_set}_lambda_{lambda_reg}_fw_{filter_width}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"💾 Saved neighbor-smoothed regression coefficients: {out_csv}")

def compute_ols_with_top_coeff_rescue(
    zone,
    all_ts_dict,
    output_dir,
    feature_set="baseline",
    required_columns=None,
    top_n=10,
    r_squared_threshold=0.75
):
    """
    OLS regression predicting ΔT = T_in(t) - T_in(t-1) from features.
    Reconstruct T_in_pred = T_in(t-1) + ΔT_pred.
    Compute MSE on ΔT and R² on T_in_pred vs actual T_in.
    
    If a building has R² < r_squared_threshold, try top_n best coefficients
    from other buildings to see if reconstruction improves.
    If none improve above threshold, keep original coefficients and metrics.
    """
    if required_columns is None:
        required_columns = {
            "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],
            "radiation": "solar_radiation_w",
            "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        }

    temp_results = []

    # Step 1: Fit OLS for each building and store original R²
    for bldg_id, df in all_ts_dict.items():
        if df.shape[0] < 2:
            continue

        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)
        df_lag = df.shift(1)

        X_dict = {}
        skip_building = False
        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                if feature_name == "heat_minus_cool_w":
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
                elif feature_name == "temp_diff_lag":
                    if col_spec[0] not in df_lag.columns or col_spec[1] not in df_lag.columns:
                        skip_building = True
                        break
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
            else:
                col = col_spec
                if "lag" in feature_name:
                    col += "_lag"
                    if col not in df_lag.columns:
                        skip_building = True
                        break
                    X_dict[feature_name] = df_lag[col]
                else:
                    if col not in df.columns:
                        skip_building = True
                        break
                    X_dict[feature_name] = df[col]
        if skip_building or not X_dict:
            continue

        X = pd.DataFrame(X_dict)
        y = df["indoor_temp_c"] - df_lag["indoor_temp_c"]
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if X.shape[0] < 2:
            continue

        model = sm.OLS(y, X).fit()
        ΔT_pred = model.predict(X)
        T_in_pred = df_lag.loc[valid_idx, "indoor_temp_c"] + ΔT_pred
        T_in_actual = df.loc[valid_idx, "indoor_temp_c"]

        mse = mean_squared_error(y, ΔT_pred)  # MSE on ΔT
        ss_res = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
        ss_tot = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot  # R² on reconstructed T_in

        result = {"bldg_id": bldg_id, "mse": mse, "r_squared": r_squared}
        for i, col in enumerate(X.columns, start=1):
            result[f"k{i}"] = model.params[col]
        temp_results.append(result)

    # Step 2: Identify top_n coefficients by original R²
    temp_results_sorted = sorted(temp_results, key=lambda x: x["r_squared"], reverse=True)
    top_coeffs = temp_results_sorted[:top_n]

    # Step 3: Rescue buildings with R² < threshold
    final_results = []
    for r in temp_results:
        if r["r_squared"] >= r_squared_threshold:
            # Already good fit
            final_results.append(r)
            continue

        # Attempt rescue
        bldg_id = r["bldg_id"]
        df = all_ts_dict[bldg_id]
        df_lag = df.shift(1)

        X_dict = {}
        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                if feature_name == "heat_minus_cool_w":
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
                elif feature_name == "temp_diff_lag":
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
            else:
                col = col_spec
                if "lag" in feature_name:
                    col += "_lag"
                    X_dict[feature_name] = df_lag[col]
                else:
                    X_dict[feature_name] = df[col]

        X = pd.DataFrame(X_dict)
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        T_in_lag = df_lag.loc[valid_idx, "indoor_temp_c"]
        T_in_actual = df.loc[valid_idx, "indoor_temp_c"]
        y = T_in_actual - T_in_lag

        best_r_squared = r["r_squared"]
        best_mse = r["mse"]
        best_coeffs = r.copy()
        rescued = False

        for candidate in top_coeffs:
            coeff_values = np.array([candidate[f"k{i+1}"] for i in range(len(X.columns))])
            ΔT_pred_candidate = X.values @ coeff_values
            T_in_pred_candidate = T_in_lag + ΔT_pred_candidate

            ss_res_cand = np.sum((T_in_actual.values - T_in_pred_candidate.values) ** 2)
            ss_tot_cand = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
            r2_cand = 1 - ss_res_cand / ss_tot_cand
            mse_cand = mean_squared_error(y, ΔT_pred_candidate)

            if r2_cand > best_r_squared and r2_cand < 1.0:  # never report R² = 1 artificially
                best_r_squared = r2_cand
                best_mse = mse_cand
                best_coeffs = r.copy()
                for i, col_name in enumerate(X.columns, start=1):
                    best_coeffs[f"k{i}"] = coeff_values[i-1]
                if r2_cand >= r_squared_threshold:
                    rescued = True
                    break

        if rescued:
            print(f"⚠ Rescued building {bldg_id} using top coefficients (R² = {best_r_squared:.4f})")
        elif best_r_squared < r_squared_threshold:
            print(f"⚠ Could not rescue building {bldg_id}, keeping original coefficients (R² = {best_r_squared:.4f})")

        best_coeffs["mse"] = best_mse
        best_coeffs["r_squared"] = best_r_squared
        final_results.append(best_coeffs)

    # Step 4: Save results
    out_csv = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}_rescue.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_results).to_csv(out_csv, index=False)
    print(f"💾 Saved {feature_set} regression coefficients with rescue: {out_csv}")

def compute_regression_coefficients_hvac_fix(
    zone, 
    all_ts_dict, 
    output_dir,
    feature_set="baseline",
    required_columns=None,
    r_squared_threshold=0.8,
    r_squared_good=0.95,
    top_n=10
):
    """
    Compute OLS regression with HVAC coefficient rescue strategy.
    
    Strategy:
    1. First pass: fit all buildings normally
    2. Identify top_n buildings with R² > r_squared_good
    3. Compute average γ (k3) from these good buildings
    4. Second pass: for buildings with R² < r_squared_threshold, 
       fix γ to the average and refit only α and β
    
    Args:
        zone: Zone identifier
        all_ts_dict: Dictionary of building timeseries
        output_dir: Output directory
        feature_set: Name identifier for this feature configuration
        required_columns: Dict mapping feature names to dataframe columns
        r_squared_threshold: Buildings below this get rescued
        r_squared_good: Buildings above this are used for average γ
        top_n: Number of good buildings to average for γ
    """
    
    # Default feature set
    if required_columns is None:
        required_columns = {
            "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],
            "radiation": "solar_radiation_w",
            "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        }

    # ========================================
    # FIRST PASS: Fit all buildings normally
    # ========================================
    first_pass_results = []
    
    for bldg_id, df in all_ts_dict.items():
        if df.shape[0] < 2:
            continue

        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)
        df_lag = df.shift(1)

        # Build X dataframe
        X_dict = {}
        skip_building = False
        
        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                if feature_name == "heat_minus_cool_w":
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
                elif feature_name == "temp_diff_lag":
                    if col_spec[0] not in df_lag.columns or col_spec[1] not in df_lag.columns:
                        skip_building = True
                        break
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
            elif "lag" in feature_name:
                col = col_spec
                if col not in df_lag.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df_lag[col]
            else:
                col = col_spec
                if col not in df.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df[col]
        
        if skip_building or not X_dict:
            continue
            
        X = pd.DataFrame(X_dict)
        y = df["indoor_temp_c"] - df_lag["indoor_temp_c"]

        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if X.shape[0] < 2:
            continue

        # Fit model
        model = sm.OLS(y, X).fit()
        y_pred_delta = model.predict(X)

        # Compute R²
        T_in_actual = df.loc[valid_idx, "indoor_temp_c"]
        T_in_lag = df_lag.loc[valid_idx, "indoor_temp_c"]
        T_in_pred = T_in_lag + y_pred_delta

        ss_res = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
        ss_tot = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot
        mse = mean_squared_error(y, y_pred_delta)

        result = {
            "bldg_id": bldg_id,
            "k1": model.params[X.columns[0]],
            "k2": model.params[X.columns[1]],
            "k3": model.params[X.columns[2]],
            "p_k1": model.pvalues[X.columns[0]],
            "p_k2": model.pvalues[X.columns[1]],
            "p_k3": model.pvalues[X.columns[2]],
            "mse": mse,
            "r_squared": r_squared,
            "rescued": False
        }
        
        first_pass_results.append(result)

    # ========================================
    # COMPUTE AVERAGE γ (k3) from good buildings
    # ========================================
    good_buildings = [r for r in first_pass_results if r["r_squared"] > r_squared_good]
    
    if len(good_buildings) < 3:
        print(f"⚠ Warning: Only {len(good_buildings)} buildings with R² > {r_squared_good}. Using all available.")
        good_buildings = sorted(first_pass_results, key=lambda x: x["r_squared"], reverse=True)[:min(top_n, len(first_pass_results))]
    else:
        good_buildings = sorted(good_buildings, key=lambda x: x["r_squared"], reverse=True)[:top_n]
    
    gamma_avg = np.mean([b["k3"] for b in good_buildings])
    print(f"✓ Computed average γ (k3) = {gamma_avg:.6e} from {len(good_buildings)} buildings")
    print(f"  R² range of reference buildings: [{min(b['r_squared'] for b in good_buildings):.4f}, {max(b['r_squared'] for b in good_buildings):.4f}]")

    # ========================================
    # SECOND PASS: Rescue low R² buildings
    # ========================================
    final_results = []
    rescued_count = 0
    
    for result in first_pass_results:
        if result["r_squared"] >= r_squared_threshold:
            # Already good, keep original
            final_results.append(result)
            continue
        
        # Attempt rescue by fixing γ
        bldg_id = result["bldg_id"]
        df = all_ts_dict[bldg_id]
        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)
        df_lag = df.shift(1)

        # Build X for only α and β (exclude HVAC term)
        X_dict_fit = {}
        for feature_name, col_spec in required_columns.items():
            if feature_name == "heat_minus_cool_w":
                continue  # Skip, we'll handle this separately
            elif isinstance(col_spec, list):
                if feature_name == "temp_diff_lag":
                    X_dict_fit[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
            elif "lag" in feature_name:
                col = col_spec
                X_dict_fit[feature_name] = df_lag[col]
            else:
                col = col_spec
                X_dict_fit[feature_name] = df[col]
        
        X_fit = pd.DataFrame(X_dict_fit)
        
        # Compute fixed HVAC contribution
        hvac_load = df_lag["heating_load_w"] - df_lag["cooling_load_w"]
        hvac_contribution = gamma_avg * hvac_load
        
        # Adjusted target: ΔT - γ·Q_HVAC
        y_raw = df["indoor_temp_c"] - df_lag["indoor_temp_c"]
        y_adjusted = y_raw - hvac_contribution

        valid_idx = X_fit.dropna().index
        valid_idx = valid_idx.intersection(y_adjusted.dropna().index)
        
        X_fit = X_fit.loc[valid_idx]
        y_adjusted = y_adjusted.loc[valid_idx]
        y_raw = y_raw.loc[valid_idx]

        if X_fit.shape[0] < 2:
            final_results.append(result)
            continue

        # Refit only α and β
        model_rescued = sm.OLS(y_adjusted, X_fit).fit()
        ΔT_pred_partial = model_rescued.predict(X_fit)
        
        # Full prediction including fixed HVAC term
        ΔT_pred = ΔT_pred_partial + hvac_contribution.loc[valid_idx].values
        
        # Compute metrics
        T_in_actual = df.loc[valid_idx, "indoor_temp_c"]
        T_in_lag = df_lag.loc[valid_idx, "indoor_temp_c"]
        T_in_pred = T_in_lag + ΔT_pred

        mse_new = np.mean((y_raw.values - ΔT_pred) ** 2)
        ss_res = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
        ss_tot = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
        r_squared_new = 1 - ss_res / ss_tot

        # Only keep rescue if it improved R²
        if r_squared_new > result["r_squared"]:
            print(f"✓ Rescued building {bldg_id}: R² {result['r_squared']:.4f} → {r_squared_new:.4f}")
            result_rescued = {
                "bldg_id": bldg_id,
                "k1": model_rescued.params[X_fit.columns[0]],
                "k2": model_rescued.params[X_fit.columns[1]],
                "k3": gamma_avg,
                "p_k1": model_rescued.pvalues[X_fit.columns[0]],
                "p_k2": model_rescued.pvalues[X_fit.columns[1]],
                "p_k3": np.nan,  # No p-value for fixed coefficient
                "mse": mse_new,
                "r_squared": r_squared_new,
                "rescued": True
            }
            final_results.append(result_rescued)
            rescued_count += 1
        else:
            print(f"⚠ Could not improve building {bldg_id}: R² {result['r_squared']:.4f} → {r_squared_new:.4f}, keeping original")
            final_results.append(result)

    print(f"\n📊 Rescue summary: {rescued_count}/{len([r for r in first_pass_results if r['r_squared'] < r_squared_threshold])} buildings improved")

    # ========================================
    # SAVE RESULTS
    # ========================================
    out_csv = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}_hvac_fix.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(final_results)
    
    # Save metadata
    metadata = {
        "feature_set": feature_set,
        "features": list(required_columns.keys()),
        "model_form": "T_in(t) = T_in(t-1) + k1*[T_out(t-1)-T_in(t-1)] + k2*Q_rad(t) + k3*[Q_heat(t-1)-Q_cool(t-1)]",
        "rescue_strategy": "fix_hvac_coefficient",
        "r_squared_threshold": r_squared_threshold,
        "r_squared_good": r_squared_good,
        "gamma_avg": gamma_avg,
        "n_reference_buildings": len(good_buildings),
        "n_rescued": rescued_count,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    metadata_file = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}_hvac_fix_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    results_df.to_csv(out_csv, index=False)
    print(f"💾 Saved {feature_set} regression coefficients with HVAC fix: {out_csv}")
    
    return results_df

def compute_regression_coefficients_alt(
    zone, 
    all_ts_dict, 
    output_dir,
    feature_set="baseline",
    required_columns=None
):
    """
    Compute OLS regression with configurable feature sets and positivity constraints.
    
    Regression form: T_in(t) = T_in(t-1) + k[1] * [T_out(t-1) - T_in(t-1)] + k[2] * Q_rad(t) + k[3] * (q_heating(t-1) - q_cooling(t-1))
    
    Constraints: k1, k2, k3 >= 0 (all coefficients must be non-negative)
    
    If R² < 0.5: Uses alternative regression on [T_out - T_in] = -k2/k1 * Q_rad - k3/k1 * (heat - cool)
    to estimate k2 and k3 using the average k1 from buildings with R² >= 0.5.
    
    Args:
        zone: Zone identifier
        all_ts_dict: Dictionary of building timeseries
        output_dir: Output directory
        feature_set: Name identifier for this feature configuration
        required_columns: Dict mapping feature names to dataframe columns
    """
    results = []
    k1_values = []  # Track k1 values from good fits
    
    # Default feature set
    if required_columns is None:
        required_columns = {
            "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],  # T_out - T_in
            "radiation": "solar_radiation_w",
            "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        }

    # First pass: collect k1 values from buildings with R² >= 0.5
    first_pass_results = []
    
    for bldg_id, df in all_ts_dict.items():
        if df.shape[0] < 2:
            continue

        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)

        df_lag = df.shift(1)

        # Build X dataframe dynamically based on required_columns
        X_dict = {}
        skip_building = False
        
        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                # Handle computed features
                if feature_name == "heat_minus_cool_w":
                    # Use lagged heating and cooling
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
                elif feature_name == "temp_diff_lag":
                    # T_out(t-1) - T_in(t-1)
                    if col_spec[0] not in df_lag.columns or col_spec[1] not in df_lag.columns:
                        skip_building = True
                        break
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
            elif "lag" in feature_name:
                # Lagged variable
                col = col_spec
                if col not in df_lag.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df_lag[col]
            else:
                # Current time variable (like radiation)
                col = col_spec
                if col not in df.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df[col]
        
        if skip_building or not X_dict:
            continue
            
        X = pd.DataFrame(X_dict)
        
        # y = T_in(t) - T_in(t-1), so we're modeling the change
        y = df["indoor_temp_c"] - df_lag["indoor_temp_c"]

        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if X.shape[0] < 2:
            continue

        # Use constrained least squares with non-negativity constraints
        n_features = X.shape[1]
        lower_bounds = np.zeros(n_features)  # All coefficients >= 0
        upper_bounds = np.inf * np.ones(n_features)  # No upper bound
        
        result_lsq = lsq_linear(X.values, y.values, bounds=(lower_bounds, upper_bounds))
        coefficients = result_lsq.x
        
        # Compute predictions
        y_pred_delta = X.values @ coefficients

        # Corrected R²: evaluate on T_in(t), not ΔT
        T_in_actual = df.loc[valid_idx, "indoor_temp_c"]
        T_in_lag    = df_lag.loc[valid_idx, "indoor_temp_c"]
        T_in_pred   = T_in_lag + y_pred_delta

        ss_res = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
        ss_tot = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
        r_squared_reconstructed = 1 - ss_res / ss_tot
        mse_reconstructed = mean_squared_error(T_in_actual.values, T_in_pred.values)
        
        # Store results from first pass
        first_pass_results.append({
            'bldg_id': bldg_id,
            'coefficients': coefficients,
            'r_squared': r_squared_reconstructed,
            'mse': mse_reconstructed,
            'X': X,
            'y': y,
            'valid_idx': valid_idx,
            'df': df,
            'df_lag': df_lag
        })
        
        # Collect k1 values from good fits
        if r_squared_reconstructed >= 0.5:
            k1_values.append(coefficients[0])
    
    # Calculate average k1 from good fits
    if len(k1_values) > 0:
        avg_k1 = np.mean(k1_values)
        print(f"Average k1 from {len(k1_values)} buildings with R² >= 0.5: {avg_k1:.6f}")
    else:
        avg_k1 = 0.01  # Fallback value if no good fits
        print(f"No buildings with R² >= 0.5, using fallback k1: {avg_k1}")
    
    # Second pass: process all buildings with final k1
    for res in first_pass_results:
        bldg_id = res['bldg_id']
        coefficients = res['coefficients']
        r_squared_reconstructed = res['r_squared']
        mse_reconstructed = res['mse']
        
        if r_squared_reconstructed < 0.5:
            # Alternative regression: [T_out - T_in] = -k2/k1 * Q_rad - k3/k1 * (heat - cool)
            X = res['X']
            valid_idx = res['valid_idx']
            df = res['df']
            df_lag = res['df_lag']
            
            # Left side: T_out(t-1) - T_in(t-1)
            y_alt = df_lag.loc[valid_idx, "outdoor_temp_c"] - df_lag.loc[valid_idx, "indoor_temp_c"]
            
            # Right side: Q_rad and (heat - cool)
            X_alt_dict = {}
            X_alt_dict['radiation'] = X['radiation']
            X_alt_dict['heat_minus_cool_w'] = X['heat_minus_cool_w']
            X_alt = pd.DataFrame(X_alt_dict)
            
            # Regression with positivity constraint (these are -k2/k1 and -k3/k1, so should be negative)
            # Actually, we want to find beta2 and beta3 where:
            # T_out - T_in = beta2 * Q_rad + beta3 * (heat-cool)
            # Then: -k2/k1 = beta2, -k3/k1 = beta3
            # So: k2 = -beta2 * k1, k3 = -beta3 * k1
            
            # No constraints on this regression
            result_lsq_alt = lsq_linear(X_alt.values, y_alt.values)
            coefficients_alt = result_lsq_alt.x
            
            # Calculate k2 and k3 from average k1
            k2 = -coefficients_alt[0] * avg_k1
            k3 = -coefficients_alt[1] * avg_k1
            
            # Ensure positivity
            k2 = max(0, k2)
            k3 = max(0, k3)
            
            # Update coefficients
            coefficients = np.array([avg_k1, k2, k3])
            
            # Recalculate predictions with new coefficients
            y = res['y']
            y_pred_delta = X.values @ coefficients
            
            T_in_actual = df.loc[valid_idx, "indoor_temp_c"]
            T_in_lag = df_lag.loc[valid_idx, "indoor_temp_c"]
            T_in_pred = T_in_lag + y_pred_delta
            
            ss_res = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
            ss_tot = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
            r_squared_reconstructed = 1 - ss_res / ss_tot
            mse_reconstructed = mean_squared_error(T_in_actual.values, T_in_pred.values)
        
        # Build result dict - save only bldg_id, k1, k2, k3, mse, r_squared
        result = {
            "bldg_id": bldg_id,
            "k1": coefficients[0],
            "k2": coefficients[1],
            "k3": coefficients[2],
            "mse": mse_reconstructed,
            "r_squared": r_squared_reconstructed
        }
        
        results.append(result)

    # Save with feature_set in filename
    out_csv = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}_alt.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    
    # Save column mapping for reference
    metadata = {
        "feature_set": feature_set,
        "features": list(required_columns.keys()),
        "constraints": "All coefficients (k1, k2, k3) constrained to be non-negative (>= 0)",
        "alternative_regression": "For R² < 0.5: [T_out - T_in] = -k2/k1 * Q_rad - k3/k1 * (heat - cool), using average k1",
        "average_k1": float(avg_k1),
        "n_good_fits": len(k1_values),
        "model_form": "T_in(t) = T_in(t-1) + k1*[T_out(t-1)-T_in(t-1)] + k2*Q_rad(t) + k3*[Q_heat(t-1)-Q_cool(t-1)]",
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Save metadata alongside results
    metadata_file = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    results_df.to_csv(out_csv, index=False)
    print(f"💾 Saved {feature_set} regression coefficients: {out_csv}")

# ============================================================
# Function 4: Plot largest MSE buildings
# ============================================================
def plot_largest_mse_buildings(
    zone, 
    all_ts_dict, 
    output_dir, 
    feature_set="baseline",
    regularized=False, 
    lambda_reg=0
):
    """
    Plot the 3 buildings with largest MSE for a given feature set.
    """

    if regularized:
        coeff_file = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_regularized_{feature_set}_lambda_{lambda_reg}.csv"
    else:
        coeff_file = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}.csv"

    if not coeff_file.exists():
        print(f"❌ Regression coefficients file not found: {coeff_file}")
        return

    coeffs_df = pd.read_csv(coeff_file)
    
    # Load metadata to determine feature mapping
    if not regularized:
        metadata_file = output_dir / zone / "input_timeseries" / f"{zone}_regression_coeff_{feature_set}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                feature_names = metadata.get("features", [])
        else:
            feature_names = ["temp_diff_lag", "radiation", "heat_minus_cool_w"]
    else:
        feature_names = ["temp_diff_lag", "radiation", "heat_minus_cool"]

    mse_list = []

    for bldg_id, df in all_ts_dict.items():

        df["heating_load_w"] = df["heating_load_w"].fillna(0)
        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)

        df_lag = df.shift(1)

        # Build X to match feature names
        X_dict = {}
        skip_building = False
        
        for feature_name in feature_names:
            if "heat_minus_cool" in feature_name:
                X_dict[feature_name] = df_lag["heating_load_w"] - df_lag["cooling_load_w"]
            elif feature_name == "temp_diff_lag":
                if "outdoor_temp_c" not in df_lag.columns or "indoor_temp_c" not in df_lag.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df_lag["outdoor_temp_c"] - df_lag["indoor_temp_c"]
            elif feature_name == "radiation":
                if "solar_radiation_w" not in df.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df["solar_radiation_w"]
            elif "lag" in feature_name:
                # Get base column name
                base_col = feature_name.replace("_lag", "")
                if base_col not in df.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df_lag[base_col]
            else:
                if feature_name not in df.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df[feature_name]
        
        if skip_building:
            continue
            
        X = pd.DataFrame(X_dict)

        coeff_row = coeffs_df[coeffs_df["bldg_id"] == int(bldg_id)]
        if coeff_row.empty:
            continue

        # Extract coefficients dynamically (no k0)
        k_values = []
        for i in range(1, len(feature_names) + 1):
            k_col = f"k{i}"
            if k_col in coeff_row.columns:
                k_values.append(coeff_row[k_col].values[0])

        # Compute prediction: delta_T = k1*temp_diff + k2*radiation + k3*heat_minus_cool
        delta_T_pred = 0
        for i, feature_name in enumerate(feature_names):
            if i < len(k_values):
                delta_T_pred = delta_T_pred + k_values[i] * X[feature_name]

        # Reconstruct T_in(t) from T_in(t-1) + delta_T
        y_pred = df_lag["indoor_temp_c"] + delta_T_pred
        
        # Compute MSE on actual vs predicted T_in(t)
        valid_idx = y_pred.dropna().index
        mse = np.mean((df.loc[valid_idx, "indoor_temp_c"].values - y_pred.loc[valid_idx].values) ** 2)
        mse_list.append((bldg_id, mse, df["datetime"], df["indoor_temp_c"], y_pred))

    if not mse_list:
        print("❌ No buildings to plot.")
        return

    top3 = sorted(mse_list, key=lambda x: x[1], reverse=True)[:3]
    plots_folder = output_dir / zone / "largest_mse_plots"
    plots_folder.mkdir(exist_ok=True)

    plt.ion()
    for bldg_id, mse, dt, y_true, y_pred in top3:
        plt.figure(figsize=(12, 5))
        plt.plot(dt, y_true, label="Actual")
        plt.plot(dt, y_pred, label="Reconstructed")
        title = f"{zone} | {feature_set} | Building {bldg_id} | MSE={mse:.3f}"
        if regularized:
            title += f" | λ={lambda_reg}"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        filename = f"{zone}_{feature_set}_{bldg_id}"
        if regularized:
            filename += f"_lambda_{lambda_reg}"
        plt.savefig(plots_folder / f"{filename}.png", dpi=300)
        plt.show(block=False)
        plt.pause(0.5)

    plt.show(block=True)


# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":

    overwrite = True     # If False: skip zone if input_timeseries already exists
    save_flag = False     # Save per-building CSVs
    verbose = True        # Print detailed progress messages

    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]

    input_dir = parent_dir / "in"
    output_dir = parent_dir / "out"

    ts_config = {
        "indoor_temp_key": "out.indoor_temperature.conditioned_space..c",
        "cooling_load_key": "out.load.cooling.energy_delivered..kbtu",
        "heating_load_key": "out.load.heating.energy_delivered..kbtu",
        "outdoor_temp_key": "Dry Bulb Temperature [°C]",
        "dnr_radiation_key": "Direct Normal Radiation [W/m2]",
        "outdoor_humidity_key": "Relative Humidity [%]",
        "wind_speed_key": "Wind Speed [m/s]",
    }
    
    filter_width = 5 # Filter width for the regularization step

    # =========================================================
    # DEFINE FEATURE SETS
    # =========================================================
    FEATURE_SETS = {
        "baseline": {
            "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],  # T_out(t-1) - T_in(t-1)
            "radiation": "solar_radiation_w",  # Q_rad(t) - now in total Watts
            "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]  # Q_heat(t-1) - Q_cool(t-1)
        },
        
        # "with_humidity": {
        #     "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],
        #     "outdoor_humidity_lag": "outdoor_humidity",
        #     "radiation": "solar_radiation_w",
        #     "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        # },
        
        # "with_wind": {
        #     "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],
        #     "wind_speed_lag": "wind_speed",
        #     "radiation": "solar_radiation_w",
        #     "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        # },
        
        # "with_humidity_wind": {
        #     "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],
        #     "outdoor_humidity_lag": "outdoor_humidity",
        #     "wind_speed_lag": "wind_speed",
        #     "radiation": "solar_radiation_w",
        #     "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        # },
    }

    lambda_values = [10.0, 100.0, 1000]
    zone_pattern = re.compile(r"^P\d+[UR]$")

    zone_folders = [f for f in output_dir.iterdir() if f.is_dir()]
    zones = [z.name for z in zone_folders if zone_pattern.match(z.name)]
    total_zones = len(zones)

    print(f"\n=== Found {total_zones} zones to process ===")
    print(f"=== Feature sets: {', '.join(FEATURE_SETS.keys())} ===\n")

    t_start_total = time.time()

    # ---------------------------------------------------------
    # Zone processing loop
    # ---------------------------------------------------------
    for idx, zone in enumerate(zones, start=1):
        print(f"\n🔹 Processing zone {idx}/{total_zones}: {zone}")

        ts_folder = output_dir / zone / "input_timeseries"

        # SKIP IF ALREADY DONE
        if ts_folder.exists() and not overwrite:
            if verbose:
                print(f"⏭  Skipping zone {zone} (input_timeseries exists).")
        else:
            # 1. Create per-building timeseries
            all_ts_dict = create_building_timeseries(
                zone=zone,
                input_dir=input_dir,
                output_dir=output_dir,
                ts_config=ts_config,
                save_outputs=save_flag,
                skip_missing_weather=True,
            )

            # 2. Run regressions for each feature set
            for feature_name, feature_cols in FEATURE_SETS.items():
                print(f"\n  📊 Running {feature_name} regression...")
                
                # OLS regression
                compute_regression_coefficients(
                    zone=zone,
                    all_ts_dict=all_ts_dict,
                    output_dir=output_dir,
                    feature_set=feature_name,
                    required_columns=feature_cols
                )
                
                # Regularized ridge regression (multiple lambda values)
                # for lam in lambda_values:
                #     compute_regularized_ridge_coefficients_sparse(
                #         zone=zone,
                #         all_ts_dict=all_ts_dict,
                #         output_dir=output_dir,
                #         lambda_reg=lam,
                #         feature_set=feature_name,
                #         required_columns=feature_cols
                #     )

                # for lam in lambda_values:
                #     compute_regularized_ridge_coefficients_sparse_fw(
                #         zone=zone,
                #         all_ts_dict=all_ts_dict,
                #         output_dir=output_dir,
                #         lambda_reg=lam,
                #         feature_set=feature_name,
                #         required_columns=feature_cols,
                #         filter_width=filter_width
                #     )
                # for lam in lambda_values:
                #     compute_neighbor_smoothed_ridge_existing_imports(
                #     zone=zone,
                #     all_ts_dict=all_ts_dict,
                #     output_dir=output_dir,
                #     lambda_reg=lam,
                #     filter_width=filter_width,
                #     feature_set=feature_name
                #     )
                # compute_ols_with_top_coeff_rescue(
                # zone=zone,
                # all_ts_dict=all_ts_dict,
                # output_dir=output_dir,
                # feature_set=feature_name
                # )
                # compute_regression_coefficients_hvac_fix(
                #     zone=zone,
                #     all_ts_dict=all_ts_dict,
                #     output_dir=output_dir,
                #     feature_set=feature_name,
                #     required_columns=feature_cols,
                #     r_squared_threshold=0.8,    # Rescue buildings below this
                #     r_squared_good=0.95,         # Use buildings above this for average γ
                #     top_n=10                     # Use top 10 buildings for average
                # )
                # compute_regression_coefficients_alt(
                #     zone=zone,
                #     all_ts_dict=all_ts_dict,
                #     output_dir=output_dir,
                #     feature_set=feature_name,
                #     required_columns=feature_cols
                # )
        # ---------------------------------------------------------
        # PROGRESS PRINT
        # ---------------------------------------------------------
        t_elapsed_total = time.time() - t_start_total
        hrs = int(t_elapsed_total // 3600)
        mins = int((t_elapsed_total % 3600) // 60)
        secs = int(t_elapsed_total % 60)

        print(
            f"📊 Progress: {idx}/{total_zones} zones processed "
            f"({100*idx/total_zones:.1f}%) | "
            f"Elapsed time: {hrs}h {mins}m {secs}s"
        )

    print("\n🎉 All zones completed successfully!\n")