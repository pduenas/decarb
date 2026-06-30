import time
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json
from scipy.optimize import lsq_linear

# ============================================================
# Function 1: Create building timeseries from a list of bldg_ids
# ============================================================
def create_building_timeseries(
    bldg_ids, input_dir, output_dir, ts_config,
    save_outputs=False,
    skip_missing_weather=False,
):
    """
    Builds per-building timeseries for a given list of building IDs.
    Returns a dict of DataFrames keyed by bldg_id.

    Args:
        bldg_ids: List/array of unique building IDs to process
        input_dir: Path to input directory
        output_dir: Path to output directory (consumption_files must be here)
        ts_config: Dict of column name keys for temperature, load, weather
        save_outputs: If True, saves per-building CSVs
        skip_missing_weather: If True, keeps buildings with no weather data
    """

    bldg_consumption_dir = output_dir / "consumption_files"
    weather_dir = input_dir / "TX_weather_data"

    all_ts_dict = {}

    # ----------------------------
    # Load ResStock metadata for square footage and county
    # ----------------------------
    resstock_file = input_dir / "TX_upgrade0.parquet"
    if not resstock_file.exists():
        raise FileNotFoundError(f"Missing ResStock parquet file: {resstock_file}")

    resstock_df = pd.read_parquet(resstock_file, engine="fastparquet")

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
    resstock_df["sqm"] = resstock_df["sqft"] * 0.092903

    # ----------------------------
    # Build a DataFrame from the provided bldg_ids and merge with ResStock
    # ----------------------------
    bldg_df = pd.DataFrame({"bldg_id": pd.array(list(bldg_ids), dtype="Int64")})

    merged_bldg_df = pd.merge(
        bldg_df,
        resstock_df,
        on="bldg_id",
        how="left",
        validate="m:1"
    )

    print(f"  Loading timeseries for {len(merged_bldg_df)} buildings...")

    # ----------------------------
    # Building loop
    # ----------------------------
    for _, row in merged_bldg_df.iterrows():

        bldg_id     = row["bldg_id"]
        county_code = row["county_code"]
        sqm         = row["sqm"]

        if pd.isna(bldg_id):
            continue

        if pd.isna(sqm):
            print(f"    Warning: Missing square footage for building {bldg_id}")
            continue

        bldg_path = bldg_consumption_dir / f"{bldg_id}-0.parquet"
        if not bldg_path.exists():
            print(f"    Warning: Missing consumption file for {bldg_id}")
            continue

        # Load building parquet
        try:
            df_bldg = pd.read_parquet(bldg_path, engine="fastparquet")
            if "timestamp" in df_bldg.columns:
                df_bldg["datetime"] = pd.to_datetime(df_bldg["timestamp"])
            else:
                print(f"    Error: {bldg_path.name} missing 'timestamp'")
                continue

            df_cols = df_bldg.columns
            if ts_config["indoor_temp_key"] not in df_cols:
                print(f"    Error: Missing indoor temp for {bldg_id}")
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
            print(f"    Error reading {bldg_id}-0.parquet: {e}")
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

                    outdoor_col = next(
                        (c for c in df_weather.columns
                         if ts_config["outdoor_temp_key"].lower().replace(" ", "")
                         in c.lower().replace(" ", "")),
                        None,
                    )
                    dnr_col = next(
                        (c for c in df_weather.columns
                         if ts_config["dnr_radiation_key"].lower().replace(" ", "")
                         in c.lower().replace(" ", "")),
                        None,
                    )

                    if "outdoor_humidity_key" in ts_config:
                        humidity_col = next(
                            (c for c in df_weather.columns
                             if ts_config["outdoor_humidity_key"].lower().replace(" ", "")
                             in c.lower().replace(" ", "")),
                            None,
                        )
                    else:
                        humidity_col = next(
                            (c for c in df_weather.columns
                             if "humidity" in c.lower() or "rh" in c.lower()),
                            None,
                        )

                    if "wind_speed_key" in ts_config:
                        wind_col = next(
                            (c for c in df_weather.columns
                             if ts_config["wind_speed_key"].lower().replace(" ", "")
                             in c.lower().replace(" ", "")),
                            None,
                        )
                    else:
                        wind_col = next(
                            (c for c in df_weather.columns
                             if "wind" in c.lower() and "speed" in c.lower()),
                            None,
                        )

                    if outdoor_col is None or dnr_col is None:
                        df_weather = None
                    else:
                        df_weather["datetime"] = pd.to_datetime(df_weather["date_time"])

                        rename_dict = {
                            dnr_col: "solar_radiation_wm2",
                            outdoor_col: "outdoor_temp_c",
                        }
                        if humidity_col is not None:
                            rename_dict[humidity_col] = "outdoor_humidity"
                        if wind_col is not None:
                            rename_dict[wind_col] = "wind_speed"

                        df_weather = df_weather.rename(columns=rename_dict)
                        df_weather["solar_radiation_w"] = df_weather["solar_radiation_wm2"] * sqm

                        cols_to_keep = ["datetime", "solar_radiation_w", "outdoor_temp_c"]
                        if "outdoor_humidity" in df_weather.columns:
                            cols_to_keep.append("outdoor_humidity")
                        if "wind_speed" in df_weather.columns:
                            cols_to_keep.append("wind_speed")

                        df_weather = df_weather[cols_to_keep]
                        df_weather.set_index("datetime", inplace=True)
                        df_weather = df_weather.reindex(df_bldg["datetime"]).interpolate("linear")
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
            save_dir = output_dir / "thermal_model" / "timeseries"
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{bldg_id}_in_ts.csv"
            merged.to_csv(out_path, index=False)

    print(f"  Timeseries loaded: {len(all_ts_dict)} / {len(merged_bldg_df)} buildings.")
    return all_ts_dict


# ============================================================
# Function 2: Regression coefficients
# ============================================================
def compute_regression_coefficients(
    all_ts_dict,
    output_dir,
    feature_set="baseline",
    required_columns=None
):
    """
    Compute constrained least-squares regression (all k >= 0) for each building.

    Indoor temperatures are rounded to the nearest 0.5 degrees C before regression.
    Both regression fitting and error metrics use this rounded temperature.

    Results saved to: out/thermal_model/<feature_set>_v2/regression_coeff_<feature_set>_v2.csv

    When called in batches via run_pipeline, results are appended across batches.

    Args:
        all_ts_dict: Dict of building DataFrames keyed by bldg_id
        output_dir: Path to the "out" directory
        feature_set: Name identifier for this feature configuration
        required_columns: Dict mapping feature names to dataframe columns
    """
    results = []

    if required_columns is None:
        required_columns = {
            "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],
            "radiation": "solar_radiation_w",
            "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        }

    for bldg_id, df in all_ts_dict.items():
        if df.shape[0] < 2:
            continue

        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)

        # Round indoor temperature to nearest 0.5 degrees C before regression.
        # Both the regression fitting and all error metrics use this rounded value.
        df["indoor_temp_c"] = (df["indoor_temp_c"] * 2).round() / 2

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

        # Constrained least squares: all coefficients >= 0
        n_features = X.shape[1]
        result_lsq = lsq_linear(
            X.values, y.values,
            bounds=(np.zeros(n_features), np.inf * np.ones(n_features))
        )
        coefficients = result_lsq.x

        y_pred_delta = X.values @ coefficients

        T_in_actual = df.loc[valid_idx, "indoor_temp_c"]
        T_in_lag    = df_lag.loc[valid_idx, "indoor_temp_c"]
        T_in_pred   = T_in_lag + y_pred_delta

        ss_res = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
        ss_tot = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
        # ss_tot can be zero if indoor temp is constant after rounding to 0.5 degrees C
        r_squared_reconstructed = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        mse_reconstructed = mean_squared_error(T_in_actual.values, T_in_pred.values)

        results.append({
            "bldg_id": bldg_id,
            "k1": coefficients[0],  # temp_diff_lag
            "k2": coefficients[1],  # radiation
            "k3": coefficients[2],  # heat_minus_cool_w
            "mse": mse_reconstructed,
            "r_squared": r_squared_reconstructed
        })

    return results


# ============================================================
# Function 3: Run pipeline (batched or single pass)
# ============================================================
def run_pipeline(
    bldg_ids, input_dir, output_dir, ts_config, feature_sets,
    save_outputs=False,
    skip_missing_weather=False,
    batch=False,
    batch_size=100,
):
    """
    Runs create_building_timeseries + compute_regression_coefficients for all
    buildings, either in one pass or in batches of batch_size.

    When batch=True, each batch independently:
      1. Builds timeseries for its buildings
      2. Runs regression on those buildings
      3. Appends results immediately to the output CSV on disk
    No results are held in memory across batches.

    When batch=False, all buildings are processed in a single pass and saved
    at the end.

    Output saved to: out/thermal_model/<feature_set>_v2/

    Args:
        bldg_ids: Array of unique building IDs to process
        input_dir: Path to input directory
        output_dir: Path to output directory
        ts_config: Dict of column name keys for temperature, load, weather
        feature_sets: Dict of {feature_set_name: required_columns_dict}
        save_outputs: If True, saves per-building timeseries CSVs
        skip_missing_weather: If True, keeps buildings with no weather data
        batch: If True, process buildings in batches of batch_size
        batch_size: Number of buildings per batch (only used when batch=True)
    """

    bldg_ids_list = list(bldg_ids)
    total = len(bldg_ids_list)

    # Pre-create output dirs and write metadata once up front
    for feature_name, feature_cols in feature_sets.items():
        out_dir = output_dir / "thermal_model" / f"{feature_name}_v2"
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "feature_set": feature_name,
            "features": list(feature_cols.keys()),
            "constraints": "All coefficients (k1, k2, k3) constrained to be non-negative (>= 0)",
            "indoor_temp_rounding": "Indoor temperatures rounded to nearest 0.5 degrees C before regression and error calculation",
            "model_form": "T_in(t) = T_in(t-1) + k1*[T_out(t-1)-T_in(t-1)] + k2*Q_rad(t) + k3*[Q_heat(t-1)-Q_cool(t-1)]",
            "source": "ercot_substation_nrel_map.parquet (unique bldg_ids)",
            "batched": batch,
            "batch_size": batch_size if batch else None,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        metadata_file = out_dir / f"regression_coeff_{feature_name}_v2_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    if batch:
        n_batches = (total + batch_size - 1) // batch_size
        print(f"Running in batched mode: {total} buildings, "
              f"{n_batches} batches of {batch_size}")

        total_saved = {fs: 0 for fs in feature_sets}

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end   = min(batch_start + batch_size, total)
            batch_ids   = bldg_ids_list[batch_start:batch_end]

            print(f"\n--- Batch {batch_idx + 1}/{n_batches} "
                  f"(buildings {batch_start + 1}-{batch_end} of {total}) ---")

            t_batch = time.time()

            # Step 1: build timeseries for this batch
            ts_dict = create_building_timeseries(
                bldg_ids=batch_ids,
                input_dir=input_dir,
                output_dir=output_dir,
                ts_config=ts_config,
                save_outputs=save_outputs,
                skip_missing_weather=skip_missing_weather,
            )

            # Step 2: run regression and save results for each feature set
            for feature_name, feature_cols in feature_sets.items():
                batch_results = compute_regression_coefficients(
                    all_ts_dict=ts_dict,
                    output_dir=output_dir,
                    feature_set=feature_name,
                    required_columns=feature_cols,
                )

                if batch_results:
                    out_csv = output_dir / "thermal_model" / f"{feature_name}_v2" / f"regression_coeff_{feature_name}_v2.csv"
                    batch_df = pd.DataFrame(batch_results)
                    # Write header only on the first batch
                    write_header = not out_csv.exists()
                    batch_df.to_csv(out_csv, mode="a", header=write_header, index=False)
                    total_saved[feature_name] += len(batch_df)

            t_elapsed = time.time() - t_batch
            saved_str = " | ".join(f"{fs}: {total_saved[fs]}" for fs in feature_sets)
            print(f"    Batch {batch_idx + 1}/{n_batches} done in {t_elapsed:.1f}s | "
                  f"Buildings saved so far -> {saved_str}")

        for feature_name in feature_sets:
            out_csv = output_dir / "thermal_model" / f"{feature_name}_v2" / f"regression_coeff_{feature_name}_v2.csv"
            print(f"\nSaved {feature_name} coefficients "
                  f"({total_saved[feature_name]} buildings): {out_csv}")

    else:
        print(f"Running in single-pass mode: {total} buildings")

        # Step 1: timeseries for all buildings at once
        ts_dict = create_building_timeseries(
            bldg_ids=bldg_ids_list,
            input_dir=input_dir,
            output_dir=output_dir,
            ts_config=ts_config,
            save_outputs=save_outputs,
            skip_missing_weather=skip_missing_weather,
        )

        # Step 2: regression and save for each feature set
        for feature_name, feature_cols in feature_sets.items():
            results = compute_regression_coefficients(
                all_ts_dict=ts_dict,
                output_dir=output_dir,
                feature_set=feature_name,
                required_columns=feature_cols,
            )

            results_df = pd.DataFrame(results)
            out_csv = output_dir / "thermal_model" / f"{feature_name}_v2" / f"regression_coeff_{feature_name}_v2.csv"
            results_df.to_csv(out_csv, index=False)
            print(f"\nSaved {feature_name} coefficients "
                  f"({len(results_df)} buildings): {out_csv}")


# ============================================================
# Function 4: Plot lowest R-squared buildings
# ============================================================
def plot_lowest_r2_buildings(
    all_ts_dict,
    output_dir,
    feature_set="baseline"
):
    """
    Plot the 3 buildings with lowest R-squared for a given feature set.
    Indoor temperatures are rounded to nearest 0.5 degrees C to match regression preprocessing.
    """

    coeff_file = output_dir / "thermal_model" / f"{feature_set}_v2" / f"regression_coeff_{feature_set}_v2.csv"
    if not coeff_file.exists():
        print(f"Regression coefficients file not found: {coeff_file}")
        return

    coeffs_df = pd.read_csv(coeff_file)

    metadata_file = output_dir / "thermal_model" / f"{feature_set}_v2" / f"regression_coeff_{feature_set}_v2_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            feature_names = metadata.get("features", [])
    else:
        feature_names = ["temp_diff_lag", "radiation", "heat_minus_cool_w"]

    r_squared_list = []

    for bldg_id, df in all_ts_dict.items():

        df["heating_load_w"] = df["heating_load_w"].fillna(0)
        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)

        # Round indoor temperature to nearest 0.5 degrees C to match regression preprocessing
        df["indoor_temp_c"] = (df["indoor_temp_c"] * 2).round() / 2

        df_lag = df.shift(1)

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

        k_values = []
        for i in range(1, len(feature_names) + 1):
            k_col = f"k{i}"
            if k_col in coeff_row.columns:
                k_values.append(coeff_row[k_col].values[0])

        delta_T_pred = sum(
            k_values[i] * X[fname]
            for i, fname in enumerate(feature_names)
            if i < len(k_values)
        )

        y_pred = df_lag["indoor_temp_c"] + delta_T_pred

        valid_idx = y_pred.dropna().index
        T_in_actual = df.loc[valid_idx, "indoor_temp_c"].values
        T_in_pred_vals = y_pred.loc[valid_idx].values

        ss_res = np.sum((T_in_actual - T_in_pred_vals) ** 2)
        ss_tot = np.sum((T_in_actual - T_in_actual.mean()) ** 2)
        r_squared = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        r_squared_list.append((bldg_id, r_squared, df["datetime"], df["indoor_temp_c"], y_pred))

    if not r_squared_list:
        print("No buildings to plot.")
        return

    bottom3 = sorted(
        [e for e in r_squared_list if not np.isnan(e[1])],
        key=lambda x: x[1]
    )[:3]

    plots_folder = output_dir / "thermal_model" / f"{feature_set}_v2" / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)

    plt.ion()
    for bldg_id, r_squared, dt, y_true, y_pred in bottom3:
        plt.figure(figsize=(12, 5))
        plt.plot(dt, y_true, label="Actual (rounded)")
        plt.plot(dt, y_pred, label="Reconstructed")
        plt.title(f"{feature_set} | Building {bldg_id} | R2={r_squared:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_folder / f"{feature_set}_{bldg_id}_low_r2.png", dpi=300)
        plt.show(block=False)
        plt.pause(0.5)

    plt.show(block=True)


# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":

    use_batching = True  # Set to False to process all buildings in one pass
    batch_size   = 100   # Only used when use_batching=True
    save_flag    = False # Save per-building timeseries CSVs

    current_path = Path(__file__).resolve()
    parent_dir   = current_path.parents[1]

    input_dir  = parent_dir / "in"
    output_dir = parent_dir / "out"

    ts_config = {
        "indoor_temp_key":      "out.indoor_temperature.conditioned_space..c",
        "cooling_load_key":     "out.load.cooling.energy_delivered..kbtu",
        "heating_load_key":     "out.load.heating.energy_delivered..kbtu",
        "outdoor_temp_key":     "Dry Bulb Temperature [°C]",
        "dnr_radiation_key":    "Direct Normal Radiation [W/m2]",
        "outdoor_humidity_key": "Relative Humidity [%]",
        "wind_speed_key":       "Wind Speed [m/s]",
    }

    # =========================================================
    # DEFINE FEATURE SETS
    # =========================================================
    FEATURE_SETS = {
        "baseline": {
            "temp_diff_lag":     ["outdoor_temp_c", "indoor_temp_c"],
            "radiation":         "solar_radiation_w",
            "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        },
        # Uncomment to enable additional feature sets:
        # "with_humidity": {
        #     "temp_diff_lag":        ["outdoor_temp_c", "indoor_temp_c"],
        #     "outdoor_humidity_lag": "outdoor_humidity",
        #     "radiation":            "solar_radiation_w",
        #     "heat_minus_cool_w":    ["heating_load_w", "cooling_load_w"]
        # },
        # "with_wind": {
        #     "temp_diff_lag":     ["outdoor_temp_c", "indoor_temp_c"],
        #     "wind_speed_lag":    "wind_speed",
        #     "radiation":         "solar_radiation_w",
        #     "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        # },
    }

    # =========================================================
    # LOAD UNIQUE BUILDING IDs FROM SUBSTATION MAP
    # =========================================================
    substation_map_file = output_dir / "ercot_substation_nrel_map.parquet"
    if not substation_map_file.exists():
        raise FileNotFoundError(f"Missing substation map: {substation_map_file}")

    substation_df = pd.read_parquet(substation_map_file, engine="fastparquet")
    bldg_ids = pd.to_numeric(
        substation_df["bldg_id"], errors="coerce"
    ).dropna().astype("Int64").unique()

    print(f"\n=== Found {len(bldg_ids)} unique building IDs in substation map ===")
    print(f"=== Feature sets: {', '.join(FEATURE_SETS.keys())} ===")
    print(f"=== Batching: {use_batching} | Batch size: {batch_size if use_batching else 'N/A'} ===\n")

    t_start = time.time()

    run_pipeline(
        bldg_ids=bldg_ids,
        input_dir=input_dir,
        output_dir=output_dir,
        ts_config=ts_config,
        feature_sets=FEATURE_SETS,
        save_outputs=save_flag,
        skip_missing_weather=True,
        batch=use_batching,
        batch_size=batch_size,
    )

    t_elapsed = time.time() - t_start
    hrs  = int(t_elapsed // 3600)
    mins = int((t_elapsed % 3600) // 60)
    secs = int(t_elapsed % 60)

    print(f"\nDone! Elapsed time: {hrs}h {mins}m {secs}s\n")