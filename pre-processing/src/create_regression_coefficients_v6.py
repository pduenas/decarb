import time
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json
from scipy.optimize import lsq_linear
from scipy import stats as scipy_stats


# ============================================================
# Helper: Newey-West HAC standard errors
# ============================================================
def newey_west_cov(X, residuals, n_lags=None):
    """
    Returns the full HAC covariance matrix (k x k), not just the diagonal SEs.
    """
    T, k = X.shape
    if n_lags is None:
        n_lags = int(4 * (T / 100) ** (2 / 9))

    Xe = X * residuals[:, np.newaxis]
    S  = Xe.T @ Xe / T
    for lag in range(1, n_lags + 1):
        w     = 1 - lag / (n_lags + 1)
        gamma = (Xe[lag:].T @ Xe[:-lag]) / T
        S    += w * (gamma + gamma.T)

    XtX = X.T @ X / T
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    V = XtX_inv @ S @ XtX_inv / T
    return V

# ============================================================
# Function 1: Create building timeseries from a list of bldg_ids
# ============================================================
def create_building_timeseries(
    bldg_ids, input_dir, output_dir, ts_config,
    save_outputs=False,
    skip_missing_weather=False,
):
    bldg_consumption_dir = output_dir / "consumption_files"

    SOLAR_COL        = "out.weather.direct_normal_solar_radiation..watt_per_m2"
    WIND_COL         = "out.weather.wind_speed..meter_per_second"
    HUMIDITY_COL     = "out.outdoor_air_relative_humidity..percentage"
    OUTDOOR_TEMP_COL = "out.outdoor_air_drybulb_temp..c"

    LIGHTING_COL  = "out.electricity.lighting_interior.energy_consumption..kwh"
    EQUIPMENT_COL = "out.electricity.plug_loads.energy_consumption..kwh"
    DHW_COL       = "out.load.hot_water.energy_delivered..kbtu"
    PEOPLE_COL    = "out.people.total_heating_rate..watt"

    LIGHTING_FRACTION  = 1.0
    EQUIPMENT_FRACTION = 0.9
    DHW_FRACTION       = 0

    all_ts_dict = {}

    resstock_file = input_dir / "TX_upgrade0.parquet"
    if not resstock_file.exists():
        raise FileNotFoundError(f"Missing ResStock parquet file: {resstock_file}")

    resstock_df = pd.read_parquet(resstock_file, engine="fastparquet")
    if "in.sqft..ft2" not in resstock_df.columns:
        raise KeyError("Missing 'in.sqft..ft2' column in ResStock data")

    resstock_df = resstock_df[["bldg_id", "in.sqft..ft2", "in.county"]].copy()
    resstock_df = resstock_df.rename(columns={"in.sqft..ft2": "sqft", "in.county": "county_code"})
    resstock_df["bldg_id"] = pd.to_numeric(resstock_df["bldg_id"], errors="coerce").astype("Int64")
    resstock_df["sqm"] = resstock_df["sqft"] * 0.092903

    bldg_df = pd.DataFrame({"bldg_id": pd.array(list(bldg_ids), dtype="Int64")})
    merged_bldg_df = pd.merge(bldg_df, resstock_df, on="bldg_id", how="left", validate="m:1")

    for _, row in merged_bldg_df.iterrows():
        bldg_id = row["bldg_id"]
        sqm     = row["sqm"]

        if pd.isna(bldg_id):
            continue
        if pd.isna(sqm):
            print(f"    Warning: Missing square footage for building {bldg_id}")
            continue

        bldg_path = bldg_consumption_dir / f"{bldg_id}-0.parquet"
        if not bldg_path.exists():
            print(f"    Warning: Missing consumption file for {bldg_id}")
            continue

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

            if OUTDOOR_TEMP_COL not in df_cols:
                print(f"    Warning: Missing outdoor temp for {bldg_id}, skipping")
                continue

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

            has_solar        = SOLAR_COL    in df_cols
            has_wind         = WIND_COL     in df_cols
            has_humidity     = HUMIDITY_COL in df_cols

            if not has_solar and not skip_missing_weather:
                continue

            has_lighting  = LIGHTING_COL  in df_cols
            has_equipment = EQUIPMENT_COL in df_cols
            has_dhw       = DHW_COL       in df_cols
            has_people    = PEOPLE_COL    in df_cols

            base_cols = [
                "datetime",
                OUTDOOR_TEMP_COL,
                ts_config["indoor_temp_key"],
                ts_config["cooling_load_key"],
                ts_config["heating_load_key"],
            ]
            weather_src_cols  = [c for c, f in [(SOLAR_COL, has_solar), (WIND_COL, has_wind), (HUMIDITY_COL, has_humidity)] if f]
            internal_src_cols = [c for c, f in [(LIGHTING_COL, has_lighting), (EQUIPMENT_COL, has_equipment), (DHW_COL, has_dhw), (PEOPLE_COL, has_people)] if f]

            df_bldg = df_bldg[base_cols + weather_src_cols + internal_src_cols].copy()

            df_bldg.rename(columns={
                OUTDOOR_TEMP_COL:                  "outdoor_temp_c",
                ts_config["indoor_temp_key"]:      "indoor_temp_c",
                ts_config["cooling_load_key"]:     "cooling_load_kbtu",
                ts_config["heating_load_key"]:     "heating_load_kbtu",
            }, inplace=True)

            df_bldg["cooling_load_w"] = df_bldg["cooling_load_kbtu"] * 1172.284
            df_bldg["heating_load_w"] = df_bldg["heating_load_kbtu"] * 1172.284

            if has_solar:
                df_bldg["solar_radiation_wm2"] = df_bldg[SOLAR_COL]
                df_bldg["solar_radiation_w"]   = df_bldg[SOLAR_COL] * sqm
                df_bldg.drop(columns=[SOLAR_COL], inplace=True)
            if has_wind:
                df_bldg.rename(columns={WIND_COL: "wind_speed"}, inplace=True)
            if has_humidity:
                df_bldg.rename(columns={HUMIDITY_COL: "outdoor_humidity"}, inplace=True)

            df_bldg["lighting_gain_w"]  = (df_bldg[LIGHTING_COL]  * 4000 * LIGHTING_FRACTION)  if has_lighting  else 0.0
            df_bldg["equipment_gain_w"] = (df_bldg[EQUIPMENT_COL] * 4000 * EQUIPMENT_FRACTION) if has_equipment else 0.0
            df_bldg["dhw_gain_w"]       = (df_bldg[DHW_COL]       * 1172.284 * DHW_FRACTION)   if has_dhw       else 0.0

            if has_lighting:  df_bldg.drop(columns=[LIGHTING_COL],  inplace=True)
            if has_equipment: df_bldg.drop(columns=[EQUIPMENT_COL], inplace=True)
            if has_dhw:       df_bldg.drop(columns=[DHW_COL],       inplace=True)

            if has_people:
                df_bldg.rename(columns={PEOPLE_COL: "people_gain_w"}, inplace=True)
            else:
                df_bldg["people_gain_w"] = 0.0

        except Exception as e:
            print(f"    Error reading {bldg_id}-0.parquet: {e}")
            continue

        all_ts_dict[int(bldg_id)] = df_bldg.copy()

        if save_outputs:
            save_dir = output_dir / "thermal_model" / "timeseries"
            save_dir.mkdir(parents=True, exist_ok=True)
            df_bldg.to_csv(save_dir / f"{bldg_id}_in_ts.csv", index=False)

    return all_ts_dict

# ============================================================
# Function 2: Regression coefficients
# ============================================================
def compute_regression_coefficients(
    all_ts_dict,
    output_dir,
    feature_set="baseline",
    required_columns=None,
):
    """
    Compute constrained least-squares regression (all k >= 0) for each building,
    plus Newey-West HAC standard errors (auto lag via Andrews rule) and a
    joint HAC F-test: H0: k1 = k2 = k3 = 0.

    Output columns per building:
        bldg_id, k1, k2, k3, mse, r_squared, nw_lags,
        se_k1, se_k2, se_k3,
        f_stat, f_pval
    """
    results = []

    if required_columns is None:
        required_columns = {
            "temp_diff_lag":     ["outdoor_temp_c", "indoor_temp_c"],
            "radiation":         "solar_radiation_w",
            "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"],
        }

    for bldg_id, df in all_ts_dict.items():
        if df.shape[0] < 2:
            continue

        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)
        df["indoor_temp_c"]  = (df["indoor_temp_c"] * 2).round() / 2

        df_lag = df.shift(1)

        X_dict = {}
        skip_building = False

        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                if feature_name == "heat_minus_cool_w":
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
                elif feature_name == "internal_gains_minus_cool_w":
                    net = df_lag[col_spec[0]].copy()
                    for extra_col in col_spec[1:-1]:
                        net = net + df_lag[extra_col].fillna(0)
                    net = net - df_lag[col_spec[-1]]
                    X_dict[feature_name] = net
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

        X_vals = X.values.astype(np.float64)
        y_vals = y.values.astype(np.float64)

        # -------------------------------------------
        # Constrained least squares: all k >= 0
        # -------------------------------------------
        n_features = X_vals.shape[1]
        result_lsq = lsq_linear(
            X_vals, y_vals,
            bounds=(np.zeros(n_features), np.inf * np.ones(n_features))
        )
        coefficients = result_lsq.x

        # -------------------------------------------
        # Fit quality
        # -------------------------------------------
        y_pred_delta = X_vals @ coefficients

        T_in_actual = df.loc[valid_idx, "indoor_temp_c"].astype(np.float64)
        T_in_lag    = df_lag.loc[valid_idx, "indoor_temp_c"].astype(np.float64)
        T_in_pred   = T_in_lag + y_pred_delta

        ss_res    = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
        ss_tot    = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
        r_squared = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        mse       = mean_squared_error(T_in_actual.values, T_in_pred.values)

        # -------------------------------------------
        # Newey-West HAC covariance and SE
        # -------------------------------------------
        T_obs     = X_vals.shape[0]
        n_lags    = int(4 * (T_obs / 100) ** (2 / 9))
        residuals = y_vals - y_pred_delta
        V_hac     = newey_west_cov(X_vals, residuals, n_lags=n_lags)
        se        = np.sqrt(np.diag(V_hac).clip(0))

        # -------------------------------------------
        # Joint HAC F-test: H0: k1 = k2 = k3 = 0
        # -------------------------------------------
        q = n_features
        try:
            V_hac_inv = np.linalg.inv(V_hac)
            f_stat = (coefficients @ V_hac_inv @ coefficients) / q
        except np.linalg.LinAlgError:
            V_hac_inv = np.linalg.pinv(V_hac)
            f_stat = (coefficients @ V_hac_inv @ coefficients) / q

        f_pval = 1 - scipy_stats.chi2.cdf(f_stat * q, df=q)

        row = {
            "bldg_id":   bldg_id,
            "k1":        coefficients[0],
            "k2":        coefficients[1],
            "k3":        coefficients[2],
            "mse":       mse,
            "r_squared": r_squared,
            "nw_lags":   n_lags,
            "se_k1":     se[0],
            "se_k2":     se[1],
            "se_k3":     se[2],
            "f_stat":    f_stat,
            "f_pval":    f_pval,
        }

        results.append(row)

    return results


# ============================================================
# Helper: Save regression results as JSON
# ============================================================
def _save_results_json(results_df, out_dir, feature_name):
    """
    Save regression coefficients DataFrame as a JSON file.

    Structure:  { "<bldg_id>": { "k1": ..., "k2": ..., "k3": ..., "f_pval": ..., "r_squared": ..., "mse": ... }, ... }
    """
    out_json = out_dir / f"regression_coeff_{feature_name}.json"
    keep_cols = ["k1", "k2", "k3", "f_pval", "r_squared", "mse"]

    buildings_dict = {}
    for _, row in results_df.iterrows():
        bldg_key = str(int(row["bldg_id"]))
        bldg_record = {}
        for col in keep_cols:
            if col not in row.index:
                continue
            val = row[col]
            if isinstance(val, (np.floating,)):
                bldg_record[col] = None if np.isnan(val) else float(val)
            elif isinstance(val, (np.integer,)):
                bldg_record[col] = int(val)
            else:
                bldg_record[col] = val
        buildings_dict[bldg_key] = bldg_record

    with open(out_json, "w") as f:
        json.dump(buildings_dict, f, indent=2)

    return out_json


# ============================================================
# Worker function
# ============================================================
def _worker(args):
    (batch_idx, batch_ids, input_dir, output_dir,
     ts_config, feature_sets, save_outputs, skip_missing_weather, tmp_dir) = args

    try:
        ts_dict = create_building_timeseries(
            bldg_ids=batch_ids,
            input_dir=input_dir,
            output_dir=output_dir,
            ts_config=ts_config,
            save_outputs=save_outputs,
            skip_missing_weather=skip_missing_weather,
        )

        counts = {}
        for feature_name, feature_cols in feature_sets.items():
            results = compute_regression_coefficients(
                all_ts_dict=ts_dict,
                output_dir=output_dir,
                feature_set=feature_name,
                required_columns=feature_cols,
            )
            if results:
                tmp_file = tmp_dir / f"{feature_name}_batch_{batch_idx:06d}.parquet"
                pd.DataFrame(results).to_parquet(tmp_file, index=False)
            counts[feature_name] = len(results)

        return counts

    except Exception as e:
        return f"ERROR in batch {batch_idx}: {e}"


# ============================================================
# Function 3: Run pipeline
# ============================================================
def run_pipeline(
    bldg_ids, input_dir, output_dir, ts_config, feature_sets,
    save_outputs=False,
    skip_missing_weather=False,
    batch=False,
    batch_size=100,
    n_workers=24,
):
    import concurrent.futures

    bldg_ids_list = list(bldg_ids)
    total = len(bldg_ids_list)

    for feature_name, feature_cols in feature_sets.items():
        out_dir = output_dir / "thermal_model" / feature_name
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "feature_set": feature_name,
            "features": list(feature_cols.keys()),
            "constraints": "All coefficients (k1, k2, k3) constrained to be non-negative (>= 0)",
            "indoor_temp_rounding": "Indoor temperatures rounded to nearest 0.5 degrees C before regression and error calculation",
            "model_form": "T_in(t) = T_in(t-1) + k1*[T_out(t-1)-T_in(t-1)] + k2*Q_rad(t) + k3*[Q_net(t-1)]",
            "standard_errors": "Newey-West HAC, auto lag = int(4*(T/100)^(2/9)) per Andrews (1991)",
            "p_values": "Two-sided, normal approximation (large T)",
            "source": "ercot_substation_nrel_map.parquet (unique bldg_ids)",
            "weather_source": "Embedded in consumption parquet files",
            "weather_columns": {
                "solar_radiation": "out.weather.direct_normal_solar_radiation..watt_per_m2",
                "wind_speed":      "out.weather.wind_speed..meter_per_second",
                "humidity":        "out.outdoor_air_relative_humidity..percentage",
            },
            "output_formats": ["csv", "parquet", "json"],
            "batched":    batch,
            "batch_size": batch_size if batch else None,
            "n_workers":  n_workers  if batch else None,
            "timestamp":  pd.Timestamp.now().isoformat(),
        }
        metadata_file = out_dir / f"regression_coeff_{feature_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    if batch:
        batches = [bldg_ids_list[i:i + batch_size] for i in range(0, total, batch_size)]
        n_batches = len(batches)
        tmp_dir = output_dir / "thermal_model" / "_tmp_batch_results"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running in concurrent mode: {total} buildings, "
              f"{n_batches} batches of {batch_size}, {n_workers} workers")

        worker_args = [
            (batch_idx, batch_ids, input_dir, output_dir,
             ts_config, feature_sets, save_outputs, skip_missing_weather, tmp_dir)
            for batch_idx, batch_ids in enumerate(batches)
        ]

        t_start_workers = time.time()
        completed, errors = 0, []

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker, args): args[0] for args in worker_args}
            for future in concurrent.futures.as_completed(futures):
                batch_idx = futures[future]
                result    = future.result()
                completed += 1

                if isinstance(result, str):
                    errors.append(result)
                    print(f"  [{completed}/{n_batches}] Batch {batch_idx} FAILED: {result}")
                else:
                    counts_str = " | ".join(f"{fs}: {n}" for fs, n in result.items())
                    print(f"  [{completed}/{n_batches}] Batch {batch_idx} done "
                          f"({time.time()-t_start_workers:.0f}s elapsed) | {counts_str}")

        if errors:
            print(f"\n{len(errors)} batch(es) failed:")
            for e in errors:
                print(f"  {e}")

        print("\nMerging batch results...")
        for feature_name in feature_sets:
            tmp_files = sorted(tmp_dir.glob(f"{feature_name}_batch_*.parquet"))
            if not tmp_files:
                print(f"  No results found for {feature_name}")
                continue

            merged_df = pd.concat([pd.read_parquet(f) for f in tmp_files], ignore_index=True)
            out_dir     = output_dir / "thermal_model" / feature_name
            out_csv     = out_dir / f"regression_coeff_{feature_name}.csv"
            out_parquet = out_dir / f"regression_coeff_{feature_name}.parquet"
            merged_df.to_csv(out_csv, index=False)
            merged_df.to_parquet(out_parquet, index=False)

            # --- Save as JSON ---
            out_json = _save_results_json(merged_df, out_dir, feature_name)
            print(f"  Saved {feature_name}: {len(merged_df)} buildings -> {out_csv}")
            print(f"  Saved {feature_name} JSON: {out_json}")

            for f in tmp_files:
                f.unlink()

        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    else:
        print(f"Running in single-pass mode: {total} buildings")

        ts_dict = create_building_timeseries(
            bldg_ids=bldg_ids_list,
            input_dir=input_dir,
            output_dir=output_dir,
            ts_config=ts_config,
            save_outputs=save_outputs,
            skip_missing_weather=skip_missing_weather,
        )

        for feature_name, feature_cols in feature_sets.items():
            results    = compute_regression_coefficients(
                all_ts_dict=ts_dict,
                output_dir=output_dir,
                feature_set=feature_name,
                required_columns=feature_cols,
            )
            results_df  = pd.DataFrame(results)
            out_dir     = output_dir / "thermal_model" / feature_name
            out_csv     = out_dir / f"regression_coeff_{feature_name}.csv"
            out_parquet = out_dir / f"regression_coeff_{feature_name}.parquet"
            results_df.to_csv(out_csv, index=False)
            results_df.to_parquet(out_parquet, index=False)

            # --- Save as JSON ---
            out_json = _save_results_json(results_df, out_dir, feature_name)
            print(f"\nSaved {feature_name} coefficients ({len(results_df)} buildings): {out_csv}")
            print(f"Saved {feature_name} JSON: {out_json}")


# ============================================================
# Function 4: Plot lowest R-squared buildings
# ============================================================
def plot_lowest_r2_buildings(all_ts_dict, output_dir, feature_set="baseline"):
    coeff_file = output_dir / "thermal_model" / feature_set / f"regression_coeff_{feature_set}.csv"
    if not coeff_file.exists():
        print(f"Regression coefficients file not found: {coeff_file}")
        return

    coeffs_df = pd.read_csv(coeff_file)

    metadata_file = output_dir / "thermal_model" / feature_set / f"regression_coeff_{feature_set}_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            feature_names = json.load(f).get("features", [])
    else:
        feature_names = ["temp_diff_lag", "radiation", "heat_minus_cool_w"]

    r_squared_list = []

    for bldg_id, df in all_ts_dict.items():
        df["heating_load_w"] = df["heating_load_w"].fillna(0)
        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["indoor_temp_c"]  = (df["indoor_temp_c"] * 2).round() / 2
        df_lag = df.shift(1)

        X_dict = {}
        skip_building = False

        for feature_name in feature_names:
            if feature_name == "internal_gains_minus_cool_w":
                net = df_lag["heating_load_w"].copy()
                for col in ["lighting_gain_w", "equipment_gain_w", "dhw_gain_w", "people_gain_w"]:
                    if col in df_lag.columns:
                        net = net + df_lag[col].fillna(0)
                net = net - df_lag["cooling_load_w"]
                X_dict[feature_name] = net
            elif feature_name == "heat_minus_cool_w":
                X_dict[feature_name] = df_lag["heating_load_w"] - df_lag["cooling_load_w"]
            elif feature_name == "temp_diff_lag":
                if "outdoor_temp_c" not in df_lag.columns or "indoor_temp_c" not in df_lag.columns:
                    skip_building = True; break
                X_dict[feature_name] = df_lag["outdoor_temp_c"] - df_lag["indoor_temp_c"]
            elif feature_name == "radiation":
                if "solar_radiation_w" not in df.columns:
                    skip_building = True; break
                X_dict[feature_name] = df["solar_radiation_w"]
            elif "lag" in feature_name:
                base_col = feature_name.replace("_lag", "")
                if base_col not in df.columns:
                    skip_building = True; break
                X_dict[feature_name] = df_lag[base_col]
            else:
                if feature_name not in df.columns:
                    skip_building = True; break
                X_dict[feature_name] = df[feature_name]

        if skip_building:
            continue

        X = pd.DataFrame(X_dict)
        coeff_row = coeffs_df[coeffs_df["bldg_id"] == int(bldg_id)]
        if coeff_row.empty:
            continue

        k_values     = [coeff_row[f"k{i}"].values[0] for i in range(1, len(feature_names) + 1) if f"k{i}" in coeff_row.columns]
        delta_T_pred = sum(k_values[i] * X[fname] for i, fname in enumerate(feature_names) if i < len(k_values))
        y_pred       = df_lag["indoor_temp_c"] + delta_T_pred

        valid_idx      = y_pred.dropna().index
        T_in_actual    = df.loc[valid_idx, "indoor_temp_c"].values
        T_in_pred_vals = y_pred.loc[valid_idx].values

        ss_res    = np.sum((T_in_actual - T_in_pred_vals) ** 2)
        ss_tot    = np.sum((T_in_actual - T_in_actual.mean()) ** 2)
        r_squared = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        r_squared_list.append((bldg_id, r_squared, df["datetime"], df["indoor_temp_c"], y_pred))

    if not r_squared_list:
        print("No buildings to plot.")
        return

    bottom3      = sorted([e for e in r_squared_list if not np.isnan(e[1])], key=lambda x: x[1])[:3]
    plots_folder = output_dir / "thermal_model" / feature_set / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)

    plt.ion()
    for bldg_id, r_squared, dt, y_true, y_pred in bottom3:
        plt.figure(figsize=(12, 5))
        plt.plot(dt, y_true, label="Actual (rounded)")
        plt.plot(dt, y_pred, label="Reconstructed")
        plt.title(f"{feature_set} | Building {bldg_id} | R²={r_squared:.4f}")
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

    use_batching = True
    batch_size   = 100
    n_workers    = 20
    save_flag    = False

    current_path = Path(__file__).resolve()
    parent_dir   = current_path.parents[1]
    input_dir    = parent_dir / "in"
    output_dir   = parent_dir / "out"

    ts_config = {
        "indoor_temp_key":  "out.indoor_temperature.conditioned_space..c",
        "cooling_load_key": "out.load.cooling.energy_delivered..kbtu",
        "heating_load_key": "out.load.heating.energy_delivered..kbtu",
    }

    # =========================================================
    # FEATURE SETS
    # =========================================================
    FEATURE_SETS = {
        # "baseline": {
        #     "temp_diff_lag":     ["outdoor_temp_c", "indoor_temp_c"],
        #     "radiation":         "solar_radiation_w",
        #     "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"],
        # },
        "baseline_internal_gains": {
            "temp_diff_lag":               ["outdoor_temp_c", "indoor_temp_c"],
            "radiation":                   "solar_radiation_w",
            "internal_gains_minus_cool_w": [
                "heating_load_w",
                "lighting_gain_w",
                "equipment_gain_w",
                "dhw_gain_w",
                "people_gain_w",
                "cooling_load_w",   # last entry is subtracted
            ],
        },
    }

    substation_map_file = output_dir / "ercot_substation_nrel_map.parquet"
    if not substation_map_file.exists():
        raise FileNotFoundError(f"Missing substation map: {substation_map_file}")

    substation_df = pd.read_parquet(substation_map_file, engine="fastparquet")
    bldg_ids = pd.to_numeric(
        substation_df["bldg_id"], errors="coerce"
    ).dropna().astype("Int64").unique()

    print(f"\n=== Found {len(bldg_ids)} unique building IDs in substation map ===")
    print(f"=== Feature sets: {', '.join(FEATURE_SETS.keys())} ===")
    print(f"=== Concurrent: {use_batching} | Batch size: {batch_size} | Workers: {n_workers if use_batching else 'N/A'} ===\n")

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
        n_workers=n_workers,
    )

    t_elapsed = time.time() - t_start
    print(f"\nDone! Elapsed time: {int(t_elapsed//3600)}h {int((t_elapsed%3600)//60)}m {int(t_elapsed%60)}s\n")