import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


def diagnose_building_fit(
    bldg_id,
    zone,
    input_dir,
    output_dir,
    ts_config,
    feature_set="baseline",
    required_columns=None
):
    """
    Diagnostic tool: Load and reconstruct a single building's timeseries,
    fit OLS (unconstrained), and plot actual vs predicted indoor temperature
    along with all input features.
    
    Args:
        bldg_id: Building ID to diagnose
        zone: Zone identifier
        input_dir: Input directory
        output_dir: Output directory
        ts_config: Configuration dict with column name mappings
        feature_set: Feature set name (for plot title)
        required_columns: Dict mapping feature names to dataframe columns
    """
    
    if required_columns is None:
        required_columns = {
            "temp_diff_lag": ["outdoor_temp_c", "indoor_temp_c"],
            "radiation": "solar_radiation_w",
            "heat_minus_cool_w": ["heating_load_w", "cooling_load_w"]
        }
    
    # ============================================================
    # RECONSTRUCT BUILDING TIMESERIES
    # ============================================================
    
    bldg_consumption_dir = output_dir / "consumption_files"
    weather_dir = input_dir / "TX_weather_data"
    
    # Load building parquet
    bldg_path = bldg_consumption_dir / f"{bldg_id}-0.parquet"
    if not bldg_path.exists():
        print(f"❌ Missing consumption file: {bldg_path}")
        return
    
    try:
        df = pd.read_parquet(bldg_path, engine="fastparquet")
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
        else:
            print(f"❌ {bldg_path.name} missing 'timestamp'")
            return
        
        # Check for required columns
        if ts_config["indoor_temp_key"] not in df.columns:
            print(f"❌ Missing indoor temp column: {ts_config['indoor_temp_key']}")
            return
        
        # Replace missing load columns with 0
        if (
            ts_config["cooling_load_key"] not in df.columns or
            df[ts_config["cooling_load_key"]].isna().all()
        ):
            df[ts_config["cooling_load_key"]] = 0
        
        if (
            ts_config["heating_load_key"] not in df.columns or
            df[ts_config["heating_load_key"]].isna().all()
        ):
            df[ts_config["heating_load_key"]] = 0
        
        df = df[
            [
                "datetime",
                ts_config["indoor_temp_key"],
                ts_config["cooling_load_key"],
                ts_config["heating_load_key"],
            ]
        ].copy()
        
        df.rename(
            columns={
                ts_config["indoor_temp_key"]: "indoor_temp_c",
                ts_config["cooling_load_key"]: "cooling_load_kbtu",
                ts_config["heating_load_key"]: "heating_load_kbtu",
            },
            inplace=True,
        )
        
        df["cooling_load_w"] = df["cooling_load_kbtu"] * 1172.284
        df["heating_load_w"] = df["heating_load_kbtu"] * 1172.284
        
    except Exception as e:
        print(f"❌ Error reading {bldg_id}-0.parquet: {e}")
        return
    
    # Load ResStock metadata for square footage
    resstock_file = input_dir / "TX_upgrade0.parquet"
    if not resstock_file.exists():
        print(f"❌ Missing ResStock parquet file: {resstock_file}")
        return
    
    resstock_df = pd.read_parquet(resstock_file, engine="fastparquet")
    
    if "in.sqft..ft2" not in resstock_df.columns:
        print(f"❌ Missing 'in.sqft..ft2' column in ResStock data")
        return
    
    resstock_df = resstock_df[["bldg_id", "in.sqft..ft2", "in.county"]].copy()
    resstock_df = resstock_df.rename(columns={
        "in.sqft..ft2": "sqft",
        "in.county": "county_code"
    })
    
    resstock_df["bldg_id"] = pd.to_numeric(
        resstock_df["bldg_id"], errors="coerce"
    ).astype("Int64")
    
    # Convert square feet to square meters
    resstock_df["sqm"] = resstock_df["sqft"] * 0.092903
    
    # Get this building's metadata
    bldg_meta = resstock_df[resstock_df["bldg_id"] == int(bldg_id)]
    
    if bldg_meta.empty:
        print(f"❌ Building {bldg_id} not found in ResStock metadata")
        return
    
    county_code = bldg_meta.iloc[0]["county_code"]
    sqm = bldg_meta.iloc[0]["sqm"]
    
    if pd.isna(sqm):
        print(f"❌ Missing square footage for building {bldg_id}")
        return
    
    # Load weather data
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
                    print(f"❌ Missing required weather columns (outdoor temp or radiation)")
                    return
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
                    
                    # Scale radiation by building area
                    df_weather["solar_radiation_w"] = df_weather["solar_radiation_wm2"] * sqm
                    
                    # Select columns (humidity and wind optional)
                    cols_to_keep = ["datetime", "solar_radiation_w", "outdoor_temp_c"]
                    if "outdoor_humidity" in df_weather.columns:
                        cols_to_keep.append("outdoor_humidity")
                    if "wind_speed" in df_weather.columns:
                        cols_to_keep.append("wind_speed")
                    
                    df_weather = df_weather[cols_to_keep]
                    
                    df_weather.set_index("datetime", inplace=True)
                    df_weather = df_weather.reindex(df["datetime"]).interpolate("linear")
                    df_weather.reset_index(inplace=True)
            except Exception as e:
                print(f"❌ Error loading weather data: {e}")
                return
        else:
            print(f"❌ Weather file not found: {weather_path}")
            return
    else:
        print(f"❌ No county code for building {bldg_id}")
        return
    
    # Merge building and weather data
    if df_weather is not None:
        df = df.merge(df_weather, on="datetime", how="left")
    else:
        print(f"❌ No weather data available")
        return
    
    print(f"✅ Loaded building {bldg_id}: {df.shape[0]} timesteps, sqm={sqm:.1f}")
    
    # ============================================================
    # FIT OLS MODEL (UNCONSTRAINED)
    # ============================================================
    
    df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
    df["heating_load_w"] = df["heating_load_w"].fillna(0)
    
    df_lag = df.shift(1)
    
    # Build X matrix
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
        print(f"❌ Missing required columns for building {bldg_id}")
        return
    
    X = pd.DataFrame(X_dict)
    y = df["indoor_temp_c"] - df_lag["indoor_temp_c"]
    
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    if X.shape[0] < 2:
        print(f"❌ Insufficient data after cleaning (only {X.shape[0]} rows)")
        return
    
    # Fit unconstrained OLS
    model = sm.OLS(y, X).fit()
    ΔT_pred = model.predict(X)
    
    # Reconstruct T_in(t)
    T_in_actual = df.loc[valid_idx, "indoor_temp_c"]
    T_in_lag = df_lag.loc[valid_idx, "indoor_temp_c"]
    T_in_pred = T_in_lag + ΔT_pred
    
    # Compute metrics
    mse = mean_squared_error(y, ΔT_pred)
    ss_res = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
    ss_tot = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    # ============================================================
    # PRINT DIAGNOSTIC INFO
    # ============================================================
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT: Building {bldg_id} (Zone {zone})")
    print(f"{'='*60}")
    print(f"Feature Set: {feature_set}")
    print(f"Data points: {X.shape[0]}")
    print(f"Building area: {sqm:.1f} m²")
    print(f"\nCOEFFICIENTS:")
    for i, (feature_name, coeff) in enumerate(zip(X.columns, model.params.values), start=1):
        p_val = model.pvalues.values[i-1]
        sign = "✓" if coeff > 0 else "✗"
        print(f"  k{i} ({feature_name:20s}): {coeff:12.6e}  (p={p_val:.4f}) {sign}")
    
    print(f"\nMETRICS:")
    print(f"  MSE (on ΔT):        {mse:.6f}")
    print(f"  R² (on T_in):       {r_squared:.6f}")
    print(f"  Mean T_in:          {T_in_actual.mean():.2f}°C")
    print(f"  Std T_in:           {T_in_actual.std():.2f}°C")
    print(f"  ΔT range:           [{y.min():.3f}, {y.max():.3f}]")
    print(f"  ΔT std:             {y.std():.3f}")
    
    print(f"\nDATA QUALITY CHECK:")
    print(f"  Indoor temp range: [{df['indoor_temp_c'].min():.6f}, {df['indoor_temp_c'].max():.6f}]")
    print(f"  Indoor temp std: {df['indoor_temp_c'].std():.6f}")
    print(f"  Outdoor temp range: [{df['outdoor_temp_c'].min():.2f}, {df['outdoor_temp_c'].max():.2f}]")
    print(f"  Cooling load sum: {df['cooling_load_w'].sum():.2f} W")
    print(f"  Heating load sum: {df['heating_load_w'].sum():.2f} W")
    print(f"  Solar radiation mean: {df['solar_radiation_w'].mean():.2f} W")
    
    if r_squared < 0:
        print(f"\n⚠️  WARNING: Negative R² indicates model is worse than predicting the mean!")
    
    negative_coeffs = [f"k{i+1}" for i, c in enumerate(model.params.values) if c < 0]
    if negative_coeffs:
        print(f"\n⚠️  WARNING: Negative coefficients found: {', '.join(negative_coeffs)}")
        print(f"    All coefficients should be positive for physical validity.")
    
    if df["indoor_temp_c"].std() < 0.1:
        print(f"\n⚠️  WARNING: Indoor temp variance very low ({df['indoor_temp_c'].std():.6f}°C)")
        print(f"    This building may have bad sensor data or perfect temperature control.")
    
    print(f"{'='*60}\n")
    
    # ============================================================
    # PLOT
    # ============================================================
    
    datetime_vals = df.loc[valid_idx, "datetime"]
    
    # Create plots directory
    plots_folder = output_dir / zone / "diagnostic_plots"
    plots_folder.mkdir(exist_ok=True)
    
    # ============================================================
    # FIGURE 1: Temperature Prediction vs Measured
    # ============================================================
    
    fig1 = plt.figure(figsize=(16, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(datetime_vals, T_in_actual, label="Measured T_in", alpha=0.8, linewidth=1.5)
    ax1.plot(datetime_vals, T_in_pred, label="Predicted T_in", alpha=0.8, linewidth=1.5)
    ax1.set_ylabel("Indoor Temp (°C)", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_title(f"Building {bldg_id} | Zone {zone} | Feature Set: {feature_set}\n"
                  f"R² = {r_squared:.4f} | MSE = {mse:.6f}", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save temperature prediction plot
    plot_file_temp = plots_folder / f"{zone}_{bldg_id}_temperature_prediction.png"
    plt.savefig(plot_file_temp, dpi=300, bbox_inches='tight')
    print(f"📊 Temperature plot saved: {plot_file_temp}")
    
    plt.show()
    
    # ============================================================
    # FIGURE 2: Diagnostic Plots (Features and Residuals)
    # ============================================================
    
    fig2 = plt.figure(figsize=(16, 10))
    gs = fig2.add_gridspec(4, 1, hspace=0.3)
    
    # Plot 1: Temperature difference (T_out - T_in)
    ax2 = fig2.add_subplot(gs[0, 0])
    temp_diff = df.loc[valid_idx, "outdoor_temp_c"] - df.loc[valid_idx, "indoor_temp_c"]
    ax2.plot(datetime_vals, temp_diff, color='orange', alpha=0.8, linewidth=1)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    ax2.set_ylabel("Temp Diff (°C)\n(T_out - T_in)")
    ax2.set_title(f"Building {bldg_id} | Zone {zone} | Diagnostic Features", fontsize=14)
    ax2.grid(alpha=0.3)
    
    # Plot 2: Solar radiation
    ax3 = fig2.add_subplot(gs[1, 0])
    ax3.plot(datetime_vals, df.loc[valid_idx, "solar_radiation_w"], color='gold', alpha=0.8, linewidth=1)
    ax3.set_ylabel("Solar Radiation (W)")
    ax3.grid(alpha=0.3)
    
    # Plot 3: Heating - Cooling load
    ax4 = fig2.add_subplot(gs[2, 0])
    heat_minus_cool = df.loc[valid_idx, "heating_load_w"] - df.loc[valid_idx, "cooling_load_w"]
    ax4.plot(datetime_vals, heat_minus_cool, color='red', alpha=0.8, linewidth=1)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax4.set_ylabel("Heat - Cool (W)")
    ax4.grid(alpha=0.3)
    
    # Plot 4: Residuals
    ax5 = fig2.add_subplot(gs[3, 0])
    residuals = T_in_actual.values - T_in_pred.values
    ax5.scatter(T_in_pred.values, residuals, alpha=0.5, s=10)
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax5.set_xlabel("Predicted T_in (°C)")
    ax5.set_ylabel("Residuals (°C)")
    ax5.set_title("Residual Plot")
    ax5.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save diagnostic plots
    plot_file_diag = plots_folder / f"{zone}_{bldg_id}_diagnostics.png"
    plt.savefig(plot_file_diag, dpi=300, bbox_inches='tight')
    print(f"📊 Diagnostics plot saved: {plot_file_diag}")
    
    plt.show()
    
    return {
        "bldg_id": bldg_id,
        "r_squared": r_squared,
        "mse": mse,
        "coefficients": dict(zip([f"k{i+1}" for i in range(len(model.params))], model.params.values)),
        "p_values": dict(zip([f"p_k{i+1}" for i in range(len(model.pvalues))], model.pvalues.values))
    }


# Example usage
if __name__ == "__main__":
    
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
    
    # Diagnose a specific building
    # zone = "P100U"  # Change this
    # bldg_id = 42152  # Change this to your building ID
    zone = "P100U"
    bldg_id = 157349


    result = diagnose_building_fit(
        bldg_id=bldg_id,
        zone=zone,
        input_dir=input_dir,
        output_dir=output_dir,
        ts_config=ts_config,
        feature_set="baseline"
    )