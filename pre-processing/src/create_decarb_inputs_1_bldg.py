#!/usr/bin/env python3
"""
Building Energy Data Generator

Generate timestamp and tariff columns
"""

import pandas as pd
from datetime import datetime, time
from pathlib import Path


def generate_building_data(bldg_id, timestep='hourly', year=2018, 
                          tariff_type='flat', tariff_value=0.22,
                          peak_start='16:00', peak_end='19:00', peak_price=0.28,
                          capacity_charge=False, capacity_peak_start='19:00', 
                          capacity_peak_end='21:00', capacity_charge_value=0.5,
                          co2_cost=0.5, gas_cost=0.051, liquid_cost=0.0739):
    """
    Generate building energy dataset
    
    Args:
        bldg_id: Building ID number (e.g., 77)
        timestep: 'hourly' or '15min'
        year: Year for the dataset (default: 2018)
        tariff_type: 'flat' or 'tou' (time-of-use)
        tariff_value: Single tariff value if flat (default: 0.22)
        peak_start: Peak period start time as 'HH:MM' for TOU (default: '16:00')
        peak_end: Peak period end time as 'HH:MM' for TOU (default: '19:00')
        peak_price: Peak tariff if tou (default: 0.28)
        capacity_charge: Whether to include capacity charges (default: False)
        capacity_peak_start: Capacity peak period start time as 'HH:MM' (default: '19:00')
        capacity_peak_end: Capacity peak period end time as 'HH:MM' (default: '21:00')
        capacity_charge_value: Capacity charge rate (default: 0.5)
        co2_cost: CO2 cost/intensity (default: 0.5)
        gas_cost: Cost of gaseous fuel (default: 0.051)
        liquid_cost: Cost of liquid fuel (default: 0.0739)
    
    Returns:
        DataFrame with timestamp index and cost columns
    """
    
    def resample_if_needed(series, method='sum'):
        """
        Helper function to resample a series from 15-min to hourly if needed
        
        Args:
            series: pandas Series with datetime index at 15-min resolution
            method: 'sum' for energy values, 'mean' for rates/schedules/weather
        
        Returns:
            Resampled series if timestep is hourly, otherwise original series
        """
        if timestep == 'hourly' and len(series) == 35040:
            if method == 'sum':
                resampled = series.resample('h').sum()
            elif method == 'mean':
                resampled = series.resample('h').mean()
            else:
                raise ValueError(f"Unknown resampling method: {method}")
            
            # Ensure exactly 8760 rows
            if len(resampled) > 8760:
                resampled = resampled.iloc[:8760]
            return resampled
        else:
            return series
    
    # Step 0: Define parent path
    parent_path = Path(__file__).parents[1]
    
    # Load consumption data from file
    # Try parquet first, fall back to CSV
    consumption_file_parquet = parent_path / "out" / "consumption_files" / f"{bldg_id}-0.parquet"
    consumption_file_csv = parent_path / "out" / "consumption_files" / f"{bldg_id}-0.csv"
    
    print(f"Loading consumption data for building {bldg_id}...")
    
    if consumption_file_parquet.exists():
        print(f"File path: {consumption_file_parquet}")
        # Load parquet file - try fastparquet, fall back to pyarrow
        try:
            consumption_df = pd.read_parquet(consumption_file_parquet, engine='fastparquet')
        except:
            consumption_df = pd.read_parquet(consumption_file_parquet, engine='pyarrow')
    elif consumption_file_csv.exists():
        print(f"File path: {consumption_file_csv}")
        consumption_df = pd.read_csv(consumption_file_csv, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(f"Consumption file not found: {consumption_file_parquet} or {consumption_file_csv}")
    
    print(f"Loaded consumption data: {consumption_df.shape}")
    
    # Ensure consumption data has datetime index
    if not isinstance(consumption_df.index, pd.DatetimeIndex):
        # If index is not datetime, assume it starts at Jan 1 of the year with 15-min frequency
        if len(consumption_df) == 35040:
            consumption_df.index = pd.date_range(start=datetime(year, 1, 1, 0, 0), periods=35040, freq='15min')
        elif len(consumption_df) == 8760:
            consumption_df.index = pd.date_range(start=datetime(year, 1, 1, 0, 0), periods=8760, freq='h')
        else:
            raise ValueError(f"Unexpected consumption data length: {len(consumption_df)}")
        print(f"Set datetime index for consumption data")
    else:
        # Shift timestamps back by 15 minutes so data starts at 0:00 instead of 0:15
        consumption_df.index = consumption_df.index - pd.Timedelta(minutes=15)
        print(f"Shifted consumption timestamps back by 15 minutes")
        print(f"  New time range: {consumption_df.index[0]} to {consumption_df.index[-1]}")
    
    # Step 0b: Load setpoint timeseries data
    setpoint_file_parquet = parent_path / "out" / "setpoint_timeseries" / f"setpoint_{bldg_id}.parquet"
    setpoint_file_csv = parent_path / "out" / "setpoint_timeseries" / f"setpoint_{bldg_id}.csv"
    
    print(f"\nLoading setpoint timeseries for building {bldg_id}...")
    
    if setpoint_file_parquet.exists():
        print(f"File path: {setpoint_file_parquet}")
        # Load parquet file - try fastparquet, fall back to pyarrow
        try:
            setpoint_df = pd.read_parquet(setpoint_file_parquet, engine='fastparquet')
        except:
            setpoint_df = pd.read_parquet(setpoint_file_parquet, engine='pyarrow')
    elif setpoint_file_csv.exists():
        print(f"File path: {setpoint_file_csv}")
        setpoint_df = pd.read_csv(setpoint_file_csv, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(f"Setpoint file not found: {setpoint_file_parquet} or {setpoint_file_csv}")
    
    print(f"Loaded setpoint timeseries: {setpoint_df.shape}")
    
    # Ensure setpoint data has datetime index
    if not isinstance(setpoint_df.index, pd.DatetimeIndex):
        # If index is not datetime, assume it starts at Jan 1 of the year with 15-min frequency
        if len(setpoint_df) == 35040:
            setpoint_df.index = pd.date_range(start=datetime(year, 1, 1, 0, 0), periods=35040, freq='15min')
        elif len(setpoint_df) == 8760:
            setpoint_df.index = pd.date_range(start=datetime(year, 1, 1, 0, 0), periods=8760, freq='h')
        else:
            raise ValueError(f"Unexpected setpoint data length: {len(setpoint_df)}")
        print(f"Set datetime index for setpoint data")
    else:
        # Shift timestamps back by 15 minutes so data starts at 0:00 instead of 0:15
        setpoint_df.index = setpoint_df.index - pd.Timedelta(minutes=15)
        print(f"Shifted setpoint timestamps back by 15 minutes")
        print(f"  New time range: {setpoint_df.index[0]} to {setpoint_df.index[-1]}")
    
    # Step 0c: Load Texas metadata for HVAC availability and fuel types
    metadata_file_parquet = parent_path / "in" / "TX_upgrade0.parquet"
    metadata_file_csv = parent_path / "in" / "TX_upgrade0.csv"
    
    print(f"\nLoading Texas metadata...")
    
    if metadata_file_parquet.exists():
        print(f"File path: {metadata_file_parquet}")
        # Load parquet file - try fastparquet, fall back to pyarrow
        try:
            metadata_df = pd.read_parquet(metadata_file_parquet, engine='fastparquet')
        except:
            metadata_df = pd.read_parquet(metadata_file_parquet, engine='pyarrow')
    elif metadata_file_csv.exists():
        print(f"File path: {metadata_file_csv}")
        metadata_df = pd.read_csv(metadata_file_csv)
    else:
        raise FileNotFoundError(f"Metadata file not found: {metadata_file_parquet} or {metadata_file_csv}")
    
    print(f"Loaded metadata: {metadata_df.shape}")
    
    # Get HVAC variables for this building
    hvac_columns = [
        'in.heating_unavailable_period',
        'in.cooling_unavailable_period'
    ]
    
    # Find the row for this building (assuming there's a building ID column)
    # Common column names: 'bldg_id', 'building_id', 'id', or index
    if 'bldg_id' in metadata_df.columns:
        building_row = metadata_df[metadata_df['bldg_id'] == bldg_id]
    elif 'building_id' in metadata_df.columns:
        building_row = metadata_df[metadata_df['building_id'] == bldg_id]
    elif str(bldg_id) in metadata_df.index.astype(str):
        building_row = metadata_df.loc[str(bldg_id):str(bldg_id)]
    elif bldg_id in metadata_df.index:
        building_row = metadata_df.loc[bldg_id:bldg_id]
    else:
        raise ValueError(f"Building {bldg_id} not found in metadata file")
    
    if len(building_row) == 0:
        raise ValueError(f"Building {bldg_id} not found in metadata file")
    
    # Extract HVAC values
    hvac_data = {}
    for col in hvac_columns:
        if col in building_row.columns:
            hvac_data[col] = building_row[col].iloc[0]
        else:
            print(f"Warning: Column {col} not found in metadata")
            hvac_data[col] = None
    
    print(f"\nHVAC availability data for building {bldg_id}:")
    for key, value in hvac_data.items():
        print(f"  {key}: {value}")
    
    # Don't resample setpoint data automatically - keep at original resolution
    # We'll handle resampling for each variable individually
    print(f"Setpoint data shape: {setpoint_df.shape}")
    
    # Don't resample consumption data automatically - keep at original resolution
    # We'll handle resampling for each variable individually based on its meaning
    print(f"Consumption data shape: {consumption_df.shape}")
    
    # Step 1: Generate timestamps
    if timestep == 'hourly':
        n_steps = 8760  # 365 days * 24 hours
        freq = 'h'
    elif timestep == '15min':
        n_steps = 35040  # 8760 * 4
        freq = '15min'
    else:
        raise ValueError("timestep must be 'hourly' or '15min'")
    
    # Create datetime index starting at Jan 1
    start_date = datetime(year, 1, 1, 0, 0)
    timestamps = pd.date_range(start=start_date, periods=n_steps, freq=freq)
    
    # Handle DST: Skip the non-existent hour (spring forward) 
    # In 2018 America/New_York: March 11, 2:00 AM doesn't exist (jumps to 3:00 AM)
    if timestep == 'hourly':
        # Find if March 11 at 2 AM exists in our timestamps
        dst_spring = datetime(year, 3, 11, 2, 0)
        if dst_spring in timestamps:
            # Skip this hour by moving it and all subsequent hours forward by 1 hour
            mask = timestamps >= dst_spring
            timestamps = pd.DatetimeIndex([ts if not mask[i] else ts + pd.Timedelta(hours=1) 
                                           for i, ts in enumerate(timestamps)])
    elif timestep == '15min':
        # For 15-min data, skip the 4 periods from 2:00 to 2:45 AM on March 11
        dst_spring_start = datetime(year, 3, 11, 2, 0)
        if dst_spring_start in timestamps:
            mask = timestamps >= dst_spring_start
            timestamps = pd.DatetimeIndex([ts if not mask[i] else ts + pd.Timedelta(hours=1) 
                                           for i, ts in enumerate(timestamps)])
    
    # Create dataframe with timestamp as index
    df = pd.DataFrame(index=timestamps)
    df.index.name = 'pP'
    
    # Step 2: Generate tariff column (pQcostBuy)
    if tariff_type == 'flat':
        # Simple flat rate
        df['pQcostBuy'] = tariff_value
        print(f"Tariff: Flat rate of ${tariff_value:.2f}/kWh")
        
    elif tariff_type == 'tou':
        # Time-of-use pricing
        off_peak_price = peak_price / 2.0
        
        # Parse peak period times
        peak_start_time = datetime.strptime(peak_start, '%H:%M').time()
        peak_end_time = datetime.strptime(peak_end, '%H:%M').time()
        
        # Get the time component of each timestamp
        timestamp_times = df.index.time
        
        # Determine if each timestamp is in peak period
        # Handle case where peak period crosses midnight
        if peak_start_time < peak_end_time:
            # Normal case: peak within same day (e.g., 16:00 to 19:00)
            is_peak = (timestamp_times >= peak_start_time) & (timestamp_times < peak_end_time)
        else:
            # Peak period crosses midnight (e.g., 22:00 to 06:00)
            is_peak = (timestamp_times >= peak_start_time) | (timestamp_times < peak_end_time)
        
        # Set tariff based on peak/off-peak
        df['pQcostBuy'] = off_peak_price
        df.loc[is_peak, 'pQcostBuy'] = peak_price
        
        print(f"Tariff: Time-of-Use")
        print(f"  Peak period: {peak_start} to {peak_end}")
        print(f"  Peak price: ${peak_price:.2f}/kWh")
        print(f"  Off-peak price: ${off_peak_price:.2f}/kWh")
        
    else:
        raise ValueError("tariff_type must be 'flat' or 'tou'")
    
    # Step 2b: Generate sellback tariff column (pQcostSell)
    df['pQcostSell'] = 0.18  # Sellback price (set to constant for now)
    print(f"Sellback tariff: $0.18/kWh")
    
    # Step 3: Generate capacity charge columns (pQmx and pQmxCost)
    if capacity_charge:
        # Parse capacity peak period times
        cap_start_time = datetime.strptime(capacity_peak_start, '%H:%M').time()
        cap_end_time = datetime.strptime(capacity_peak_end, '%H:%M').time()
        
        # Get the time component of each timestamp
        timestamp_times = df.index.time
        
        # Determine if each timestamp is in capacity peak period
        if cap_start_time < cap_end_time:
            # Normal case: peak within same day (e.g., 19:00 to 21:00)
            is_capacity_peak = (timestamp_times >= cap_start_time) & (timestamp_times < cap_end_time)
        else:
            # Peak period crosses midnight (e.g., 22:00 to 02:00)
            is_capacity_peak = (timestamp_times >= cap_start_time) | (timestamp_times < cap_end_time)
        
        # pQmx: Binary flag (1 during capacity peak, 0 otherwise)
        df['pQmx'] = 0
        df.loc[is_capacity_peak, 'pQmx'] = 1
        
        # pQmxCost: Capacity charge value during peak, 0 otherwise
        df['pQmxCost'] = 0.0
        df.loc[is_capacity_peak, 'pQmxCost'] = capacity_charge_value
        
        print(f"\nCapacity Charge: Enabled")
        print(f"  Capacity peak period: {capacity_peak_start} to {capacity_peak_end}")
        print(f"  Capacity charge: ${capacity_charge_value:.2f}")
        
    else:
        # No capacity charges - all zeros
        df['pQmx'] = 0
        df['pQmxCost'] = 0.0
        print(f"\nCapacity Charge: Disabled")
    
    # Step 4: Generate constant cost columns
    df['pQco2'] = co2_cost
    df['pGcost'] = gas_cost
    df['pLcost'] = liquid_cost
    
    print(f"\nConstant Costs:")
    print(f"  CO2 cost (pQco2): {co2_cost}")
    print(f"  Gas cost (pGcost): {gas_cost}")
    print(f"  Liquid fuel cost (pLcost): {liquid_cost}")
    
    # Step 5: Generate lighting consumption column (pQlight)
    lighting_columns = [
        'out.electricity.lighting_exterior.energy_consumption..kwh',
        'out.electricity.lighting_garage.energy_consumption..kwh',
        'out.electricity.lighting_interior.energy_consumption..kwh',
        'out.natural_gas.lighting.energy_consumption..kwh'
    ]
    
    # Check which columns exist in the consumption data
    available_lighting_cols = [col for col in lighting_columns if col in consumption_df.columns]
    missing_cols = [col for col in lighting_columns if col not in consumption_df.columns]
    
    if missing_cols:
        print(f"\nWarning: Missing lighting columns: {missing_cols}")
    
    if available_lighting_cols:
        print(f"\nGenerating lighting consumption from {len(available_lighting_cols)} columns...")
        # Sum all available lighting columns element-wise (gives kWh at 15-min)
        lighting_kwh_15min = consumption_df[available_lighting_cols].sum(axis=1)
        
        if timestep == '15min':
            # For 15-min data: kWh over 0.25 hours, so multiply by 4 to get kW
            df['pQlight'] = (lighting_kwh_15min.values * 4)
            print(f"  Timestep is 15-min: converting kWh to kW (multiply by 4)")
        elif timestep == 'hourly':
            # For hourly data: sum to get kWh over 1 hour, which equals kW
            lighting_kwh_hourly = resample_if_needed(lighting_kwh_15min, method='sum')
            df['pQlight'] = lighting_kwh_hourly.values
            print(f"  Timestep is hourly: summed to hourly kWh = kW (no conversion needed)")
        
        print(f"  Lighting power (kW) range: {df['pQlight'].min():.6f} to {df['pQlight'].max():.6f} kW")
    else:
        raise ValueError("No lighting consumption columns found in consumption file")
    
    # Step 6: Calculate heating energy consumption (store as variable, not column yet)
    heating_columns = [
        'out.fuel_oil.heating.energy_consumption..kwh',
        'out.propane.heating.energy_consumption..kwh',
        'out.natural_gas.heating.energy_consumption..kwh',
        'out.electricity.heating.energy_consumption..kwh',
        'out.electricity.heating_fans_pumps.energy_consumption..kwh',
        'out.natural_gas.heating_hp_bkup.energy_consumption..kwh',
        'out.electricity.heating_hp_bkup.energy_consumption..kwh',
        'out.electricity.heating_hp_bkup_fa.energy_consumption..kwh'
    ]
    
    # Check which columns exist
    available_heating_cols = [col for col in heating_columns if col in consumption_df.columns]
    missing_heating_cols = [col for col in heating_columns if col not in consumption_df.columns]
    
    if missing_heating_cols:
        print(f"\nNote: Missing heating columns: {len(missing_heating_cols)}/{len(heating_columns)}")
    
    if available_heating_cols:
        print(f"\nCalculating heating energy consumption from {len(available_heating_cols)} columns...")
        # Sum all available heating columns element-wise (gives kWh at 15-min)
        heating_energy_15min = consumption_df[available_heating_cols].sum(axis=1)
        # Resample if needed (sum energy values)
        heating_energy_kwh = resample_if_needed(heating_energy_15min, method='sum').values
        print(f"  Heating energy (kWh) range: {heating_energy_kwh.min():.6f} to {heating_energy_kwh.max():.6f} kWh")
    else:
        print(f"\nWarning: No heating consumption columns found in consumption file")
        heating_energy_kwh = None
    
    # Step 7: Calculate cooling energy consumption (store as variable, not column yet)
    cooling_columns = [
        'out.electricity.cooling.energy_consumption..kwh',
        'out.electricity.cooling_fans_pumps.energy_consumption..kwh'
    ]
    
    # Check which columns exist
    available_cooling_cols = [col for col in cooling_columns if col in consumption_df.columns]
    missing_cooling_cols = [col for col in cooling_columns if col not in consumption_df.columns]
    
    if missing_cooling_cols:
        print(f"\nNote: Missing cooling columns: {len(missing_cooling_cols)}/{len(cooling_columns)}")
    
    if available_cooling_cols:
        print(f"\nCalculating cooling energy consumption from {len(available_cooling_cols)} columns...")
        # Sum all available cooling columns element-wise (gives kWh at 15-min)
        cooling_energy_15min = consumption_df[available_cooling_cols].sum(axis=1)
        # Resample if needed (sum energy values)
        cooling_energy_kwh = resample_if_needed(cooling_energy_15min, method='sum').values
        print(f"  Cooling energy (kWh) range: {cooling_energy_kwh.min():.6f} to {cooling_energy_kwh.max():.6f} kWh")
    else:
        print(f"\nWarning: No cooling consumption columns found in consumption file")
        cooling_energy_kwh = None
    
    # Step 8: Get EV charging energy consumption (store as variable, not column yet)
    ev_column = 'out.electricity.ev_charging.energy_consumption..kwh'
    
    if ev_column in consumption_df.columns:
        print(f"\nGetting EV charging energy consumption...")
        # Get EV charging column (gives kWh at 15-min)
        ev_charging_15min = consumption_df[ev_column]
        # Resample if needed (sum energy values)
        ev_charging_kwh = resample_if_needed(ev_charging_15min, method='sum').values
        print(f"  EV charging energy (kWh) range: {ev_charging_kwh.min():.6f} to {ev_charging_kwh.max():.6f} kWh")
    else:
        print(f"\nNote: EV charging column not found in consumption file")
        ev_charging_kwh = None
    
    # Step 9: Calculate hot water energy consumption (store as variable, not column yet)
    hot_water_columns = [
        'out.electricity.hot_water.energy_consumption..kwh',
        'out.electricity.hot_water_solar_th.energy_consumption..kwh',
        'out.fuel_oil.hot_water.energy_consumption..kwh',
        'out.natural_gas.hot_water.energy_consumption..kwh',
        'out.propane.hot_water.energy_consumption..kwh'
    ]
    
    # Check which columns exist
    available_hot_water_cols = [col for col in hot_water_columns if col in consumption_df.columns]
    missing_hot_water_cols = [col for col in hot_water_columns if col not in consumption_df.columns]
    
    if missing_hot_water_cols:
        print(f"\nNote: Missing hot water columns: {len(missing_hot_water_cols)}/{len(hot_water_columns)}")
    
    if available_hot_water_cols:
        print(f"\nCalculating hot water energy consumption from {len(available_hot_water_cols)} columns...")
        # Sum all available hot water columns element-wise (gives kWh at 15-min)
        hot_water_energy_15min = consumption_df[available_hot_water_cols].sum(axis=1)
        # Resample if needed (sum energy values)
        hot_water_energy_kwh = resample_if_needed(hot_water_energy_15min, method='sum').values
        print(f"  Hot water energy (kWh) range: {hot_water_energy_kwh.min():.6f} to {hot_water_energy_kwh.max():.6f} kWh")
    else:
        print(f"\nWarning: No hot water consumption columns found in consumption file")
        hot_water_energy_kwh = None
    
    # Step 10: Calculate total energy consumption (store as variable, not column yet)
    total_energy_columns = [
        'out.electricity.total.energy_consumption..kwh',
        'out.fuel_oil.total.energy_consumption..kwh',
        'out.natural_gas.total.energy_consumption..kwh',
        'out.propane.total.energy_consumption..kwh'
    ]
    
    # Check which columns exist
    available_total_energy_cols = [col for col in total_energy_columns if col in consumption_df.columns]
    missing_total_energy_cols = [col for col in total_energy_columns if col not in consumption_df.columns]
    
    if missing_total_energy_cols:
        print(f"\nNote: Missing total energy columns: {len(missing_total_energy_cols)}/{len(total_energy_columns)}")
    
    if available_total_energy_cols:
        print(f"\nCalculating total energy consumption from {len(available_total_energy_cols)} columns...")
        # Sum all available total energy columns element-wise (gives kWh at 15-min)
        total_energy_15min = consumption_df[available_total_energy_cols].sum(axis=1)
        # Resample if needed (sum energy values)
        total_energy_kwh = resample_if_needed(total_energy_15min, method='sum').values
        print(f"  Total energy (kWh) range: {total_energy_kwh.min():.6f} to {total_energy_kwh.max():.6f} kWh")
    else:
        print(f"\nWarning: No total energy consumption columns found in consumption file")
        total_energy_kwh = None
    
    # Step 11: Calculate equipment power (pQequip)
    # Equipment = Total - Heating - Cooling - EV Charging - Hot Water
    if total_energy_kwh is not None:
        print(f"\nCalculating equipment energy consumption...")
        equipment_energy_kwh = total_energy_kwh.copy()
        
        # Subtract each component if available
        if heating_energy_kwh is not None:
            equipment_energy_kwh -= heating_energy_kwh
            print(f"  Subtracted heating")
        if cooling_energy_kwh is not None:
            equipment_energy_kwh -= cooling_energy_kwh
            print(f"  Subtracted cooling")
        if ev_charging_kwh is not None:
            equipment_energy_kwh -= ev_charging_kwh
            print(f"  Subtracted EV charging")
        if hot_water_energy_kwh is not None:
            equipment_energy_kwh -= hot_water_energy_kwh
            print(f"  Subtracted hot water")
        
        # Convert to kW based on timestep
        if timestep == 'hourly':
            # For hourly data: kWh over 1 hour = kW (identical)
            df['pQequip'] = equipment_energy_kwh
            print(f"  Timestep is hourly: kWh = kW (no conversion needed)")
        elif timestep == '15min':
            # For 15-min data: kWh over 0.25 hours, so multiply by 4 to get kW
            df['pQequip'] = equipment_energy_kwh * 4
            print(f"  Timestep is 15-min: converting kWh to kW (multiply by 4)")
        
        print(f"  Equipment power (kW) range: {df['pQequip'].min():.6f} to {df['pQequip'].max():.6f} kW")
    else:
        raise ValueError("Cannot calculate equipment power: total energy data not available")
    
    # Step 12: Get outdoor temperature (pTout)
    temp_column = 'out.outdoor_air_drybulb_temp..c'
    
    if temp_column in consumption_df.columns:
        print(f"\nGetting outdoor temperature...")
        temp_15min = consumption_df[temp_column]
        # Resample if needed (mean for temperature)
        df['pTout'] = resample_if_needed(temp_15min, method='mean').values
        print(f"  Outdoor temperature (°C) range: {df['pTout'].min():.2f} to {df['pTout'].max():.2f} °C")
    else:
        raise ValueError(f"Outdoor temperature column not found: {temp_column}")
    
    # Step 13: Get temperature setpoints (pTmx and pTmn)
    if 'max_comfort_temp' in setpoint_df.columns:
        print(f"\nGetting maximum comfort temperature setpoint...")
        tmx_15min = setpoint_df['max_comfort_temp']
        # Resample if needed (mean for setpoints)
        df['pTmx'] = resample_if_needed(tmx_15min, method='mean').values
        print(f"  Max comfort temp (°C) range: {df['pTmx'].min():.2f} to {df['pTmx'].max():.2f} °C")
    else:
        raise ValueError(f"Max comfort temperature column not found in setpoint file")
    
    if 'min_comfort_temp' in setpoint_df.columns:
        print(f"\nGetting minimum comfort temperature setpoint...")
        tmn_15min = setpoint_df['min_comfort_temp']
        # Resample if needed (mean for setpoints)
        df['pTmn'] = resample_if_needed(tmn_15min, method='mean').values
        print(f"  Min comfort temp (°C) range: {df['pTmn'].min():.2f} to {df['pTmn'].max():.2f} °C")
    else:
        raise ValueError(f"Min comfort temperature column not found in setpoint file")
    
    # Step 14: Calculate HVAC on/off status (pTon)
    print(f"\nCalculating HVAC on/off status (pTon)...")
    
    # Check if mode column exists in setpoint data
    if 'mode' not in setpoint_df.columns:
        raise ValueError("Mode column not found in setpoint file - needed for pTon calculation")
    
    # Get mode at appropriate resolution
    if timestep == '15min':
        mode_series = setpoint_df['mode']
    else:  # hourly
        mode_15min = setpoint_df['mode']
        mode_series = resample_if_needed(mode_15min, method='mean')
        # Round to get most common mode in the hour
        mode_series = mode_series.round()
    
    # Start with all periods available (1)
    df['pTon'] = 1
    
    # Process heating unavailable period
    heating_unavail = hvac_data.get('in.heating_unavailable_period', 'Never')
    print(f"  Heating unavailable period: {heating_unavail}")
    
    if heating_unavail and str(heating_unavail).lower() not in ['never', 'nan', 'none']:
        if str(heating_unavail).lower() in ['year round', 'year-round']:
            # Check if mode is heating (1) or shoulder (0)
            heating_mask = (mode_series.values == 1) | (mode_series.values == 0)
            df.loc[heating_mask, 'pTon'] = 0
            print(f"    Heating unavailable year-round: set pTon=0 for heating/shoulder seasons")
        else:
            # Parse date range like "Mar 25 - Apr 23"
            try:
                parts = str(heating_unavail).split('-')
                if len(parts) == 2:
                    start_str = parts[0].strip()
                    end_str = parts[1].strip()
                    
                    # Parse dates (assuming format like "Mar 25" or "Jan 1")
                    start_date = datetime.strptime(f"{start_str} {year}", "%b %d %Y")
                    end_date = datetime.strptime(f"{end_str} {year}", "%b %d %Y")
                    
                    # Determine which timestamps are in the unavailable period
                    if end_date < start_date:
                        # Period crosses year boundary
                        period_mask = (df.index.month > start_date.month) | \
                                     ((df.index.month == start_date.month) & (df.index.day >= start_date.day)) | \
                                     (df.index.month < end_date.month) | \
                                     ((df.index.month == end_date.month) & (df.index.day <= end_date.day))
                    else:
                        # Normal period within same year
                        period_mask = ((df.index.month > start_date.month) | \
                                     ((df.index.month == start_date.month) & (df.index.day >= start_date.day))) & \
                                     ((df.index.month < end_date.month) | \
                                     ((df.index.month == end_date.month) & (df.index.day <= end_date.day)))
                    
                    # Only set to 0 if BOTH in unavailable period AND in heating/shoulder season (mode = 1 or 0)
                    heating_season_mask = (mode_series.values == 1) | (mode_series.values == 0)
                    combined_mask = period_mask & heating_season_mask
                    df.loc[combined_mask, 'pTon'] = 0
                    print(f"    Heating unavailable {start_str} to {end_str}: set to 0 for heating/shoulder periods")
            except Exception as e:
                print(f"    Warning: Could not parse heating unavailable period: {e}")
    
    # Process cooling unavailable period
    cooling_unavail = hvac_data.get('in.cooling_unavailable_period', 'Never')
    print(f"  Cooling unavailable period: {cooling_unavail}")
    
    if cooling_unavail and str(cooling_unavail).lower() not in ['never', 'nan', 'none']:
        if str(cooling_unavail).lower() in ['year round', 'year-round']:
            # Check if mode is cooling (-1) or shoulder (0)
            cooling_mask = (mode_series.values == -1) | (mode_series.values == 0)
            df.loc[cooling_mask, 'pTon'] = 0
            print(f"    Cooling unavailable year-round: set pTon=0 for cooling/shoulder seasons")
        else:
            # Parse date range like "Mar 25 - Apr 23"
            try:
                parts = str(cooling_unavail).split('-')
                if len(parts) == 2:
                    start_str = parts[0].strip()
                    end_str = parts[1].strip()
                    
                    # Parse dates
                    start_date = datetime.strptime(f"{start_str} {year}", "%b %d %Y")
                    end_date = datetime.strptime(f"{end_str} {year}", "%b %d %Y")
                    
                    # Determine which timestamps are in the unavailable period
                    if end_date < start_date:
                        period_mask = (df.index.month > start_date.month) | \
                                     ((df.index.month == start_date.month) & (df.index.day >= start_date.day)) | \
                                     (df.index.month < end_date.month) | \
                                     ((df.index.month == end_date.month) & (df.index.day <= end_date.day))
                    else:
                        period_mask = ((df.index.month > start_date.month) | \
                                     ((df.index.month == start_date.month) & (df.index.day >= start_date.day))) & \
                                     ((df.index.month < end_date.month) | \
                                     ((df.index.month == end_date.month) & (df.index.day <= end_date.day)))
                    
                    # Only set to 0 if BOTH in unavailable period AND in cooling/shoulder season (mode = -1 or 0)
                    cooling_season_mask = (mode_series.values == -1) | (mode_series.values == 0)
                    combined_mask = period_mask & cooling_season_mask
                    df.loc[combined_mask, 'pTon'] = 0
                    print(f"    Cooling unavailable {start_str} to {end_str}: set to 0 for cooling/shoulder periods")
            except Exception as e:
                print(f"    Warning: Could not parse cooling unavailable period: {e}")
    
    print(f"  pTon values: {df['pTon'].unique()}")
    print(f"  HVAC available periods: {(df['pTon'] == 1).sum()}")
    print(f"  HVAC unavailable periods: {(df['pTon'] == 0).sum()}")
    
    # Step 15: Calculate hot water demand (pHWdem)
    hot_water_load_columns = [
        'out.load.hot_water.energy_delivered..kbtu',
        'out.load.hot_water_solar_thermal..kbtu',
        'out.load.hot_water_tank_losses..kbtu'
    ]
    
    # Check which columns exist
    available_hw_load_cols = [col for col in hot_water_load_columns if col in consumption_df.columns]
    missing_hw_load_cols = [col for col in hot_water_load_columns if col not in consumption_df.columns]
    
    if missing_hw_load_cols:
        print(f"\nNote: Missing hot water load columns: {len(missing_hw_load_cols)}/{len(hot_water_load_columns)}")
    
    if available_hw_load_cols:
        print(f"\nCalculating hot water demand from {len(available_hw_load_cols)} columns...")
        # Sum all available hot water load columns element-wise (in kBtu at 15-min)
        hw_demand_kbtu_15min = consumption_df[available_hw_load_cols].sum(axis=1)
        
        # Convert kBtu to kWh (1 kBtu = 0.293071 kWh)
        hw_demand_kwh_15min = hw_demand_kbtu_15min * 0.293071
        
        # Resample if needed (sum energy values for hourly)
        if timestep == '15min':
            df['pHWdem'] = hw_demand_kwh_15min.values
        elif timestep == 'hourly':
            hw_demand_kwh_hourly = resample_if_needed(hw_demand_kwh_15min, method='sum')
            df['pHWdem'] = hw_demand_kwh_hourly.values
        
        print(f"  Hot water demand (kWh) range: {df['pHWdem'].min():.6f} to {df['pHWdem'].max():.6f} kWh")
    else:
        print(f"\nWarning: No hot water load columns found, setting pHWdem to 0")
        df['pHWdem'] = 0.0
    
    # Step 16: Get solar radiation and wind speed
    # Direct Normal Irradiance (pSdni)
    dni_column = 'out.weather.direct_normal_solar_radiation..watt_per_m2'
    if dni_column in consumption_df.columns:
        print(f"\nGetting direct normal solar irradiance...")
        dni_15min = consumption_df[dni_column]
        # Resample if needed (mean for weather)
        dni = resample_if_needed(dni_15min, method='mean')
        # Convert W/m² to kW/m² by dividing by 1000
        df['pSdni'] = dni.values / 1000.0
        print(f"  Direct normal irradiance (kW/m²) range: {df['pSdni'].min():.6f} to {df['pSdni'].max():.6f} kW/m²")
    else:
        print(f"\nWarning: DNI column not found, setting pSdni to 0")
        df['pSdni'] = 0.0
    
    # Diffuse Horizontal Irradiance (pSdhi)
    dhi_column = 'out.weather.diffuse_solar_radiation..watt_per_m2'
    if dhi_column in consumption_df.columns:
        print(f"\nGetting diffuse horizontal solar irradiance...")
        dhi_15min = consumption_df[dhi_column]
        # Resample if needed (mean for weather)
        dhi = resample_if_needed(dhi_15min, method='mean')
        # Convert W/m² to kW/m² by dividing by 1000
        df['pSdhi'] = dhi.values / 1000.0
        print(f"  Diffuse horizontal irradiance (kW/m²) range: {df['pSdhi'].min():.6f} to {df['pSdhi'].max():.6f} kW/m²")
    else:
        print(f"\nWarning: DHI column not found, setting pSdhi to 0")
        df['pSdhi'] = 0.0
    
    # Wind Speed (pWms)
    wind_column = 'out.weather.wind_speed..meter_per_second'
    if wind_column in consumption_df.columns:
        print(f"\nGetting wind speed...")
        wind_15min = consumption_df[wind_column]
        # Resample if needed (mean for weather)
        df['pWms'] = resample_if_needed(wind_15min, method='mean').values
        print(f"  Wind speed (m/s) range: {df['pWms'].min():.2f} to {df['pWms'].max():.2f} m/s")
    else:
        print(f"\nWarning: Wind speed column not found, setting pWms to 0")
        df['pWms'] = 0.0
    
    # Step 17: EV charging columns (set to zeros for now)
    print(f"\nSetting EV charging columns to zero...")
    df['pEVdem'] = 0.0  # EV demand (float)
    df['pEVtm'] = 0.0   # EV time (float)
    df['pEVtype'] = 0   # EV type (integer)
    
    # Step 18: Calculate number of occupants (pBppl)
    occupants_schedule_column = 'out.schedules.occupants'
    occupants_metadata_column = 'in.occupants'
    
    if occupants_schedule_column in consumption_df.columns:
        print(f"\nCalculating number of occupants...")
        
        # Get the occupancy schedule (fraction from 0 to 1) at 15-min resolution
        occupancy_schedule_15min = consumption_df[occupants_schedule_column]
        
        # Resample if needed (mean for schedules/rates)
        occupancy_schedule = resample_if_needed(occupancy_schedule_15min, method='mean')
        
        print(f"  Occupancy schedule range: {occupancy_schedule.min():.3f} to {occupancy_schedule.max():.3f}")
        
        # Get the total number of occupants from metadata
        if occupants_metadata_column in hvac_data or occupants_metadata_column in building_row.columns:
            if occupants_metadata_column in building_row.columns:
                total_occupants = int(building_row[occupants_metadata_column].iloc[0])
            else:
                total_occupants = int(hvac_data.get(occupants_metadata_column, 1))
            
            print(f"  Total occupants for building: {total_occupants}")
            
            # Multiply schedule by total occupants and round to nearest integer
            df['pBppl'] = (occupancy_schedule.values * total_occupants).round().astype(int)
            
            print(f"  Number of occupants range: {df['pBppl'].min()} to {df['pBppl'].max()}")
        else:
            print(f"  Warning: {occupants_metadata_column} not found in metadata, using schedule only")
            df['pBppl'] = occupancy_schedule.values.round().astype(int)
    else:
        print(f"\nWarning: Occupancy schedule column not found, setting pBppl to 0")
        df['pBppl'] = 0
    
    return df


def save_building_data(df, bldg_id, parent_path):
    """
    Save the building data to CSV file
    
    Args:
        df: DataFrame with building data
        bldg_id: Building ID
        parent_path: Parent directory path
    """
    # Create output directory structure
    output_dir = parent_path / "in" / "decarb_inputs" / "base" / str(bldg_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = output_dir / "tm.csv"
    # Format datetime index as YYYY-MM-DDTHH:MM
    df_to_save = df.copy()
    df_to_save.index = df_to_save.index.strftime('%Y-%m-%dT%H:%M')
    df_to_save.index.name = 'pP'  # Preserve the index name
    df_to_save.to_csv(output_file, sep=',')
    
    print(f"\n" + "="*80)
    print(f"Data saved to: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("="*80)
    
    return output_file


def test_function():
    """Run test cases for the building data generator"""
    print("="*80)
    print("TEST 1: Building 77 - Hourly with Flat Tariff, No Capacity Charge")
    print("="*80)
    df1 = generate_building_data(bldg_id=77, timestep='hourly', year=2018, 
                                  tariff_type='flat', tariff_value=0.22,
                                  capacity_charge=False)
    print(f"\nShape: {df1.shape}")
    print("\nFirst 24 hours:")
    print(df1.head(24))
    
    print("\n" + "="*80)
    print("TEST 2: Building 77 - Hourly with TOU Tariff + Capacity Charge")
    print("="*80)
    df2 = generate_building_data(bldg_id=77, timestep='hourly', year=2018,
                                  tariff_type='tou', 
                                  peak_start='16:00', peak_end='19:00',
                                  peak_price=0.28,
                                  capacity_charge=True,
                                  capacity_peak_start='19:00',
                                  capacity_peak_end='21:00',
                                  capacity_charge_value=0.5)
    print(f"\nShape: {df2.shape}")
    print("\nSample day (2 PM - midnight) showing all columns:")
    print(df2.iloc[14:24])
    
    print("\n" + "="*80)
    print("TEST 3: Building 77 - 15-minute with Flat Tariff + Capacity Charge")
    print("="*80)
    df3 = generate_building_data(bldg_id=77, timestep='15min', year=2018,
                                  tariff_type='flat',
                                  tariff_value=0.22,
                                  capacity_charge=True,
                                  capacity_peak_start='19:00',
                                  capacity_peak_end='21:00',
                                  capacity_charge_value=0.5)
    print(f"\nShape: {df3.shape}")
    print("\nSample around capacity peak (6 PM - 8 PM, every 30 min):")
    # Show every other 15-min interval from 6pm to 8pm
    sample_indices = list(range(72, 80, 2))  # 18:00 to 20:00
    print(df3.iloc[sample_indices])
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"\nTest 1 (Hourly Flat):")
    print(f"  Total lighting consumption: {df1['pQlight'].sum():.2f} kW")
    print(f"  Average lighting per hour: {df1['pQlight'].mean():.4f} kW")
    
    print(f"\nTest 3 (15-min):")
    print(f"  Total lighting consumption: {df3['pQlight'].sum():.2f} kW")
    print(f"  Average lighting per 15-min: {df3['pQlight'].mean():.4f} kW")


if __name__ == "__main__":
    # Configuration
    test_mode = False  # Set to True for testing, False for production
    bldg_id = 250
    timestep = 'hourly'  # 'hourly' or '15min'
    
    parent_path = Path(__file__).parents[1]
    
    if test_mode:
        # TEST MODE
        test_function()
        
    else:
        # PRODUCTION MODE
        print("="*80)
        print(f"Generating data for Building {bldg_id} - {timestep} resolution")
        print("="*80)
        
        df = generate_building_data(
            bldg_id=bldg_id,
            timestep=timestep,
            year=2018,
            tariff_type='flat',
            tariff_value=0.22,
            capacity_charge=False
        )
        
        # Save the data
        output_file = save_building_data(df, bldg_id, parent_path)
        
        # Format for display
        df_display = df.copy()
        df_display.index = df_display.index.strftime('%Y-%m-%dT%H:%M')
        df_display.index.name = 'pP'
        
        print(f"\nFirst 10 rows:")
        print(df_display.head(10))
        print(f"\nLast 10 rows:")
        print(df_display.tail(10))