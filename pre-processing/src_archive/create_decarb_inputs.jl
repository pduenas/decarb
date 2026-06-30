using CSV
using DataFrames
using Dates
using Tables
using StatsBase
using Interpolations
using GLM

HSPF_MIN = 9

function fahrenheit_to_celsius(f_temp)
    c_temp =(f_temp - 32) * (5.0 / 9.0)
    return c_temp
end

function celsius_to_fahrenheit(c_temp)
    f_temp = c_temp * (9.0 / 5.0) + 32
    return f_temp
end

function single_family(geometry)
    geometry == "Single-Family Detached"
end

function hspf_filter(hspf_rating)
    hspf_rating >= HSPF_MIN
end

function success_only(c_status)
    c_status == "Success"
end

function ashp_only(hvac_equip)
    typeof(match(r".*ASHP.*", hvac_equip)) != Nothing
end

function neep_only(hvac_equip)
    typeof(match(r".*NEEP.*", hvac_equip)) != Nothing
end

function match_wall_insulation(type, search, target)
    search == target && type == "wall"
end

function match_window_insulation(type, search, target)
    search == target && type == "window"
end

function match_roof_insulation(type, search, target)
    search == target && type == "roof"
end

function hp_cost(capacitykW, sqft, dim_eff)
    # convert units from kW to BTU
    capacityBTU = capacitykW * 3412.142
    HSPF = dim_eff * 3.412142
    # calculate cost elements
    appliance_est = (933.88 .+ 0.14801 .* capacityBTU .+ 1.11258 .* sqft)+(HSPF.-11) .* 0.1 .* (933.88 .+ 0.14801 .* capacityBTU .+ 1.11258 .* sqft)
    labor_est = 2286 .+ 0.1179 .* capacityBTU .+ 0.6556 .* sqft
    scaledmisc_est = 74.797 .+ 1.1028 .* sqft
    flatmisc_est = 180

    total_cost = appliance_est .+ labor_est .+ scaledmisc_est .+ flatmisc_est

    println(size(total_cost))
    return total_cost
end

in_alias = "build_existing_model."
template_dirname = "in_templates"
wd = pwd()
println(wd)
resstock_dirname = joinpath(wd, "in", "resstock_files") # change this
fname_summary = "results-Baseline_jan_15.csv" # metadata.csv in R code

# read in metadata file
df_summary = CSV.File(joinpath(resstock_dirname, fname_summary)) |> DataFrame
# println(df_summary)

# Filter for only successful runs
df_success_only = filter("completed_status" => success_only, df_summary)

# filter for only single family detached homes
df_sf_only = filter((in_alias * "geometry_building_type_acs") => single_family, df_success_only)

# println(df_sf_only.building_id)

building_id_vector = []
write_dirname_root = joinpath(wd, "Models")

pdls_vector = []
bid_vector = []
duct_vector = []
hs_vector = []

# for each row in summary, find the appropriate timeseries file and create input files
# loop over all filenames (either through some logic or file with all names as column)
for i in 1:size(df_sf_only,1)
    row = df_sf_only[i,:]
    
    # Extract county and puma values
    county_puma = row[in_alias * "county_and_puma"]
    comma_index = findfirst(",", county_puma)
    county = county_puma[1:(comma_index[1]-1)]

    # Find correct weather data from directory with all weather
    weather_dirname = joinpath(wd, "in", "weather_files")
    weather_fname = county * "_2018.csv"

    # write building directories to vector (written at end to CSV file)
    building_id = row["building_id"]
    building_id_dirname = "Model_" * string(building_id)
    push!(building_id_vector, joinpath(write_dirname_root, building_id_dirname))

    # create building directory, will overwrite if one exists
    building_dirname = joinpath(write_dirname_root, building_id_dirname)
    mkdir(building_dirname)

    # Grab dry bulb temperature, direct radiation, and diffuse radiation from temperature file
    weather_df = CSV.File(joinpath(weather_dirname, weather_fname)) |> DataFrame
    temp_vector = weather_df[!,"Dry Bulb Temperature [°C]"]
    direct_radiation_vector = weather_df[!,"Direct Normal Radiation [W/m2]"] ./ 1000.0
    diffuse_radiation_vector = weather_df[!,"Diffuse Horizontal Radiation [W/m2]"] ./ 1000.0

    # Fetch timeseries data
    # currently hardcoded; could fetch from data.openei.org and HTTP library
    # these should be named according to the building ids

    fname_time_series = "results_timeseries.csv"
    #df_ts = CSV.File(joinpath(resstock_dirname, fname_time_series)) |> DataFrame
    run_path = joinpath(resstock_dirname, "400 samples single-family MA 2018 weather", "ResStock_outputs", "run" * string(building_id), "run")
    println("Time series data path for each house")
    println(run_path)
    df_ts_temp = CSV.File(joinpath(run_path, fname_time_series);type=Float64) |> DataFrame # Causes warning because of date column
    df_ts = df_ts_temp[2:end, :] # omit first row with units

    ### Process timeseries data for DECARB inputs

    mkdir(joinpath(building_dirname, "in"))
    mkdir(joinpath(building_dirname, "out"))
    filenames = ["abs", "bdg_i", "bdg_ii", "bess", "chp", "hvac", "pv", "sp", "topo", "wh"]

    # Assign values from ResStock to DECARB input files
    in_dirname = joinpath(wd, template_dirname)
    tm_fname = "tm.csv"
    bdgi_fname = "bdg_i.csv"
    bdgii_fname = "bdg_ii.csv"
    chp_fname = "chp.csv"
    hvac_fname = "hvac.csv"
    wh_fname = "wh.csv"
    sp_fname = "sp.csv"
    topo_fname = "topo.csv"

    # Read timeseries file into a dataframe
    df_tm = CSV.File(joinpath(in_dirname, tm_fname)) |> DataFrame

    lighting_interior = 0.0
    lighting_exterior = 0.0

    # Total lighting
    try
        lighting_interior = df_ts[:,"End Use: Electricity: Lighting Interior"]
    catch
        println("NO LIGHTING INTERIOR")
    end

    try
        lighting_exterior = df_ts[:,"End Use: Electricity: Lighting Exterior"]
    catch
        println("NO EXTERIOR INTERIOR")
    end
    
    lighting_total_vector = lighting_interior .+ lighting_exterior # broadcast in case either one is 0

    # Total equipment load
    #=
    mech_vent = 0
    refrigerator = 0
    freezer = 0
    dishwasher = 0
    clothes_washer = 0
    clothes_dryer = 0
    range_oven = 0
    ceiling_fan = 0
    plug_loads = 0

    End Use: Electricity: Mech Vent
    End Use: Electricity: Refrigerator
    End Use: Electricity: Freezer
    End Use: Electricity: Dishwasher
    End Use: Electricity: Clothes Washer
    End Use: Electricity: Clothes Dryer
    End Use: Electricity: Range/Oven
    End Use: Electricity: Ceiling Fan
    End Use: Electricity: Plug Loads
    =#

    # @TODO need to change this so that doesn't accidentally grab solar
    el_df = df_ts[!, r".*End Use: Electricity.*"] # This gets all end use electricity loads
    el_minus_lighting_df = el_df[!, r"^(?!.*(Lighting|Cooling|Heating)).*$"] # Omit lighting, heating, and cooling loads
    equipment_total_vector = sum(eachcol(el_minus_lighting_df))
    
    #equipment_total_vector = mech_vent .+ refrigerator .+ freezer .+ dishwasher .+ clothes_washer .+ clothes_dryer .+ range_oven .+ ceiling_fan .+ plug_loads

    # Total water heater load
    water_df = df_ts[!, r"Load: Hot Water: Delivered$"] # this captures all water heating types because is delivered energy
    water_consumption = sum(eachcol(water_df))
    water_heating_vector = water_consumption .* 0.293071 # convert from kBtu to kWh

    # Assign columns to tm file
    df_tm[!, :pQlight] .= lighting_total_vector # Needs to be broadcast in case the lighting vector is empty
    df_tm.pQequip = equipment_total_vector
    df_tm.pTout = temp_vector
    df_tm.pSdni = direct_radiation_vector
    df_tm.pSdhi = diffuse_radiation_vector
    df_tm.pHWdem = water_heating_vector

    # Write timeseries file
    CSV.write(joinpath(building_dirname, "in", "tm.csv"), df_tm)

    # Read in building files from templates
    df_bdg1 = CSV.File(joinpath(in_dirname, bdgi_fname); transpose=true) |> DataFrame
    df_bdg2 = CSV.File(joinpath(in_dirname, bdgii_fname)) |> DataFrame
    
    # Grab latitude and longitude
    latitude = row[in_alias * "weather_file_latitude"]
    longitude = row[in_alias * "weather_file_longitude"]

    # Assign latitude and longitude to dataframe
    df_bdg1.pBlat[1] = latitude
    df_bdg1.pBlon[1] = longitude

    # Grab average of the sqft range (note: this differs from R code because
    # Morgan's file includes a range versus the actual square footage)
    # Could also try upgrade_costs.floor_area_conditioned_ft_2
    sqft = 0
    floor_area_range = row[in_alias * "geometry_floor_area"] # @TODO ask Morgan about which column is closest to floor area
    dash_index = findfirst("-", floor_area_range)
    if typeof(dash_index) == Nothing
        plus_index = findfirst("+", floor_area_range)
        sqft = parse(Float64, floor_area_range[1:(plus_index[1]-1)])
    else
        low = parse(Float64, floor_area_range[1:(dash_index[1]-1)])
        high = parse(Float64, floor_area_range[(dash_index[1]+1):end])
        sqft = (low + high) / 2.0
    end

    # Calculate square building footprint (constant is sqft2 to m2)
    df_bdg1.pBfoot[1] = (sqft / row[in_alias * "geometry_stories"]) * 0.092903

    # Lateral area (assume 3 vertical meters per story)
    # Instead, could use variable wall area above grade
    df_bdg1.pBeast[1] = sqrt(df_bdg1.pBfoot[1]) * row[in_alias * "geometry_stories"] * 3
    df_bdg1.pBwest[1] = sqrt(df_bdg1.pBfoot[1]) * row[in_alias * "geometry_stories"] * 3
    df_bdg1.pBnorth[1] = sqrt(df_bdg1.pBfoot[1]) * row[in_alias * "geometry_stories"] * 3
    df_bdg1.pBsouth[1] = sqrt(df_bdg1.pBfoot[1]) * row[in_alias * "geometry_stories"] * 3
    
    #=
    df_bdg1.pBeast[1] = df_bdg1.pBfoot[1] * 2.3/14 # hack backed on room measurements
    df_bdg1.pBwest[1] = 0
    df_bdg1.pBnorth[1] = 0
    df_bdg1.pBsouth[1] = df_bdg1.pBfoot[1] * 2.3/14
    =#

    # Roof footprint assumed to be same as building footprint (rectangular roof)
    df_bdg1.pBpv[1] = df_bdg1.pBfoot[1]

    insulation_df = CSV.File(joinpath(wd, "insulation_crosswalk.csv");types=[String,String,Float64]) |> DataFrame
    
    # Wall insulation
    wall_insulation = string(row[in_alias * "insulation_wall"])
    filtered_wall_df = filter(row -> row.type == "wall" && row.rs_name == wall_insulation, insulation_df)
    pbkwall_value = filtered_wall_df[!, :pBkwall][1]
    
    # Window insulation
    windows = row[in_alias * "windows"]
    filtered_window_df = filter(row -> row.type == "windows" && row.rs_name == windows, insulation_df)
    pbkwindow_value = filtered_window_df[!, :pBkwall][1]

    # Roof insulation
    roof_insulation = row[in_alias * "insulation_roof"]
    filtered_roof_df = filter(row -> row.type == "roof" && row.rs_name == roof_insulation, insulation_df)
    pbkroof_value = filtered_roof_df[!, :pBkwall][1]

    # Extract surface to wall ratio
    # @TODO - check with Jameson on .[1] notation
    numeric_matches = []
    for s in eachmatch(r"[0-9]+", row[in_alias * "window_areas"])
        push!(numeric_matches, parse(Int64, s.match))
    end
    window_ratio = numeric_matches[1]
    df_bdg2.pBwall[1] = (df_bdg1.pBwest[1] + df_bdg1.pBeast[1] + df_bdg1.pBnorth[1] + df_bdg1.pBsouth[1]) * (1 - window_ratio)
    df_bdg2.pBwall[2] = df_bdg2.pBwall[1] * window_ratio
    df_bdg2.pBwall[3] = df_bdg1.pBpv[1]

    df_bdg2.pBkwall[1] = pbkwall_value / 2.0 # @TODO check with Jameson on .[1] notation
    df_bdg2.pBkwall[2] = pbkwindow_value
    df_bdg2.pBkwall[3] = pbkroof_value

    # Write building files
    CSV.write(joinpath(building_dirname, "in", "bdg_ii.csv"), df_bdg2)

    # CHP catalog
    heating_vector = df_ts[!, "Load: Heating: Delivered"] # delivered kbtu
    #heating_consumption = sum(eachcol(heating_df))
    max_heat = maximum(heating_vector) * 0.293071 # convert from kBtu/hr to kW
    println("MAX HEAT")
    println(max_heat)

    # Grab 99th percentile temperature and load
    df_lt = DataFrame(temperature = temp_vector, load = heating_vector)
    design_temp = percentile(temp_vector, 1)
    println("DESIGN TEMP")
    println(design_temp)
    #=
    println(design_temp)
    filtered_df_lt = df_lt[df_lt.temperature .< design_temp, :]
    println("DESIGN LOADS")
    design_load = mean(filtered_df_lt[!, :load]) * 1000 # convert to BTU
    println(design_load)
    =#
    design_load = percentile(heating_vector, 99) * 1000 # convert to BTU to match NEEP
    println("DESIGN LOAD")
    println(design_load)

    # Choose the heat pump from the catalog such that max load @ design temp is 90% - 120% of the design load
    
    # 1. Draw a linear curve to interpolate the max cap at design temp
    # 2. sort the heat pumps in catalog according to max cap
    # 3. Pick closest one s.t. max load @ design temp is 90-120% of the design load
    # 4. Add equipment to sp so that it is used (pick whatever is currently installed as backup system)

    heating_fuel = row[in_alias * "heating_fuel"]
    push!(hs_vector, heating_fuel)

    duct_type = row[in_alias * "hvac_heating_type"]
    neep_df = nothing

    # get applicable neep database and remove duplicate models (matching indoor and outdoor model #)
    if duct_type == "Ducted Heating"
        push!(duct_vector, "Ducted")
        global neep_df = CSV.File(joinpath(wd, "neep_catalog_ducted.csv")) |> DataFrame
        global neep_df = unique(neep_df, ["Outdoor Unit Model", "Indoor Model(s)"])
    elseif duct_type == "Non-Ducted Heating"
        push!(duct_vector, "Non-Ducted")
        global neep_df = CSV.File(joinpath(wd, "neep_catalog_ductless.csv")) |> DataFrame
        global neep_df = unique(neep_df, ["Outdoor Unit Model", "Indoor Model(s)"])
        global neep_df = filter("HSPF (Region IV)" => hspf_filter, neep_df) # Filter for only high-efficiency models
    else
        println("BLANK DUCT")
    end
    core_neep_df = neep_df[!, ["Old AHRI Certified Reference No.","Maximum Capacity 5°F", "Maximum Capacity 17°F", "Maximum Capacity 95°F", "COP at Max. Capacity 47°F", "COP at Max. Capacity 95°F"]]
    max_cap_vector = []
    design_load_served_vector = []
    heating_m_vector = []
    heating_b_vector = []
    cooling_m_vector = []
    cooling_b_vector = []

    # loop over all heat pumps to calculate % design served and max cap at design temp
    for i in 1:size(neep_df,1)
        neep_row = neep_df[i,:]

        cop_row = neep_row[r".*COP at Max\..*"]

        heating_temperatures = Array{Float64}(undef, 0)
        heating_cops = Array{Float64}(undef, 0)
        cooling_temperatures = Array{Float64}(undef, 0)
        cooling_cops = Array{Float64}(undef, 0)

        for colname in names(cop_row)
            cap_index = findfirst("Capacity", colname)
            deg_index = findfirst("°", colname)
            colstr = colname[cap_index[end]+2:deg_index[1]-1]
            temp_num = parse(Float64, colstr)

            if temp_num > 60 # cooling
                push!(cooling_temperatures, temp_num)
                push!(cooling_cops, cop_row[colname])
            else # heating
                push!(heating_temperatures, temp_num)
                push!(heating_cops, cop_row[colname])
            end
        end

        # Heating COP
        heating_cop_df = DataFrame(Temperature=heating_temperatures, COP=heating_cops)

        heating_model = lm(@formula(COP ~ Temperature), heating_cop_df)
        heating_m = coef(heating_model)[2]
        heating_b = coef(heating_model)[1]

        push!(heating_m_vector, heating_m)
        push!(heating_b_vector, heating_b)

        # Cooling COP
        cooling_cop_df = DataFrame(Temperature=cooling_temperatures, COP=cooling_cops)

        cooling_model = lm(@formula(COP ~ Temperature), cooling_cop_df)
        cooling_m = coef(cooling_model)[2]
        cooling_b = coef(cooling_model)[1]

        push!(cooling_m_vector, cooling_m)
        push!(cooling_b_vector, cooling_b)

        # Linear estimation of max capacity at design temp
        # @TODO ask morgan if better to switch to using all heating capacity measures (including 47)
        low = neep_row["Maximum Capacity 5°F"]
        high = neep_row["Maximum Capacity 17°F"]

        slope = (high - low) / (fahrenheit_to_celsius(17.0) - fahrenheit_to_celsius(5.0))

        # Calculate maximum capacity at design temperature using linear interpolation/extrapolation
        max_cap_at_dt = low + (design_temp - fahrenheit_to_celsius(5.0)) * slope
        push!(max_cap_vector, max_cap_at_dt)

        # Percent design load served
        #=
        From NEEP guide: The percent of the home’s design load met by the heat pump operating at
        maximum capacity at the design temperature. For whole-home heating, this
        should be between 90% and 120%
        =#
        design_load_served_pct = (max_cap_at_dt / design_load) * 100
        push!(design_load_served_vector, design_load_served_pct)
    end

    core_neep_df[!, "Maximum Capacity at Design Temperature"] = max_cap_vector
    core_neep_df[!, "Percent Design Load Served"] = design_load_served_vector
    core_neep_df[!, "Design Temp"] .= celsius_to_fahrenheit(design_temp)
    core_neep_df[!, "Design Load"] .= design_load
    core_neep_df[!, "Heating COP M"] = heating_m_vector
    core_neep_df[!, "Heating COP B"] = heating_b_vector
    core_neep_df[!, "Cooling COP M"] = cooling_m_vector
    core_neep_df[!, "Cooling COP B"] = cooling_b_vector

    # Write vector
    CSV.write(joinpath(building_dirname, "in", "design.csv"), core_neep_df)

    hp_thresholds = [90, 100, 115]
    hp_params = Dict()
    for i in 1:size(hp_thresholds,1)
        # Only consider units with percent design load served above each threshold (90/100/115) per NEEP guide
        qualified_hp_df = core_neep_df[core_neep_df[!, "Percent Design Load Served"] .>= hp_thresholds[i], :]
    
        if size(qualified_hp_df, 1) == 0 # no heat pumps can meet the heating load, just pick the biggest
            global qualified_hp_df = copy(core_neep_df)
            sort!(qualified_hp_df, "Percent Design Load Served", rev=true)
        else
            sort!(qualified_hp_df, "Percent Design Load Served")
        end
        # Choose the unit with the lowest percent design load served above threshold
        # Idea is that this balances between cost of unit and load coverage
        hv_max = qualified_hp_df[1, "Maximum Capacity at Design Temperature"] / 3412.14 # convert to kW
        ac_max = qualified_hp_df[1, "Maximum Capacity 95°F"] / 3412.14 # convert to kW
        heating_eff = qualified_hp_df[1, "COP at Max. Capacity 47°F"]
        cooling_eff = qualified_hp_df[1, "COP at Max. Capacity 95°F"]
        pct_design_ld_served = qualified_hp_df[1, "Percent Design Load Served"]
        heating_m = qualified_hp_df[1, "Heating COP M"]
        heating_b = qualified_hp_df[1, "Heating COP B"]
        cooling_m = qualified_hp_df[1, "Cooling COP M"]
        cooling_b = qualified_hp_df[1, "Cooling COP B"]
        
        hp_params[hp_thresholds[i]] = Dict("pHVmx" => hv_max, "pACmx" => ac_max, "pDLserved" => pct_design_ld_served,
                                    "pHVeff" => heating_eff, "pACeff" => cooling_eff, "pHVCOPm" => heating_m, "pHVCOPb" => heating_b,
                                    "pACCOPm" => cooling_m, "pACCOPb" => cooling_b)
    end

    push!(pdls_vector, hp_params[hp_thresholds[end]]["pDLserved"])
    push!(bid_vector, building_id)

    df_chp = CSV.File(joinpath(in_dirname, chp_fname)) |> DataFrame
    df_chp[!, :pCHPmx] .= (max_heat * (1/0.8) * 1.15) #80 AFUE and 15% oversize
    # Linear interpolation of CHP costs based on EIA data
    df_chp[!, :pCHPinv] .= (df_chp.pCHPinv .* 1 .+ ((sqft-1660)/1660))
    CSV.write(joinpath(building_dirname, "in", "chp.csv"), df_chp)

    # HVAC catalog
    df_hvac = CSV.File(joinpath(in_dirname, hvac_fname)) |> DataFrame
    df_hvac[!, :pHVmx] .= max_heat # For non-ASHP sources in catalog. ASHP will be overwritten further down
    df_hvac[!, :pACmx] .= df_hvac[:, :pACmx] # @TODO double check this with Jameson and about max HP sizing
    df_hvac[!, :pDLserved] .= (max_heat / (design_load/3412.0)) * 100
    
    # For every listing in hvac catalog
    df_hvac[!, :pDtemp] .= design_temp
    df_hvac[!, :pDload] .= design_load / 3412.14 # in kW

    # Fill with zeros, only relevant for NEEP CCHP, populated beloow
    df_hvac[!, :pHVCOPm] .= 0.0
    df_hvac[!, :pHVCOPb] .= 0.0
    df_hvac[!, :pACCOPm] .= 0.0
    df_hvac[!, :pACCOPb] .= 0.0

    df_in = CSV.File(joinpath(in_dirname, "in.csv")) |> DataFrame
    df_in[!, :pDtemp] .= design_temp

    df_neep_only = filter("pHVACty" => neep_only, df_hvac) # filter only for the 90/100/120 ASHP
    for i in 1:size(hp_thresholds,1)
        pct = hp_thresholds[i]
        println("PCT")
        println(pct)
        neep_row = df_neep_only[i,:]
        println(neep_row[:pHVACty])

        df_neep_only[i, :pHVmx] = hp_params[pct]["pHVmx"]
        df_neep_only[i, :pACmx] = hp_params[pct]["pACmx"]
        df_neep_only[i, :pDLserved] = hp_params[pct]["pDLserved"]
        df_neep_only[i, :pHVeff] = hp_params[pct]["pHVeff"]
        df_neep_only[i, :pACeff] = hp_params[pct]["pACeff"]
        df_neep_only[i, :pHVCOPm] = hp_params[pct]["pHVCOPm"]
        df_neep_only[i, :pHVCOPb] = hp_params[pct]["pHVCOPb"]
        df_neep_only[i, :pACCOPm] = hp_params[pct]["pACCOPm"]
        df_neep_only[i, :pACCOPb] = hp_params[pct]["pACCOPb"]
    end

    df_hvac[7:9,:] = df_neep_only

    # heat pump costs
    df_ashp_only = filter("pHVACty" => ashp_only, df_hvac) #filter for only ASHP rows, includes NEEP rows
    df_ashp_only[!, :pHVACinv] .= hp_cost(df_ashp_only.pHVmx, sqft, df_ashp_only.pHVeff)
    df_hvac[1:9,:] = df_ashp_only # first six rows only

    CSV.write(joinpath(building_dirname, "in", "hvac.csv"), df_hvac)

    # wh catalog
    # Water heater tank size based on occupants
    df_wh = CSV.File(joinpath(in_dirname, wh_fname)) |> DataFrame
    temp_occ = (20 + row[in_alias * "occupants"] * 10) * (10.0/75) * 2
    println(temp_occ)
    df_wh[df_wh.pWHtank .== 99, :pWHtank] .= trunc(Int, temp_occ)
    df_wh[df_wh.pWHtank .!= 0, :].pWHmx = (df_wh[df_wh.pWHtank .!= 0, :].pWHtank) * 2
    # Assume very roughly that 2x capacity is needed for tankless
    df_wh[df_wh.pWHtank .== 0, :].pWHmx .= minimum(unique(df_wh[df_wh.pWHtank .!= 0, :].pWHmx)) * 2
    CSV.write(joinpath(building_dirname, "in", "wh.csv"), df_wh)

    # sp equipment specification
    heatsource = row[in_alias * "heating_fuel"] * row[in_alias * "hvac_heating_efficiency"]
    coolsource = row[in_alias * "hvac_cooling_efficiency"]
    println("HEAT AND COOL SOURCE")
    println(heatsource)
    println(coolsource)

    df_sp = CSV.File(joinpath(in_dirname, sp_fname)) |> DataFrame
    
    # TODO ask Jameson about how this works with adding different equipment and if it's only first row
    if heatsource in df_chp[:, :pCHPty]
        df_sp[1, :pCHP0] = heatsource
        df_sp[1, :pCHPz0] = 1
    end
    if coolsource in df_hvac[:, :pHVACty]
        df_sp[1, :pHVAC0] = coolsource
        df_sp[1, :pHVACz0] = 1
    end
    if heatsource in df_hvac[:, :pHVACty]
        df_sp[1, :pHVAC0] = heatsource
        df_sp[1, :pHVACz0] = 1
    end
    if row[in_alias * "water_heater_in_unit"] == "Yes"
       df_sp[1, :pWH0] = row[in_alias * "water_heater_efficiency"]
       df_sp[1, :pWHz0] = 1
    end
    CSV.write(joinpath(building_dirname, "in", "sp.csv"), df_sp)

    # topo (thermal equipment topology) TODO- check with Pablo and Jameson about this
    # It's important that topo specifies the same hot air source as the chp or hvac in sp
    # If you're changing things by hand in the input csvs, double check this is the case
    df_topo = CSV.File(joinpath(in_dirname, topo_fname)) |> DataFrame

    if df_sp.pCHPz0[1] > 0
        df_topo[1,1] = df_sp.pCHP0[1]
    end
    if df_sp.pCHPz0[1] == 0
        df_topo[1,1] = df_sp.pHVAC0[1]
    end
    if df_sp.pWHz0[1] > 0
        df_topo[2,1] = df_sp.pWH0[1]
    end

    CSV.write(joinpath(building_dirname, "in", "topo.csv"), df_topo)

    colnames = names(df_bdg1)
    df_bdg1[!, :id] = 1:size(df_bdg1, 1)
    dfl = stack(df_bdg1, colnames)
    df_bdg1_new = unstack(dfl, :variable, :id, :value)
    println(names(df_bdg1_new))
    #println(df_bdg1_new[1:2, :])
    rename!(df_bdg1_new, ["pB", "BDG"])
    #println(df_bdg1_new[1:2, :])
    CSV.write(joinpath(building_dirname, "in", "bdg_i.csv"), df_bdg1_new[2:end, :])

    df_abs = CSV.File(joinpath(in_dirname, "abs.csv")) |> DataFrame
    #df_in = CSV.File(joinpath(in_dirname, "in.csv")) |> DataFrame
    df_pv = CSV.File(joinpath(in_dirname, "pv.csv")) |> DataFrame
    df_bess = CSV.File(joinpath(in_dirname, "bess.csv")) |> DataFrame
    
    CSV.write(joinpath(building_dirname, "in", "abs.csv"), df_abs)
    CSV.write(joinpath(building_dirname, "in", "in.csv"), df_in)
    CSV.write(joinpath(building_dirname, "in", "pv.csv"), df_pv)
    CSV.write(joinpath(building_dirname, "in", "bess.csv"), df_bess)
end

CSV.write(joinpath(wd, "building_directories.csv"), DataFrame(building_directory = building_id_vector))
CSV.write(joinpath(write_dirname_root, "pdl_served.csv"), DataFrame(Building_ID = bid_vector, Pct_Design_Load_Served = pdls_vector, Duct_Type = duct_vector, Heating_Fuel = hs_vector))
