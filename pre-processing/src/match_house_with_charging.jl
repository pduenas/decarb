using CSV
using DataFrames
using Dates
using StatsBase

# The goal of this file is to select a vehicle charging profile 
# to match with a ResStock house profile
# The vehicle profile is chosen by matching demographic characteristics
# (e.g. number of occupants & income) of the ResStock home with NHTS data

MAX_CHG_CAP = 7.7 # maximum hourly kWh on 40A breaker

# Returns filename in correct format given day, # of occupants, and income signifiers
function get_sample_str(daytype, occtype, inctype)
    return "df_" * daytype * "_" * occtype * "_" * inctype * ".csv"
end

# Given vehicle number, returns departure string in the column header format
function get_departure_header(vehicle_num)
    return "Vehicle " * string(vehicle_num) * " Departure"
end

# Given vehicle number, returns arrival string in the column header format
function get_arrival_header(vehicle_num)
    return "Vehicle " * string(vehicle_num) * " Arrival"
end

# Given vehicle number, returns VMT string in the column header format
function get_miles_header(vehicle_num)
    return "Vehicle " * string(vehicle_num) * " Miles"
end

# Given miles driven, returns equivalent kWh (currently not differentiated for season)
# @TODO: currently a hack, will use specific vehicle stats later
function miles_to_kwh(miles)
    return miles/3.1
end

# Prefix for ResStock column headers
in_alias = "in."

# Income thresholds (currently arbitrary) @TODO come up with better thresholds
LI_MAX = 34999
MI_MAX = 124999

# All of this will be in a loop to capture all the buildings in the sample
wd = pwd()
resstock_dirname = joinpath(wd, "in", "resstock_files")
metadata_fname = "metadata_nhts.csv"

# Read in metadata file into dataframe
df_summary = CSV.File(joinpath(resstock_dirname, metadata_fname)) |> DataFrame

occupants = df_summary[1, in_alias * "occupants"]

# Convert number of occupants into proper occupant signifier format for file selection
occupants_str = ""
if occupants == 1
    global occupants_str = "1o"
elseif occupants == 2
    global occupants_str = "2o"
else
    global occupants_str = "3o"
end

# Convert income into proper income signifier format for file selection
hh_income = df_summary[1, in_alias * "income"]

derived_income = 0
dash_index = findfirst("-", hh_income)
if typeof(dash_index) == Nothing
    plus_index = findfirst("+", hh_income)
    if typeof(plus_index) == Nothing
        derived_income = 10000
    else
        global derived_income = parse(Float64, hh_income[1:(plus_index[1]-1)])
    end
else
    low = parse(Float64, hh_income[1:(dash_index[1]-1)])
    high = parse(Float64, hh_income[(dash_index[1]+1):end])
    global derived_income = (low + high) / 2.0
end

income_str = ""
if derived_income < LI_MAX
    global income_str = "li"
elseif derived_income < MI_MAX
    global income_str = "mi"
else
    global income_str = "hi"
end

# Read in weekday and weekend dataframes
df_wd_fname = get_sample_str("wd", occupants_str, income_str)
println(df_wd_fname)

df_we_fname = get_sample_str("we", occupants_str, income_str)
println(df_we_fname)

df_wd = CSV.File(joinpath(wd, "in", "nhts_data", "out", df_wd_fname)) |> DataFrame
df_wd = df_wd[df_wd.Utilized_Vehicle_Count .< 3, :]

# Randomly sample based on household weights for weekday and weekend
wd_wts = df_wd[:, :HH_Weight]
wd_keys = df_wd[:, :Key]

wd_samp = sample(wd_keys, Weights(wd_wts))
println(wd_samp) # prints selected house key

# Now that we have our samples, we need to produce a charging profile
# The samples are arrival and departure times for each vehicle, plus total miles driven
# during the surveyed day

# Weekday profile generation
df_wd_match = df_wd[df_wd.Key .== wd_samp, :]
utilized_vehicles_wd = df_wd_match[1, :Utilized_Vehicle_Count] # check that utilized vehicles not zero

wd_chg_vector = zeros(utilized_vehicles_wd, 24)
println("Vehicles: " * string(utilized_vehicles_wd))

# Number of vehicles matches weekend and weekday
df_we = CSV.File(joinpath(wd, "in", "nhts_data", "out", df_we_fname)) |> DataFrame
df_we = df_we[df_we.Utilized_Vehicle_Count .== utilized_vehicles_wd, :] # Need the number of vehicles to match
vehicle_counter = utilized_vehicles_wd - 1
while size(df_we, 1) == 0
    println("CAN'T MATCH VEHICLE COUNT WEEKDAY AND WEEKEND")
    global df_we = df_we[df_we.Utilized_Vehicle_Count == vehicle_counter, :] # Need the number of vehicles to match
    global vehicle_counter -= 1
end
we_wts = df_we[:, :HH_Weight]
we_keys = df_we[:, :Key]

we_samp = sample(we_keys, Weights(we_wts))
println(we_samp) # prints house key

# Weekend profile generation
df_we_match = df_we[df_we.Key .== we_samp, :]
utilized_vehicles_we = df_we_match[1, :Utilized_Vehicle_Count]
we_chg_vector = zeros(utilized_vehicles_wd, 24) # weekday to match # of vehicles
println("Vehicles: " * string(utilized_vehicles_we))

# Merged hourly
# Parse each vehicle, adding the equivalent kwh to the arrival hour
for i in 1:utilized_vehicles_wd
    dh = get_departure_header(i)
    ah = get_arrival_header(i)
    mh = get_miles_header(i)

    miles = df_wd_match[1, mh]
    dep = Dates.DateTime(df_wd_match[1, dh])
    arr = Dates.DateTime(df_wd_match[1, ah])

    kwh = miles_to_kwh(miles)
    arr_hr = Dates.hour(arr)
    dep_hr = Dates.hour(dep)
    arr_min = Dates.minute(arr)
    dep_min = Dates.minute(dep)

    # Round up to next hour if it's over the 30-minute mark (implicitly round down for < 30)
    if arr_min >= 30
        global arr_hr += 1
        global arr_hr %= 24
    end
    if dep_min >= 30
        global dep_hr += 1
        global dep_hr %= 24
    end
    
    wd_chg_vector[i, arr_hr+1] += kwh

    # Check to make sure that there are sufficient hours to charge
    total_chg_window = 24 - arr_hr + dep_hr
    max_kwh = total_chg_window * MAX_CHG_CAP
    if kwh > max_kwh
        println("NOT ENOUGH TIME TO GET FULL CHARGE")
    end
end

for i in 1:utilized_vehicles_we
    dh = get_departure_header(i)
    ah = get_arrival_header(i)
    mh = get_miles_header(i)

    miles = df_we_match[1, mh]
    dep = Dates.DateTime(df_we_match[1, dh])
    arr = Dates.DateTime(df_we_match[1, ah])

    kwh = miles_to_kwh(miles)
    arr_hr = Dates.hour(arr)
    dep_hr = Dates.hour(dep)
    arr_min = Dates.minute(arr)
    dep_min = Dates.minute(dep)

    # Round up to next hour if it's over the 30-minute mark (implicitly round down for < 30)
    if arr_min >= 30
        global arr_hr += 1
        global arr_hr %= 24
    end
    if dep_min >= 30
        global dep_hr += 1
        global dep_hr %= 24
    end
    
    we_chg_vector[i, arr_hr+1] += kwh

    # Check to make sure that there are sufficient hours to charge
    total_chg_window = 24 - arr_hr + dep_hr
    max_kwh = total_chg_window * MAX_CHG_CAP
    if kwh > max_kwh
        println("NOT ENOUGH TIME TO GET FULL CHARGE")
    end
end

println("Weekday and weekend charging vectors")
println(wd_chg_vector)
println(we_chg_vector)

summary_df = DataFrame()
for i in 1:utilized_vehicles_wd
    colname_wd = "Vehicle " * string(i) * " Weekday" 
    colname_we = "Vehicle " * string(i) * " Weekend"
    summary_df[!, colname_wd] = wd_chg_vector[i,:]
    summary_df[!, colname_we] = we_chg_vector[i,:]
end

CSV.write(joinpath(resstock_dirname, "summary_profile.csv"), summary_df)

status_vector = Array{Array{Int64}}(undef, utilized_vehicles_wd)
dt_vector  = []
dow_vector = []
global this_dt = DateTime(2018,1,1,0)
while Dates.year(this_dt) < 2019
    hr = Dates.hour(this_dt)
    push!(dt_vector, this_dt)
    push!(dow_vector, Dates.dayofweek(this_dt))

    comp_dt = this_dt - Dates.Hour(4)
    comp_hr = Dates.hour(comp_dt) # offset by 4 hr to get correct time

    for i in 1:utilized_vehicles_wd
        if !isassigned(status_vector, i)
            status_vector[i] = []
        end
        dh = get_departure_header(i)
        ah = get_arrival_header(i)

        if Dates.dayofweek(comp_dt) <= 5 # the 4-hr offset will allow us to pick the correct times, since Saturday 0-4am should be part of the Friday
            dep = Dates.DateTime(df_wd_match[1, dh])
            arr = Dates.DateTime(df_wd_match[1, ah])

            dep = dep - Dates.Hour(4)
            arr = arr - Dates.Hour(4)
    
            arr_hr = Dates.hour(arr)
            dep_hr = Dates.hour(dep)
            arr_min = Dates.minute(arr)
            dep_min = Dates.minute(dep)

            if arr_min >= 30
                global arr_hr += 1
            end
            if dep_min >= 30
                global dep_hr += 1
            end

            if comp_hr < dep_hr || comp_hr > arr_hr
                push!(status_vector[i], 1)
            else
                push!(status_vector[i], 0)
            end
        else
            dep = Dates.DateTime(df_we_match[1, dh])
            arr = Dates.DateTime(df_we_match[1, ah])

            dep = dep - Dates.Hour(4)
            arr = arr - Dates.Hour(4)
    
            arr_hr = Dates.hour(arr)
            dep_hr = Dates.hour(dep)
            arr_min = Dates.minute(arr)
            dep_min = Dates.minute(dep)

            if arr_min >= 30
                global arr_hr += 1
            end
            if dep_min >= 30
                global dep_hr += 1
            end

            if comp_hr < dep_hr || comp_hr > arr_hr
                push!(status_vector[i], 1)
            else
                push!(status_vector[i], 0)
            end
        end
    end
    global this_dt = this_dt + Dates.Hour(1)
end

status_df = DataFrame(Datetime = dt_vector, Day_of_Week = dow_vector)
for i in 1:utilized_vehicles_wd
    colname = "Vehicle " * string(i)
    status_df[!, colname] = status_vector[i]
end
status_df[!, "All Vehicles"] = sum(status_vector)

CSV.write(joinpath(resstock_dirname, "status_8760.csv"), status_df)

# At this point have the kWh requirement by hour
# Iterate over the full year (hourly) and populate the charging profile

# Start at first hour of the year
global this_dt = DateTime(2018,1,1,0)

dt_vector = []
chg_vector = Array{Array{Float64}}(undef, utilized_vehicles_wd)
chg_req_remaining = zeros(utilized_vehicles_wd) # represents currently unmet charging need

while Dates.year(this_dt) < 2019
    hr = Dates.hour(this_dt)
    push!(dt_vector, this_dt)
    
    for i in 1:utilized_vehicles_wd
        if !isassigned(chg_vector, i)
            chg_vector[i] = []
        end

        # check if weekend or weekday
        if Dates.dayofweek(this_dt) <= 5
            new_chg_required = wd_chg_vector[i, hr+1] # @TODO convert miles to kWh here based on the date (winter and summer different efficiency)
            chg_req_remaining[i] += new_chg_required

            if chg_req_remaining[i] > 0
                delivered_chg = MAX_CHG_CAP
                if delivered_chg > chg_req_remaining[i]
                    global delivered_chg = chg_req_remaining[i]
                end
                push!(chg_vector[i], delivered_chg)
                global chg_req_remaining[i] -= delivered_chg
            else
                push!(chg_vector[i], 0)
            end
        else
            new_chg_required = we_chg_vector[i, hr+1] # @TODO convert miles to kWh here based on the date
            chg_req_remaining[i] += new_chg_required

            if chg_req_remaining[i] > 0
                delivered_chg = MAX_CHG_CAP
                if delivered_chg > chg_req_remaining[i]
                    global delivered_chg = chg_req_remaining[i]
                end
                push!(chg_vector[i], delivered_chg)
                global chg_req_remaining[i] -= delivered_chg
            else
                push!(chg_vector[i], 0)
            end
        end
    end
    global this_dt = this_dt + Dates.Hour(1)
end

println("chg vector size")
println(size(chg_vector[1],1))
println(size(dt_vector,1))

# Write hourly charging profile to dataframe (will be in a loop using house id for folder)
write_df = DataFrame(Datetime = dt_vector)
for i in 1:utilized_vehicles_wd
    colname = "Consumption: Vehicle " * string(i)
    write_df[!, colname] = chg_vector[i]
end
write_df[!, "Total Consumption"] = sum(chg_vector)
CSV.write(joinpath(resstock_dirname, "hourly_charging_profile.csv"), write_df)