using CSV
using DataFrames
using Dates

# The goal of this module is to parse the NHTS survey data to produce
# pages to sample from to generate charging profiles

# Generic filter to match specified weekday
function weekday_filter(is_wd, target)
    return is_wd == target
end

# Generic filter for income cutoffs
function income_filter(segment, flag)
    if segment == "low"
        return flag < 4
    elseif segment == "middle"
        return flag > 4 && flag < 8
    elseif segment == "high"
        return flag >= 8
    else
        error("Invalid income flag")
    end
end

# Filter for occupant count; round down to 3 adults
function adult_filter(target, num_occupants)
    na = num_occupants
    if num_occupants > 3
        global na = 3
    end
    return target == na
end

# Convert a string in the format 0000 to date format
function time_to_datetime(hrmin)
    parse_str = hrmin
    if length(hrmin) == 3
        parse_str = "0" * hrmin
    elseif length(hrmin) == 2
        parse_str = "00" * hrmin
    elseif length(hrmin) == 1
        parse_str = "000" * hrmin
    end

    ps_hr = parse(Int64,parse_str[1:2])
    ps_min = parse(Int64,parse_str[3:4])

    ps_date = DateTime(2018,1,1,ps_hr,ps_min,0)
    return ps_date
end

# Convert a string in the format 0000 to total number of minutes
# Used for purposes of time comparison
function time_to_minutes(hrmin)
    parse_str = hrmin
    if length(hrmin) == 3
        parse_str = "0" * hrmin
    elseif length(hrmin) == 2
        parse_str = "00" * hrmin
    elseif length(hrmin) == 1
        parse_str = "000" * hrmin
    end
    try
        hr = parse(Int64,parse_str[1:2])
        if hr >= 0 && hr < 4 # Survey day starts at 4am
            hr += 24
        end
        hr -= 4
        min = parse(Int64,parse_str[3:4])
        return hr * 60 + min
    catch e
        println(e)
        println(parse_str)
        return 0
    end
end

#=
wd = pwd()
println("\u2328  type working directory or press <Enter> for ", wd)
dir_name = readline(stdin)
dir_name=="" ? dir_name=joinpath(wd,"in") : isdir(dir_name)==false ? error("\u26A0 ", dir_name," is not a directory.\n") :  println("\u21B3 is now the working directory.\n")
=#
dir_name = joinpath(pwd(), "in", "nhts_data")

println(dir_name)
#println("Type name of trip file")
#fname = readline(stdin)
#fname == "" ? error("$(fname) must not be blank") : println("Reading $(fname) into DataFrame")
fname = "nc_trips.csv"

a0 = time()			# elapsed time
df = CSV.File(joinpath(dir_name, fname);types=[Int64,Int64,Int64,String,String,Int64,Float64,Int64,Int64,Int64,Int64,Int64,Int64,Int64,String,Int64,Float64,Int64,Int64,String,Int64,Int64,Int64,Int64,Int64,String,Int64,Int64,String,Int64,Int64,Int64,Float64,Int64,String,Int64,Int64,Float64,Int64,Float64,Int64,String,String]) |> DataFrame
a1 = time()
println("   \u29D6 elapsed time ... ", round(a1-a0; digits=2), " seconds.\n")

#println("Type name of household file")
#hh_fname = readline(stdin)
#hh_fname == "" ? error("$(hh_fname) must not be blank") : println("Reading $(fname) into DataFrame")
hh_fname = "hhpub.csv"
df_hh = CSV.File(joinpath(dir_name, hh_fname)) |> DataFrame

#mkdir(joinpath(dir_name, "out"))

#=
vmt_dict = Dict()

for i in 1:size(df,1)
    row = df[i,:]
    house_id = row[:HOUSEID]
    vehicle_id = row[:VEHID]
    merged_id = string(house_id) * "-" * string(vehicle_id)
    vmt_mile = row[:VMT_MILE]
    trip_num = row[:TDTRPNUM]
    if !haskey(vmt_dict, merged_id)
        vmt_dict[merged_id] = Dict("miles"=>0, "trip_count"=>0)
    end
    vmt_dict[merged_id]["miles"] += vmt_mile
    vmt_dict[merged_id]["trip_count"] += 1
end
=#

trip_dict = Dict{Int64, Any}()
max_time = "0359"
min_time = "0400"

max_vehicles = 0

# Iterate through each survey response
for i in 1:size(df,1)
    row = df[i,:]

    origin_purpose = row[:WHYFROM]
    destination_purpose = row[:WHYTO]

    trip_start = row[:STRTTIME]
    trip_end = row[:ENDTIME]

    day_of_week = row[:TRAVDAY]
    monthyear = row[:TDAYDATE]

    house_id = row[:HOUSEID]
    vehicle_id = row[:VEHID]

    vmt_mile = row[:TRPMILES]

    num_occupants = row[:HHSIZE]
    vehicle_count = row[:HHVEHCNT]
    hh_income = row[:HHFAMINC]

    if vehicle_count > max_vehicles
        global max_vehicles = vehicle_count
    end

    is_weekday = true
    if day_of_week == 1 || day_of_week == 7
        is_weekday = false
    end

    #= Use these for estimates involving numbers of trips or miles of travel, 
    for example, number of vehicle trips by trip purpose =#
    trip_weight = row[:WTTRDFIN]
    hh_weight = df_hh[df_hh.HOUSEID .== house_id, :].WTHHFIN[1] # find weight in hh file

    #=
    if i == 1
        println(typeof(origin_purpose))
        println(typeof(destination_purpose))
        println(typeof(trip_start))
        println(typeof(trip_end))
        println(typeof(day_of_week))
        println(typeof(monthyear))
        println(typeof(house_id))
        println(typeof(vehicle_id))
        println(typeof(vmt_mile))
    end
    =#

    year = monthyear[1:4]
    month = monthyear[5:6]

    #merged_id = string(house_id) * "-" * string(vehicle_id)

    # Catches error if number of vehicle exceeds total number of vehicles for this household
    if vehicle_id > vehicle_count
        continue
    end

    # If this is a new house, set up the dictionary item
    if !haskey(trip_dict, house_id)
        trip_dict[house_id] = Dict("utilized_vehicles"=>0, "hh_weight"=>hh_weight, 
        "is_weekday"=>is_weekday, "num_vehicles"=>vehicle_count, "hh_income"=>hh_income, 
        "num_occupants"=>num_occupants, "vehicles"=>Array{Dict{String,Any}}(undef, vehicle_count))
    end

    # If this is a new vehicle within the household, set up the dictionary item
    if !isassigned(trip_dict[house_id]["vehicles"], vehicle_id)
        trip_dict[house_id]["utilized_vehicles"] += 1
        trip_dict[house_id]["vehicles"][vehicle_id] = Dict{String,Any}("daily_miles"=>0, "has_departure"=>false, "has_arrival"=>false, "earliest_departure"=>max_time, "latest_arrival"=>min_time)
    end

    # Increment miles driven in this trip
    trip_dict[house_id]["vehicles"][vehicle_id]["daily_miles"] += vmt_mile
    
    # Calculate earliest leaving time from home (1 and 2 are home-based)
    if origin_purpose == 1 || origin_purpose == 2
        trip_dict[house_id]["vehicles"][vehicle_id]["has_departure"] = true
        if time_to_minutes(trip_start) < time_to_minutes(trip_dict[house_id]["vehicles"][vehicle_id]["earliest_departure"])
            trip_dict[house_id]["vehicles"][vehicle_id]["earliest_departure"] = trip_start
        end
    end

    # Calculate latest arrival time to home (1 and 2 are home-based)
    if destination_purpose == 1 || destination_purpose == 2
        trip_dict[house_id]["vehicles"][vehicle_id]["has_arrival"] = true
        try
            if time_to_minutes(trip_end) > time_to_minutes(trip_dict[house_id]["vehicles"][vehicle_id]["latest_arrival"])
                trip_dict[house_id]["vehicles"][vehicle_id]["latest_arrival"] = trip_end
            end
        catch e
            println("IN CATCH BLOCK")
            println(e)
            println(trip_end)
            println("OUT OF CATCH BLOCK")
        end
    end
end

a2 = time()
println("   \u29D6 elapsed time ... ", round(a2-a1; digits=2), " seconds.\n")


# Calculate the maximum number of utilized vehicles across all surveyed households
max_utilized_vehicles = 0
for k in keys(trip_dict)
    if trip_dict[k]["utilized_vehicles"] > max_utilized_vehicles
        global max_utilized_vehicles = trip_dict[k]["utilized_vehicles"]
    end
end

#for i in 1:max_utilized_vehicles
# Only consider households with i number of utilized vehicles
#this_dict = filter(((k,v),) -> v["utilized_vehicles"] == i, trip_dict)

key_vector = []
veh_count_vector = []
utilized_count_vector = []
num_occupants_vector = []
hh_income_vector = []
weekday_vector = []
hh_weight_vector = []
    
# Each vehicle has departure, arrival, and miles driven
vehicle_vector = Array{Dict{String,Any}}(undef, max_utilized_vehicles)

for j in 1:size(vehicle_vector,1)
    vehicle_vector[j] = Dict{String,Any}("departure_vector"=>[], "arrival_vector"=>[], "miles_vector"=>[])
end

skipped_count = 0

# Loop over all vehicles and populate arrival, departure, and miles vectors
for k in keys(trip_dict)
    vv_index = 1
    for p in 1:size(trip_dict[k]["vehicles"],1)
        # Skip unutilized vehicles
        if !isassigned(trip_dict[k]["vehicles"], p)
            continue
        end

        veh_obj = trip_dict[k]["vehicles"][p]

        if veh_obj["has_arrival"] == false || veh_obj["has_departure"] == false
            trip_dict[k]["utilized_vehicles"] -= 1 # need to do this so that the number of utilized vehicles matches the columns
            continue
        end

        if time_to_minutes(veh_obj["earliest_departure"]) > time_to_minutes(veh_obj["latest_arrival"])
            trip_dict[k]["utilized_vehicles"] -= 1 # need to do this so that the number of utilized vehicles matches the columns
            global skipped_count += 1
            continue
        end

        ed = veh_obj["earliest_departure"]
        la = veh_obj["latest_arrival"]

        ed_date = time_to_datetime(ed)
        la_date = time_to_datetime(la)
        miles = veh_obj["daily_miles"]

        push!(vehicle_vector[vv_index]["departure_vector"], ed_date)
        push!(vehicle_vector[vv_index]["arrival_vector"], la_date)
        push!(vehicle_vector[vv_index]["miles_vector"], miles)
        vv_index += 1
    end

    # Need to push empty so that columns remain aligned
    for j in vv_index:max_utilized_vehicles
        push!(vehicle_vector[vv_index]["departure_vector"], "")
        push!(vehicle_vector[vv_index]["arrival_vector"], "")
        push!(vehicle_vector[vv_index]["miles_vector"], "")
        vv_index += 1
    end
    
    # Populate vectors for non-trip information
    veh_count = trip_dict[k]["num_vehicles"]
    utilized_count = trip_dict[k]["utilized_vehicles"]
    num_occupants = trip_dict[k]["num_occupants"]
    hh_income = trip_dict[k]["hh_income"]
    is_weekday = trip_dict[k]["is_weekday"]
    hh_weight = trip_dict[k]["hh_weight"]

    push!(key_vector, k)
    push!(veh_count_vector, veh_count)
    push!(utilized_count_vector, utilized_count)
    push!(num_occupants_vector, num_occupants)
    push!(hh_income_vector, hh_income)
    push!(weekday_vector, is_weekday)
    push!(hh_weight_vector, hh_weight)
end

# Write output to CSV file
out_df = DataFrame(Key = key_vector, Is_Weekday = weekday_vector, HH_Weight = hh_weight_vector, Vehicle_Count = veh_count_vector, Utilized_Vehicle_Count = utilized_count_vector, HH_Income = hh_income_vector, Num_Occupants = num_occupants_vector)
for j in 1:max_utilized_vehicles
    departure_header = "Vehicle " * string(j) * " Departure"
    arrival_header = "Vehicle " * string(j) * " Arrival"
    miles_header = "Vehicle " * string(j) * " Miles"
    out_df[!, departure_header] = vehicle_vector[j]["departure_vector"]
    out_df[!, arrival_header] = vehicle_vector[j]["arrival_vector"]
    out_df[!, miles_header] = vehicle_vector[j]["miles_vector"]
end

out_df = out_df[out_df.Utilized_Vehicle_Count .> 0, :]

println("Skipped Count")
println(skipped_count)

out_fname = "nhts_ma_out.csv"
CSV.write(joinpath(dir_name, "out", out_fname), out_df)

# Build individual pages for random sampling
# weekend/weekday
weekday_df = filter(row -> row.Is_Weekday == true, out_df)
weekend_df = filter(row -> row.Is_Weekday == false, out_df)

# Low income
#df_wd_1a_li = filter(row -> row.Is_Weekday .== true && row.HH_Income .< 4 && row.Num_Occupants .== 1, out_df)

df_wd_1o_li = out_df[(out_df.Is_Weekday .== true) .& (out_df.HH_Income .< 4) .& (out_df.Num_Occupants .== 1), :]
df_we_1o_li = out_df[(out_df.Is_Weekday .== false) .& (out_df.HH_Income .< 4) .& (out_df.Num_Occupants .== 1), :]

df_wd_2o_li = out_df[(out_df.Is_Weekday .== true) .& (out_df.HH_Income .< 4) .& (out_df.Num_Occupants .== 2), :]
df_we_2o_li = out_df[(out_df.Is_Weekday .== false) .& (out_df.HH_Income .< 4) .& (out_df.Num_Occupants .== 2), :]

df_wd_3o_li = out_df[(out_df.Is_Weekday .== true) .& (out_df.HH_Income .< 4) .& (out_df.Num_Occupants .>= 3), :]
df_we_3o_li = out_df[(out_df.Is_Weekday .== false) .& (out_df.HH_Income .< 4) .& (out_df.Num_Occupants .>= 3), :]

#CSV.write(joinpath(dir_name, "out", "df_wd_1o_li.csv"), df_wd_1o_li)
#CSV.write(joinpath(dir_name, "out", "df_we_1o_li.csv"), df_we_1o_li)
#CSV.write(joinpath(dir_name, "out", "df_wd_2o_li.csv"), df_wd_2o_li)
#CSV.write(joinpath(dir_name, "out", "df_we_2o_li.csv"), df_we_2o_li)
#CSV.write(joinpath(dir_name, "out", "df_wd_3o_li.csv"), df_wd_3o_li)
#CSV.write(joinpath(dir_name, "out", "df_we_3o_li.csv"), df_we_3o_li)

## Moderate Income

df_wd_1o_mi = out_df[(out_df.Is_Weekday .== true) .& (out_df.HH_Income .>= 4 .& out_df.HH_Income .< 8) .& (out_df.Num_Occupants .== 1), :]
df_we_1o_mi = out_df[(out_df.Is_Weekday .== false) .& (out_df.HH_Income .>= 4 .& out_df.HH_Income .< 8) .& (out_df.Num_Occupants .== 1), :]

df_wd_2o_mi = out_df[(out_df.Is_Weekday .== true) .& (out_df.HH_Income .>= 4 .& out_df.HH_Income .< 8) .& (out_df.Num_Occupants .== 2), :]
df_we_2o_mi = out_df[(out_df.Is_Weekday .== false) .& (out_df.HH_Income.>= 4 .& out_df.HH_Income .< 8) .& (out_df.Num_Occupants .== 2), :]

df_wd_3o_mi = out_df[(out_df.Is_Weekday .== true) .& (out_df.HH_Income .>= 4 .& out_df.HH_Income .< 8) .& (out_df.Num_Occupants .>= 3), :]
df_we_3o_mi = out_df[(out_df.Is_Weekday .== false) .& (out_df.HH_Income .>= 4 .& out_df.HH_Income .< 8) .& (out_df.Num_Occupants .>= 3), :]

#CSV.write(joinpath(dir_name, "out", "df_wd_1o_mi.csv"), df_wd_1o_mi)
#CSV.write(joinpath(dir_name, "out", "df_we_1o_mi.csv"), df_we_1o_mi)
#CSV.write(joinpath(dir_name, "out", "df_wd_2o_mi.csv"), df_wd_2o_mi)
#CSV.write(joinpath(dir_name, "out", "df_we_2o_mi.csv"), df_we_2o_mi)
#CSV.write(joinpath(dir_name, "out", "df_wd_3o_mi.csv"), df_wd_3o_mi)
#CSV.write(joinpath(dir_name, "out", "df_we_3o_mi.csv"), df_we_3o_mi)

## High Income

df_wd_1o_hi = out_df[(out_df.Is_Weekday .== true) .& (out_df.HH_Income .>= 8) .& (out_df.Num_Occupants .== 1), :]
df_we_1o_hi = out_df[(out_df.Is_Weekday .== false) .& (out_df.HH_Income .>= 8) .& (out_df.Num_Occupants .== 1), :]

df_wd_2o_hi = out_df[(out_df.Is_Weekday .== true) .& (out_df.HH_Income .>= 8) .& (out_df.Num_Occupants .== 2), :]
df_we_2o_hi = out_df[(out_df.Is_Weekday .== false) .& (out_df.HH_Income .>= 8) .& (out_df.Num_Occupants .== 2), :]

df_wd_3o_hi = out_df[(out_df.Is_Weekday .== true) .& (out_df.HH_Income .>= 8) .& (out_df.Num_Occupants .>= 3), :]
df_we_3o_hi = out_df[(out_df.Is_Weekday .== false) .& (out_df.HH_Income .>= 8) .& (out_df.Num_Occupants .>= 3), :]

#CSV.write(joinpath(dir_name, "out", "df_wd_1o_hi.csv"), df_wd_1o_hi)
#CSV.write(joinpath(dir_name, "out", "df_we_1o_hi.csv"), df_we_1o_hi)
#CSV.write(joinpath(dir_name, "out", "df_wd_2o_hi.csv"), df_wd_2o_hi)
#CSV.write(joinpath(dir_name, "out", "df_we_2o_hi.csv"), df_we_2o_hi)
#CSV.write(joinpath(dir_name, "out", "df_wd_3o_hi.csv"), df_wd_3o_hi)
#CSV.write(joinpath(dir_name, "out", "df_we_3o_hi.csv"), df_we_3o_hi)

# Print all files into their own file

#end


#=
a3 = time()
println("   \u29D6 elapsed time ... ", round(a3-a2; digits=2), " seconds.\n")

time_df = DataFrame(Key = key_vector, Home_Departure = departure_vector, Home_Arrival = arrival_vector, Daily_Miles = miles_vector)
CSV.write(joinpath(dir_name, "veh_time.csv"), time_df)

a4 = time()
println("   \u29D6 elapsed time ... ", round(a4-a3; digits=2), " seconds.\n")

output_keys_vector = []
output_miles_vector = []

for k in keys(vmt_dict)
    push!(output_keys_vector, k)
    push!(output_miles_vector, vmt_dict[k]["miles"])
end

df_out = DataFrame(HouseID_VehicleID = output_keys_vector, VMT = output_miles_vector)

CSV.write(joinpath(dir_name, "total_vmt.csv"), df_out)
=#