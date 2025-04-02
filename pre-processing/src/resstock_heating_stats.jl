using CSV
using DataFrames
using Dates
using Plots

in_alias = "build_existing_model."
template_dirname = "in_templates"
wd = pwd()
resstock_dirname = joinpath(wd, "in", "resstock_files") # change this
fname_summary = "results-Baseline_jan.csv" # metadata.csv in R code

df_summary = CSV.File(joinpath(resstock_dirname, fname_summary)) |> DataFrame
df_summary = filter(:completed_status => cs -> cs == "Success", df_summary)
df_summary = filter((in_alias * "geometry_building_type_acs") => bt -> bt == "Single-Family Detached", df_summary)

max_vector = []
bid_vector = []
for i in 1:size(df_summary, 1)
    #println(i)
    row = df_summary[i,:]
    building_id = row["building_id"]
    fname_time_series = "results_timeseries.csv"
    run_path = joinpath(resstock_dirname, "400 samples single-family MA 2018 weather", "ResStock_outputs", "run" * string(building_id), "run")
    df_ts = CSV.File(joinpath(run_path, fname_time_series)) |> DataFrame
    df_ts = df_ts[2:end, :] # omit first row with units

    try
        heating_vector = df_ts[!, "Load: Heating: Delivered"] # delivered kbtu
        heating_vector = parse.(Float64, heating_vector)
        max_heat = maximum(heating_vector) # in kBtu
        push!(max_vector, max_heat)
        push!(bid_vector, building_id)
    catch
        println(building_id)
    end
end

df_out = DataFrame(Building_ID = bid_vector, Peak_Heating_Load = max_vector)
CSV.write(joinpath(resstock_dirname, "peak_heating_loads.csv"), df_out)
histogram(df_out.Peak_Heating_Load)