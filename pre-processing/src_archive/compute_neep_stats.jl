using CSV
using DataFrames
using Dates
using Tables
using StatsBase
using Interpolations
using GLM
using Clustering
using RDatasets, Plots

HSPF_MIN = 9

function fahrenheit_to_celsius(f_temp)
    c_temp =(f_temp - 32) * (5.0 / 9.0)
    return c_temp
end

function celsius_to_fahrenheit(c_temp)
    f_temp = c_temp * (9.0 / 5.0) + 32
    return f_temp
end

function hspf_filter(hspf_rating)
    hspf_rating >= HSPF_MIN
end

function ashp_only(hvac_equip)
    typeof(match(r".*ASHP.*", hvac_equip)) != Nothing
end

wd = pwd()
neep_df = CSV.File(joinpath(wd, "neep_catalog_ductless.csv")) |> DataFrame
neep_df = unique(neep_df, ["Outdoor Unit Model", "Indoor Model(s)"])
neep_df = filter("HSPF (Region IV)" => hspf_filter, neep_df) # Filter for only high-efficiency models

core_neep_df = neep_df[!, ["Outdoor Unit Model", "Indoor Model(s)", "Brand Name"]]

heating_cop_m_vector = []
heating_cop_b_vector = []
heating_cop_r2_vector = []
cooling_cop_m_vector = []
cooling_cop_b_vector = []
cooling_cop_r2_vector = []
heating_load_m_vector = []
heating_load_b_vector = []
heating_load_r2_vector = []
cooling_load_m_vector = []
cooling_load_b_vector = []
cooling_load_r2_vector = []

for i in 1:size(neep_df,1)
    neep_row = neep_df[i,:]

    cop_row = neep_row[r".*COP at Max\..*"]

    heating_cop_temperatures = Array{Float64}(undef, 0)
    heating_cops = Array{Float64}(undef, 0)
    cooling_cop_temperatures = Array{Float64}(undef, 0)
    cooling_cops = Array{Float64}(undef, 0)

    for colname in names(cop_row)
        cap_index = findfirst("Capacity", colname)
        deg_index = findfirst("°", colname)
        colstr = colname[cap_index[end]+2:deg_index[1]-1]
        temp_num = parse(Float64, colstr)

        if temp_num > 60 # cooling
            push!(cooling_cop_temperatures, temp_num)
            push!(cooling_cops, cop_row[colname])
        else # heating
            push!(heating_cop_temperatures, temp_num)
            push!(heating_cops, cop_row[colname])
        end
    end

    # Heating COP
    heating_cop_df = DataFrame(Temperature=heating_cop_temperatures, COP=heating_cops)

    heating_cop_model = lm(@formula(COP ~ Temperature), heating_cop_df)
    heating_cop_m = coef(heating_cop_model)[2]
    heating_cop_b = coef(heating_cop_model)[1]
    heating_cop_r2 = r2(heating_cop_model)

    push!(heating_cop_m_vector, heating_cop_m)
    push!(heating_cop_b_vector, heating_cop_b)
    push!(heating_cop_r2_vector, heating_cop_r2)

    # Cooling COP
    cooling_cop_df = DataFrame(Temperature=cooling_cop_temperatures, COP=cooling_cops)

    cooling_cop_model = lm(@formula(COP ~ Temperature), cooling_cop_df)
    cooling_cop_m = coef(cooling_cop_model)[2]
    cooling_cop_b = coef(cooling_cop_model)[1]
    cooling_cop_r2 = r2(cooling_cop_model)

    push!(cooling_cop_m_vector, cooling_cop_m)
    push!(cooling_cop_b_vector, cooling_cop_b)
    push!(cooling_cop_r2_vector, cooling_cop_r2)

    load_row = neep_row[r".*Maximum Capacity.*"]
    load_row = load_row[r"^(?!.*(Optional)).*$"]
    
    #println("LOAD ROW")
    #println(names(load_row))

    heating_load_temperatures = Array{Float64}(undef, 0)
    heating_loads = Array{Float64}(undef, 0)
    cooling_load_temperatures = Array{Float64}(undef, 0)
    cooling_loads = Array{Float64}(undef, 0)

    for colname in names(load_row)
        cap_index = findfirst("Capacity", colname)
        deg_index = findfirst("°", colname)
        colstr = colname[cap_index[end]+2:deg_index[1]-1]
        temp_num = parse(Float64, colstr)

        if temp_num > 60 # cooling
            push!(cooling_load_temperatures, temp_num)
            push!(cooling_loads, load_row[colname])
        else # heating
            push!(heating_load_temperatures, temp_num)
            push!(heating_loads, load_row[colname])
        end
    end

    # Heating Loads
    heating_load_df = DataFrame(Temperature=heating_load_temperatures, Load=heating_loads)

    heating_load_model = lm(@formula(Load ~ Temperature), heating_load_df)
    heating_load_m = coef(heating_load_model)[2]
    heating_load_b = coef(heating_load_model)[1]
    heating_load_r2 = r2(heating_load_model)

    push!(heating_load_m_vector, heating_load_m)
    push!(heating_load_b_vector, heating_load_b)
    push!(heating_load_r2_vector, heating_load_r2)

    # Cooling Loads
    cooling_load_df = DataFrame(Temperature=cooling_cop_temperatures, Load=cooling_loads)

    cooling_load_model = lm(@formula(Load ~ Temperature), cooling_load_df)
    cooling_load_m = coef(cooling_load_model)[2]
    cooling_load_b = coef(cooling_load_model)[1]
    cooling_load_r2 = r2(cooling_load_model)

    push!(cooling_load_m_vector, cooling_load_m)
    push!(cooling_load_b_vector, cooling_load_b)
    push!(cooling_load_r2_vector, cooling_load_r2)
end

# COPs
core_neep_df[!, "Heating COP M"] = heating_cop_m_vector
core_neep_df[!, "Heating COP B"] = heating_cop_b_vector
core_neep_df[!, "Heating COP R2"] = heating_cop_r2_vector
#core_neep_df[!, "Cooling COP M"] = cooling_cop_m_vector
#core_neep_df[!, "Cooling COP B"] = cooling_cop_b_vector
#core_neep_df[!, "Cooling COP R2"] = cooling_cop_r2_vector

# Loads
core_neep_df[!, "Heating Load M"] = heating_load_m_vector
core_neep_df[!, "Heating Load B"] = heating_load_b_vector
core_neep_df[!, "Heating Load R2"] = heating_load_r2_vector
#core_neep_df[!, "Cooling Load M"] = cooling_load_m_vector
#core_neep_df[!, "Cooling Load B"] = cooling_load_b_vector
#core_neep_df[!, "Cooling Load R2"] = cooling_load_r2_vector

heating_ref = 32
cooling_ref = 90

k_means_df = DataFrame()
k_means_df[!, "KM Heating COP at 32F"] = core_neep_df[!, "Heating COP B"] .+ core_neep_df[!, "Heating COP M"] .* heating_ref
k_means_df[!, "KM Heating Load at 32F"] = core_neep_df[!, "Heating Load B"] .+ core_neep_df[!, "Heating Load M"] .* heating_ref
#k_means_df[!, "KM Cooling COP at 90F"] = core_neep_df[!, "Cooling COP B"] .+ core_neep_df[!, "Cooling COP M"] .* cooling_ref
#k_means_df[!, "KM Cooling Load at 32F"] = core_neep_df[!, "Cooling Load B"] .+ core_neep_df[!, "Cooling Load M"] .* cooling_ref

M = transpose(Matrix(k_means_df))
print(M)

R = kmeans(M, 6; maxiter=200, display=:iter)

a = assignments(R) # get the assignments of points to clusters
println(assignments)
c = counts(R) # get the cluster sizes
centers = R.centers # get the cluster centers

# plot with the point color mapped to the assigned cluster index
scatter(k_means_df[!, "KM Heating COP at 32F"], k_means_df[!, "KM Heating Load at 32F"], marker_z=R.assignments, color=:lightrainbow, legend=false)

core_neep_df[!, "Cluster"] .= a
core_neep_df[!, "Heating COP at 32F"] .= k_means_df[!, "KM Heating COP at 32F"]
core_neep_df[!, "Heating Load at 32F"] .= k_means_df[!, "KM Heating Load at 32F"]

CSV.write(joinpath(wd, "neep_stats.csv"), core_neep_df)
