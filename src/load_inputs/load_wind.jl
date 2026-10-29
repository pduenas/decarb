"""
load_wind(path::AbstractString,tm::Dict)

Loads wind.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
tm      dictionary with time series data

returns wind-type inputs in dictionary object
"""

function load_wind(path::AbstractString,tm::Dict)

    # declare dictionary object to store parameters
    wind = Dict()

    path2file = joinpath(path,"wind.csv")

    # load file into dataframe with predefined types
    df_wind = CSV.File(path2file;delim=',',types=Dict{String,Type}("ty"=>String,
                    "mx"=>Float64,"mn"=>Float64,"vmn"=>Float64,"vmx"=>Float64,
                    "fail"=>Float64,"inv"=>Float64,"lc"=>Float64,"fom"=>Float64,
                    "life"=>Float64)) |> DataFrame

    require_valid_columns(df_wind,"wind.csv",["mx","mn","vmn","vmx","fail","inv","lc",
        "fom","life"];key="ty")

    # delete non-existing rows
    filter!(row -> !ismissing(row.ty), df_wind)
    # delete all zero rows
    filter!(row -> !all(x -> x==0, row), df_wind)
    # delete zero-capacity elements
    filter!(row -> !iszero(row.mx), df_wind)

    # delete repeated rows
    unique!(df_wind)

    wind["ty"] = df_wind.ty         # name of wind turbine
    wind["mx"] = df_wind.mx         # maximum output capacity [kW]
    wind["mn"] = df_wind.mn         # minimum output capacity [kW]
    wind["vmn"] = df_wind.vmn       # minimum wind speed [m/s]
    wind["vmx"] = df_wind.vmx       # cutoff wind speed [m/s]
    wind["fail"] = df_wind.fail     # wind turbine failure rate [p.u.]
    wind["inv"] = df_wind.inv       # capital cost [$]
    wind["lc"] = df_wind.lc         # learning curve [%/year]
    wind["fom"] = df_wind.fom       # fixed O&M cost [$/year]
    wind["life"] = df_wind.life     # lifetime [years]

    wind["N"] = size(wind["ty"],1)  # number of wind turbines

    wind["Q"] = calculate_wind_output(tm["P"],tm["Wms"],wind["mx"],
        wind["mn"],wind["vmn"],wind["vmx"],wind["N"])

    return wind

end


function calculate_wind_output(nP,Wms,Qmx,Qmn,SPmn,SPmx,nD)

    Q = zeros(Float64, nP,nD)

    for d=1:nD
        for t=1:nP
            if Wms[t]<=SPmx[d] && Wms[t]>=SPmn[d] && SPmx[d]!=SPmn[d]
                Q[t,d] = (Qmx[d]-Qmn[d])/(SPmx[d]-SPmn[d])*(Wms[t]-SPmn[d])+Qmn[d]
            end
        end
    end

    return Q

end