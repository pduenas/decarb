"""
load_bess(path::AbstractString)

Loads bess.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory

returns bess-type inputs in dictionary object
"""

function load_bess(path::AbstractString)

    # declare dictionary object to store parameters
    bess = Dict()

    path2file = joinpath(path,"bess.csv")

    # load file into dataframe with predefined types
    df_bess = CSV.File(path2file;delim=',',types=[String,Float64,Float64,Float64,
                    Float64,Float64,Float64,Float64,Float64,Float64]) |> DataFrame

    # delete non-existing rows
    filter!(row -> !ismissing(row.ty), df_bess)
    # delete all zero rows
    filter!(row -> !all(x -> x==0, row), df_bess)
    # delete zero-capacity elemenets
    filter!(row -> !iszero(row.mx), df_bess)
    # delete repeated rows
    unique!(df_bess)

    bess["ty"] = df_bess.ty         # name of BESS types
    bess["mx"] = df_bess.mx         # maximum capacity [kWh]
    bess["up"] = df_bess.up         # charging rate [kWh/h]
    bess["dn"] = df_bess.dn         # discharging rate [kWh/h]
    bess["effu"] = df_bess.effu     # charging efficiency [%]
    bess["effd"] = df_bess.effd     # discharging efficiency [%]
    bess["inv"] = df_bess.inv       # capital cost [$]
    bess["lc"] = df_bess.lc         # learning curve [%/year]
    bess["fom"] = df_bess.fom       # fixed O&M cost [$/year]
    bess["life"] = df_bess.life     # lifetime [years]

    bess["N"] = size(bess["ty"],1)  # number of BESS types

    return bess

end