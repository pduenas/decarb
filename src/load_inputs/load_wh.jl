"""
load_wh(path::AbstractString,sp::Dict,bdg::Dict)

Loads wh.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
sp      dictionary with equipment selection
bdg     dictionary with building data

returns wh-type inputs in dictionary object
"""

function load_wh(path::AbstractString,sp::Dict,bdg::Dict)

    # declare dictionary object to store parameters
    wh = Dict()

    path2file = joinpath(path,"wh.csv")

    # load file into dataframe with predefined types
    df_wh = CSV.File(path2file;delim=',',types=[String,Float64,Float64,String,Float64,
                    Float64,Float64,Float64,Float64,Float64]) |> DataFrame

    # delete non-existing rows
    filter!(row -> !ismissing(row.ty), df_wh)
    # delete all zero rows
    filter!(row -> !all(x -> x==0, row), df_wh)
    # delete zero-capacity elemenets
    filter!(row -> !iszero(row.mx), df_wh)

    # delete repeated rows
    unique!(df_wh)

    df_wh = extend_catalog(df_wh,sp["WH0"],sp["WHz0"],sp["WHyn"],bdg["Bwh"])

    wh["ty"] = df_wh.ty         # name of water heater
    wh["mx"] = df_wh.mx         # maximum capacity [kW] || [kWh]
    wh["fcf"] = df_wh.fcf       # fuel conversion factor
    wh["fuel"] = df_wh.fuel     # type of fuel --gaseous|liquid-- {G,L}
    wh["eff"] = df_wh.eff       # efficiency --water heater--
    wh["tank"] = df_wh.tank     # in-site storage tank size [kWh]
    wh["inv"] = df_wh.inv       # capital cost [$]
    wh["fom"] = df_wh.fom       # fixed O&M cost [$/year]
    wh["vom"] = df_wh.vom       # variable O&M cost [$/kWh]
    wh["life"] = df_wh.life     # lifetime [years]

    wh["N"] = size(wh["ty"],1)  # number of water heaters

    return wh

end