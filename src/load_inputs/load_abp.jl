"""
load_abp(path::AbstractString,sp::Dict,bdg::Dict)

Loads abp.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
sp      dictionary with equipment selection
bdg     dictionary with building data

returns abp-type inputs in dictionary object
"""

function load_abp(path::AbstractString,sp::Dict,bdg::Dict)

    # declare dictionary object to store parameters
    abp = Dict()
    
    path2file = joinpath(path,"abp.csv")

    # load file into dataframe with predefined types
    df_abp = CSV.File(path2file;delim=',',types=[String,Float64,Float64,String,
                    Float64,Float64,Float64,Float64,Float64]) |> DataFrame
    
    # delete non-existing rows
    filter!(row -> !ismissing(row.ty), df_abp)
    # delete all zero rows
    filter!(row -> !all(x -> x==0, row), df_abp)
    # delete zero-capacity elemenets
    filter!(row -> !iszero(row.mx), df_abp)
    # delete repeated rows
    unique!(df_abp)

    df_abp = extend_catalog(df_abp,sp["ABP0"],sp["ABPz0"],sp["ABPyn"],bdg["Babp"])

    abp["ty"] = df_abp.ty           # name of absorption chiller
    abp["mx"] = df_abp.mx           # maximum absorption chiller capacity [kW]
    abp["fcf"] = df_abp.fcf         # fuel conversion factor
    abp["fuel"] = df_abp.fuel       # type of fuel --gaseous|liquid-- {G,L}
    abp["ac"] = df_abp.ac           # efficiency --cooling--
    abp["inv"] = df_abp.inv         # capital cost [$]
    abp["fom"] = df_abp.fom         # fixed O&M cost [$/year]
    abp["vom"] = df_abp.vom         # variable O&M cost [$/kWh]
    abp["life"] = df_abp.life       # lifetime [years]

    abp["N"] = size(abp["ty"],1)    # number of absorption chillers

    return abp

end