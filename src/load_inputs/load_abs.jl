"""
load_abs(path::AbstractString,sp::Dict,bdg::Dict)

Loads abs.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
sp      dictionary with equipment selection
bdg     dictionary with building data

returns abs-type inputs in dictionary object
"""

function load_abs(path::AbstractString,sp::Dict,bdg::Dict)

    # declare dictionary object to store parameters
    abs = Dict()
    
    path2file = joinpath(path,"abs.csv")

    # load file into dataframe with predefined types
    df_abs = CSV.File(path2file;delim=',',types=[String,Float64,Float64,String,
                    Float64,Float64,Float64,Float64,Float64]) |> DataFrame

    # delete non-existing rows
    delete!(df_abs,findall(ismissing.(df_abs.ty)))
    # delete zero-capacity elemenets
    delete!(df_abs,findall(iszero.(df_abs.mx)))
    # delete repeated rows
    unique!(df_abs)

    df_abs = extend_catalog(df_abs,sp["ABS0"],sp["ABSz0"],sp["ABSyn"],bdg["Babs"])

    abs["ty"] = df_abs.ty           # name of absorption chiller
    abs["mx"] = df_abs.mx           # maximum absorption chiller capacity [kW]
    abs["hr"] = df_abs.hr           # heat rate [MMBtu/kWh]
    abs["fuel"] = df_abs.fuel       # type of fuel --gaseous|liquid-- {G,L}
    abs["ac"] = df_abs.ac           # efficiency --cooling--
    abs["inv"] = df_abs.inv         # capital cost [$]
    abs["fom"] = df_abs.fom         # fixed O&M cost [$/year]
    abs["vom"] = df_abs.vom         # variable O&M cost [$/kWh]
    abs["life"] = df_abs.life       # lifetime [years]

    abs["N"] = size(abs["ty"],1)    # number of absorption chillers

    return abs

end