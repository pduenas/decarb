"""
load_chp(path::AbstractString,sp::Dict,bdg::Dict)

Loads chp.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
sp      dictionary with equipment selection
bdg     dictionary with building data

returns chp-type inputs in dictionary object
"""

function load_chp(path::AbstractString,sp::Dict,bdg::Dict)

    # declare dictionary object to store parameters
    chp = Dict()

    path2file = joinpath(path,"chp.csv")

    # load file into dataframe with predefined types
    df_chp = CSV.File(path2file;delim=',',types=[String,Float64,Float64,Float64,String,
                    Float64,Float64,Float64,Float64,Float64,Float64,Float64]) |> DataFrame

    # delete non-existing rows
    delete!(df_chp,findall(ismissing.(df_chp.ty)))
    # delete zero-capacity elemenets
    delete!(df_chp,findall(iszero.(df_chp.mx)))
    # delete repeated rows
    unique!(df_chp)

    df_chp = extend_catalog(df_chp,sp["CHP0"],sp["CHPz0"],sp["CHPyn"],bdg["Bchp"])

    chp["ty"] = df_chp.ty           # name of CHP device
    chp["mx"] = df_chp.mx           # maximum CHP capacity [kW]
    chp["mn"] = df_chp.mn           # minimum CHP output [%]
    chp["hr"] = df_chp.hr           # thermal efficiency factor [%]
    chp["fuel"] = df_chp.fuel       # type of fuel --gaseous|liquid-- {G,L}
    chp["tank"] = df_chp.tank       # in site storage tank size [kWh]
    chp["h2p"] = df_chp.h2p         # heat-to-power ratio
    chp["sup"] = df_chp.sup         # start-up cost [$/start]
    chp["inv"] = df_chp.inv         # capital cost [$]
    chp["fom"] = df_chp.fom         # fixed O&M cost [$/year]
    chp["vom"] = df_chp.vom         # variable O&M cost [$/kWh]
    chp["life"] = df_chp.life       # lifetime [years]

    chp["N"] = size(chp["ty"],1)    # number of CHP units

    return chp

end