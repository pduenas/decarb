"""
load_ev(path::AbstractString)

Loads df_ev.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory

returns ev-type inputs in dictionary object
"""

function load_ev(path::AbstractString)

    # declare dictionary object to store parameters
    ev = Dict()

    path2file = joinpath(path,"ev.csv")

    # load file into dataframe with predefined types
    df_ev = CSV.File(path2file;delim=',',types=[String,Float64,Float64,Float64,
                    Float64,Float64,Float64,Float64]) |> DataFrame

    # delete non-existing rows
    delete!(df_ev,findall(ismissing.(df_ev.ty)))
    # delete zero-capacity elemenets
    delete!(df_ev,findall(iszero.(df_ev.mx)))
    # delete repeated rows
    unique!(df_ev)

    ev["ty"] = df_ev.ty         # name of electric vehicle
    ev["mx"] = df_ev.mx         # maximum battery capacity [kWh]
    ev["drv"] = df_ev.drv       # driving effiency [kWh/km]
    ev["cold"] = df_ev.cold     # winter efficiency penalty [%]
    ev["up"] = df_ev.up         # charging rate [kWh/h]
    ev["dn"] = df_ev.dn         # discharging rate [kWh/h]
    ev["effu"] = df_ev.effu     # charging efficiency [%]
    ev["effd"] = df_ev.effd     # discharging efficiency [%]

    ev["N"] = size(ev["ty"],1)  # number of EV types

    return ev

end