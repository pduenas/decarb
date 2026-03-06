"""
load_bdgii(path::AbstractString,bdg::Dict)

Loads bdg_ii.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory

returns bdg-type inputs in dictionary object
"""

function load_bdgii(path::AbstractString,bdg::Dict)

    path2file = joinpath(path,"bdg_ii.csv")

    # load file into dataframe with predefined types
    df_bdg = CSV.File(path2file;delim=',',types=[Float64,Float64,Float64]) |> DataFrame

    # delete non-existing rows
    filter!(row -> !ismissing(row.pBk1), df_bdg)
    # delete all zero rows
    filter!(row -> !all(x -> x==0, row), df_bdg)

    bdg["Bk1"] = df_bdg.pBk1[1]     # thermal coefficient- outdoor to indoor temperature [-]
    bdg["Bk2"] = df_bdg.pBk2[1]     # thermal coefficient- external radiation [°C/kW]
    bdg["Bk3"] = df_bdg.pBk3[1]     # thermal coefficient- internal heating/cooling [°C/kW]

    return bdg

end