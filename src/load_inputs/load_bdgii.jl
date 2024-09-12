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
    df_bdg = CSV.File(path2file;delim=',',types=[Float64,Float64]) |> DataFrame

    # delete non-existing rows
    delete!(df_bdg,findall(ismissing.(df_bdg.pBwall)))

    bdg["Bwall"] = df_bdg.pBwall        # surface of wall section [m2]
    bdg["Bkwall"] = df_bdg.pBkwall      # material U-value of wall section [W/m2-°C]

    # unit transformations
    bdg["Bkwall"] = bdg["Bkwall"]/1000  # in [kW/m2-°C]

    return bdg

end