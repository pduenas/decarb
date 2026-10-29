"""
load_topo(path::AbstractString,chp::Dict,abp::Dict)

Loads topo.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
chp     dictionary with CHP catalog
abp     dictionary with absorption chiller catalog

returns topo-type inputs in dictionary object
"""

function load_topo(path::AbstractString,chp::Dict,abp::Dict)

    # declare dictionary object to store parameters
    topo = Dict()

    path2file = joinpath(path,"topo.csv")

    # load file into dataframe with predefined types
    df_topo = CSV.File(path2file;delim=',',types=Dict{String,Type}("up"=>String,
                    "lo"=>String,"in"=>String,"eff"=>Float64)) |> DataFrame

    require_valid_columns(df_topo,"topo.csv",["lo","in","eff"];key="up")

    # delete non-existing rows
    filter!(row -> !ismissing(row.up), df_topo)
    # delete all zero rows
    filter!(row -> !all(x -> x==0, row), df_topo)

    topo["up"] = df_topo.up         # upper link of equipment
    topo["lo"] = df_topo.lo         # lower link of equipment or building
    topo["in"] = df_topo.in         # input product to lower link
    topo["eff"] = df_topo.eff       # efficiency of connection

    topo["N"] = size(topo["up"],1)  # number of thermal links

    # create heating topology connections
    topo["chp_bdg"] = create_chp_bdg(topo,chp)
    topo["chp_abp"] = create_chp_abp(topo,chp,abp)

    return topo

end

function create_chp_bdg(topo,chp)

    # rows: chp units | col1: hot air | col2: hot water
    chpbdg = zeros(chp["N"],2)
    chp_type = replace.(chp["ty"], r"-\d+$" => "")

    for i=1:topo["N"]
        topo["lo"][i]!="Building" && continue
        col = topo["in"][i]=="hot air" ? 1 : topo["in"][i]=="hot water" ? 2 : 0
        col == 0 && continue
        for c=1:chp["N"]
            if chp_type[c]!=topo["up"][i]
                continue
            end
            chpbdg[c,col] = topo["eff"][i]
        end
    end

    return chpbdg

end

function create_chp_abp(topo,chp,abp)

    # rows: chp units | cols: absorption chillers
    chpabp = zeros(chp["N"],abp["N"])
    chp_type = replace.(chp["ty"], r"-\d+$" => "")
    abp_type = replace.(abp["ty"], r"-\d+$" => "")

    for i=1:topo["N"]
        for c=1:chp["N"]
            for a=1:abp["N"]
                if chp_type[c]!=topo["up"][i] || abp_type[a]!=topo["lo"][i]
                    continue
                end
                chpabp[c,a] = topo["eff"][i]
            end
        end
    end

    return chpabp

end