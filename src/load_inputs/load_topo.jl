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
    df_topo = CSV.File(path2file;delim=',',types=[String,String,String,String,Float64,
                    Float64,Float64]) |> DataFrame

    # delete non-existing rows
    filter!(row -> !ismissing(row.up), df_topo)
    # delete all zero rows
    filter!(row -> !all(x -> x==0, row), df_topo)

    topo["up"] = df_topo.up         # upper link of equipment
    topo["lo"] = df_topo.lo         # lower link of equipment or building
    topo["in"] = df_topo.in         # input product to lower link
    topo["fuel"] = df_topo.fuel     # existing supplemental firing {G,D}
    topo["mx"] = df_topo.mx         # nominal capacity [kW]
    topo["fcf"] = df_topo.fcf       # fuel conversion factor
    topo["eff"] = df_topo.eff       # efficiency of connection

    topo["N"] = size(topo["up"],1)  # number of thermal links

    # create heating topology connections
    topo["chp_bdg"] = create_chp_bdg(topo,chp)
    topo["chp_abp"] = create_chp_abp(topo,chp,abp)
    topo["chp_fire"] = create_chp_fire(topo,chp,abp)

    return topo

end

function create_chp_bdg(topo,chp)

    # rows: chp units | col1: hot air | col2: hot water
    chpbdg = zeros(chp["N"],2)

    for i=1:topo["N"]
        if topo["in"][i] == "hot air"
            col = 1
        elseif topo["in"][i] == "hot water"
            col = 2
        end
        for c=1:chp["N"]
            if chp["ty"][c][1:end-2]!=topo["up"][i]
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

    for i=1:topo["N"]
        for c=1:chp["N"]
            for a=1:abp["N"]
                if chp["ty"][c][1:end-2]!=topo["up"][i] || 
                    abp["ty"][a][1:end-2]!=topo["lo"][i]
                    continue
                end
                chpabp[c,a] = topo["eff"][i]
            end
        end
    end

    return chpabp

end

function create_chp_fire(topo,chp,abp)

    # dim1: chp units | dim2: absorption chillers, hot air, hot water | dim3: gaseous, liquid
    chpfire = zeros(chp["N"],abp["N"]+2,2)

    for i=1:topo["N"]
        if topo["fuel"][i] == "G"
            dim3 = 1
        elseif topo["fuel"][i] == "L"
            dim3 = 2
        else
            continue
        end
        if topo["in"][i] == "hot air"
            dim2 = abp["N"]+1
        elseif topo["in"][i] == "hot water"
            dim2 = abp["N"]+2
        else
            dim2 = 0
        end
        if dim2 != 0
            for c=1:chp["N"]
                chpfire[chp["ty"][c][1:end-2]==topo["up"][i],dim2,dim3] .= topo["fcf"][i]*topo["mx"][i]
            end
        elseif dim2 == 0
            for c=1:chp["N"]
                for a=1:abp["N"]
                    chpfire[chp["ty"][c][1:end-2]==topo["up"][i],
                            abp["ty"][a][1:end-2]==topo["up"][i],dim3] .= topo["fcf"][i]*topo["mx"][i]
                end
            end
        end
    end

    return chpfire

end