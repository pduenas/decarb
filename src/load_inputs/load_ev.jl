"""
load_ev(path::AbstractString,tm::Dict,sp::Dict)

Loads ev.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
tm      dictionary with time series data
sp      dictionary with equipment selection

returns ev-type inputs in dictionary object
"""

function load_ev(path::AbstractString,tm::Dict,sp::Dict)

    # declare dictionary object to store parameters
    ev = Dict()

    path2file = joinpath(path,"ev.csv")

    # load file into dataframe with predefined types
    df_ev = CSV.File(path2file;delim=',',types=Dict{String,Type}("ty"=>String,
                    "mx"=>Float64,"drv"=>Float64,"cold"=>Float64,"up"=>Float64,
                    "dn"=>Float64,"effu"=>Float64,"effd"=>Float64)) |> DataFrame

    require_valid_columns(df_ev,"ev.csv",["mx","drv","cold","up","dn","effu","effd"];
        key="ty")

    # delete non-existing rows
    filter!(row -> !ismissing(row.ty), df_ev)
    # delete all zero rows
    filter!(row -> !all(x -> x==0, row), df_ev)
    # delete zero-capacity elements
    filter!(row -> !iszero(row.mx), df_ev)
    # delete repeated rows
    unique!(df_ev)

    # Expand existing vehicles only; EV investment is not supported.
    df_ev = extend_catalog(df_ev,sp["EV0"],sp["EVz0"],fill("NO", length(sp["EV0"])),nothing)

    ev["ty"] = df_ev.ty         # name of electric vehicle
    ev["mx"] = df_ev.mx         # maximum battery capacity [kWh]
    ev["drv"] = df_ev.drv       # driving efficiency [kWh/km]
    ev["cold"] = df_ev.cold     # winter efficiency penalty [p.u.]
    ev["up"] = df_ev.up         # charging rate [kWh/h]
    ev["dn"] = df_ev.dn         # discharging rate [kWh/h]
    ev["effu"] = df_ev.effu     # charging efficiency [p.u.]
    ev["effd"] = df_ev.effd     # discharging efficiency [p.u.]

    ev["N"] = size(ev["ty"],1)  # number of EV types

    ev["kwh"] = zeros(tm["P"],ev["N"])
    ev["time"] = zeros(tm["P"],ev["N"])

    ty_names = ev_names(sp["EV0"],sp["EVz0"])

    for (col,name) in enumerate(ty_names)
        # find index where electric vehicle type is found
        idx = findfirst(x -> x == name, ev["ty"])
        # Unmatched selections are reported after loading by validate_inputs.
        isnothing(idx) && continue
        i = 1   # initialize counter
        start = true
        while i <= length(tm["EVdem"])
            # find departure time of vehicle
            if tm["EVtype"][i] == col
                # correction for initial plugged status
                if start
                    ev["time"][1:(i-1),idx] .= 1
                    start = false
                end
                for j in i+1:length(tm["EVdem"])
                    # find arrival time of vehicle
                    if tm["EVtype"][j] == col
                        # save consumption as two-way drive at hours i-th and j-th
                        ev["kwh"][i,idx] = min(ev["mx"][idx],
                            (km_to_kwh(ev["drv"][idx],tm["EVdem"][i],tm["Tout"][i]) +
                             km_to_kwh(ev["drv"][idx],tm["EVdem"][i],tm["Tout"][j-1]))/2)
                        i = j
                        # flag periods plugged in at building since hour j-th
                        h = 0
                        hours = tm["EVtm"][j]
                        while h < hours && j <= tm["P"]
                            ev["time"][j,idx] = 1
                            h += tm["TM"][j]    # increase number of hours
                            j += 1              # move one step forward
                        end
                        break
                    end
                end
            end
            i += 1
        end
    end

    return ev

end



"""
ev_names(ty::Vector{String},z0::AbstractVector{<:Integer})
"""

function ev_names(ty,z0)

    name = String[]

    for i in eachindex(ty)
        # update name of units of coincident type for new lines
        if z0[i] > 0
            append!(name,string.(ty[i],"-",Vector(1:z0[i])))
        end
    end

    return name

end

"""
km_to_kwh(drv::Float64,km::Float64,T::Float64)
ref: https://pubs.acs.org/doi/pdf/10.1021/es505621s
"""

function km_to_kwh(drv,km,T)
    
    kwh2km = drv - 0.002460636*T + 0.000185174*T^2 - 0.0000142232*T^3 + 
        0.00000034518*T^4 - 0.00000000242563*T^5

    # intensity cannot be negative at extreme temperatures
    return km*max(kwh2km,0.0)

end
