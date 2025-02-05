"""
read_der(model::Model,sp::Dict,equip::String,attr:Dict)

Reads DER component outputs from model and load them into dataframe

inputs:
model   optimization model object
sp      dictionary with equipment selection
in      dictionary with miscellaneous data
equip   character vector with equipment type
attr    dictionary of equipment attributes

returns dataframes of outputs
"""

function read_der(model::Model,sp::Dict,in::Dict,equip::String,attr::Dict)

    # read specific output and attributes
    type = attr["ty"]
    cost = attr["inv"]
    if equip=="pv"
        unit = vec(value.(model[:zPV]))
        sp_yn = sp["PVyn"]
        sp_0 = sp["PV0"]
        sp_z0 = sp["PVz0"]
    elseif equip=="bess"
        unit = vec(value.(model[:zBESS]))
        sp_yn = sp["BESSyn"]
        sp_0 = sp["BESS0"]
        sp_z0 = sp["BESSz0"]
    elseif equip=="wind"
        unit = vec(value.(model[:zWIND]))
        sp_yn = sp["WINDyn"]
        sp_0 = sp["WIND0"]
        sp_z0 = sp["WINDz0"]
    end

    n_unit = UInt8(sum(unit, dims=1)[1])	# number of existing equips

    if n_unit>0
        type = collect(Tuple(type[findall(unit -> unit!=0, unit)]))
        cost = cost[findall(unit -> unit!=0, unit)]
        quantity = zeros(Int8,n_unit)
        for n=1:n_unit
            quantity[n] = sum(type.==type[n])
        end
        new = zeros(length(quantity))
        for n in findall(sp_yn.!="0")
            name = findall(type.==sp_0[n])
            if !isempty(name)
                new[name] = quantity[name] .- sp_z0[n]
            end
        end
        capex = cost.*new
        df_eq = DataFrame(Eq=unit[unit .> 0])
        if length(type)==ncol(df_eq)
            rename!(df_eq, Dict(names(df_eq) .=> Symbol.(type)))
        end
        return DataFrame(Eq=type,Inv=cost,Qty=quantity,New=new,CAPEX=capex), df_eq
    else
        return DataFrame(Eq=nothing,Inv=nothing,Qty=nothing,New=nothing,CAPEX=nothing),
            DataFrame(zeros(Int64,in["IT"],0),:auto)
    end

end