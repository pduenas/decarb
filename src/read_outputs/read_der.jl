"""
read_der(model::Model,sp::Dict,equip::String,attr:Dict)

Reads DER component outputs from model and load them into dataframe

inputs:
model   optimization model object
sp      dictionary with equipment selection
cfg     dictionary with miscellaneous data
equip   character vector with equipment type
attr    dictionary of equipment attributes

returns dataframes of outputs
"""

function read_der(model::Model,sp::Dict,cfg::Dict,equip::String,attr::Dict)

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

    idx = findall(x -> x != 0, unit)        # indices of existing equipment

    if !isempty(idx)
        type = type[idx]
        cost = cost[idx]
        quantity = unit[idx]
        new = zeros(length(quantity))
        for n in findall(x -> x!="0", sp_yn)
            name = findall(x -> x==sp_0[n], type)
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
            DataFrame(zeros(Int64,cfg["IT"],0),:auto)
    end

end