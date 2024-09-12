"""
read_thermal(model::Model,sp::Dict,equip::String,attr:Dict)

Reads thermal component outputs from model and load them into dataframe

inputs:
model   optimization model object
sp      dictionary with equipment selection
equip   character vector with equipment type
attr    dictionary of equipment attributes

returns dataframes of outputs
"""

function read_thermal(model::Model,sp::Dict,equip::String,attr::Dict)

    # read specific output and attrs
    type = attr["ty"]
    cost = attr["inv"]
    if equip=="chp"
        unit = vec(value.(model[:bCHPty]))
        sp_yn = sp["CHPyn"]
        sp_0 = sp["CHP0"]
        sp_z0 = sp["CHPz0"]
    elseif equip=="hvac"
        unit = vec(value.(model[:bHVACty]))
        sp_yn = sp["HVACyn"]
        sp_0 = sp["HVAC0"]
        sp_z0 = sp["HVACz0"]
    elseif equip=="abp"
        unit = vec(value.(model[:bABSty]))
        sp_yn = sp["ABSyn"]
        sp_0 = sp["ABS0"]
        sp_z0 = sp["ABSz0"]
    elseif equip=="wh"
        unit = vec(value.(model[:bWHty]))
        sp_yn = sp["WHyn"]
        sp_0 = sp["WH0"]
        sp_z0 = sp["WHz0"]
    end

    n_unit = UInt8(sum(unit, dims=1)[1])	# number of existing equips

    if n_unit>0
        type = type[findall(unit -> unit!=0, unit)]
        cost = cost[findall(unit -> unit!=0, unit)]
        hyphen = findlast.("-",type)
        for n=1:n_unit
            type[n] = type[n][1:hyphen[n][1]-1]
        end
        quantity = zeros(Int8,n_unit)
        for n=1:n_unit
            quantity[n] = sum(type.==type[n])
        end
        new = zeros(Int8,n_unit)
        for n in findall(sp_yn.!="0")
            name = type.==sp_0[n]
            if any(name)
                new[name] = quantity[name] .- sp_z0[n]
            end
        end
        capex = cost.*new
        return DataFrame(Eq=type,Inv=cost,Qty=quantity,New=new,CAPEX=capex)
    else
        return DataFrame(Eq=nothing,Inv=nothing,Qty=nothing,New=nothing,CAPEX=nothing)
    end

end
