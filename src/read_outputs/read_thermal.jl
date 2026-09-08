"""
read_thermal(model::Model,sp::Dict,equip::String,attr::Dict,tmDate::Vector,tmIW::Vector)

Reads thermal component outputs from model and load them into dataframe

inputs:
model   optimization model object
sp      dictionary with equipment selection
equip   character vector with equipment type
attr    dictionary of equipment attributes
tmDate  date/time vector for all periods
tmIW    investment window period indices

returns dataframes of outputs
"""

function read_thermal(model::Model,sp::Dict,equip::String,attr::Dict,tmDate::Vector,tmIW::Vector)

    # read specific output and attrs
    type = attr["ty"]
    cost = attr["inv"]
    if equip=="chp"
        unit = value.(model[:bCHPty])
        sp_yn = sp["CHPyn"]
        sp_0 = sp["CHP0"]
        sp_z0 = sp["CHPz0"]
    elseif equip=="hvac"
        unit = value.(model[:bHVACty])
        sp_yn = sp["HVACyn"]
        sp_0 = sp["HVAC0"]
        sp_z0 = sp["HVACz0"]
    elseif equip=="abp"
        unit = value.(model[:bABPty])
        sp_yn = sp["ABPyn"]
        sp_0 = sp["ABP0"]
        sp_z0 = sp["ABPz0"]
    elseif equip=="wh"
        unit = value.(model[:bWHty])
        sp_yn = sp["WHyn"]
        sp_0 = sp["WH0"]
        sp_z0 = sp["WHz0"]
    end

    idx = findall(!iszero, unit)    # number of existing equips

    if !isempty(idx)
        n_unit = length(idx)
        eq = [i[2] for i in idx]
        win = [i[1] for i in idx]
        date = [tmDate[tmIW[w]] for w in win]
        type = type[eq]
        cost = cost[eq]
        hyphen = findlast.("-",type)
        for n=1:n_unit
            type[n] = type[n][1:hyphen[n][1]-1]
        end
        quantity = zeros(Int64,n_unit)
        for n=1:n_unit
            quantity[n] = sum(type.==type[n])
        end
        new = zeros(Int64,n_unit)
        for n in findall(x -> x!="0", sp_yn)
            name = findall(x -> x==sp_0[n], type)
            if !isempty(name)
                first_idx = name[argmin(win[name])]
                new[first_idx] -= sp_z0[n]
            end
        end
        capex = cost.*new
        return DataFrame(Eq=type,Inv=cost,Qty=quantity,New=new,CAPEX=capex,Date=date)
    else
        return DataFrame(Eq=nothing,Inv=nothing,Qty=nothing,New=nothing,CAPEX=nothing,Date=nothing)
    end

end
