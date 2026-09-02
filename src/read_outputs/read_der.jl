"""
read_der(model::Model,sp::Dict,cfg::Dict,equip::String,attr::Dict,tmDate::Vector,tmIW::Vector)

Reads DER component outputs from model and load them into dataframe

inputs:
model   optimization model object
sp      dictionary with equipment selection
cfg     dictionary with miscellaneous data
equip   character vector with equipment type
attr    dictionary of equipment attributes
tmDate  date/time vector for all periods
tmIW    investment window period indices

returns dataframes of outputs
"""

function read_der(model::Model,sp::Dict,cfg::Dict,equip::String,attr::Dict,tmDate::Vector,tmIW::Vector)

    # read specific output and attributes
    type = attr["ty"]
    cost = attr["inv"]
    if equip=="pv"
        vals = value.(model[:zPV])
        sp_yn = sp["PVyn"]
        sp_0 = sp["PV0"]
        sp_z0 = sp["PVz0"]
    elseif equip=="bess"
        vals = value.(model[:zBESS])
        sp_yn = sp["BESSyn"]
        sp_0 = sp["BESS0"]
        sp_z0 = sp["BESSz0"]
    elseif equip=="wind"
        vals = value.(model[:zWIND])
        sp_yn = sp["WINDyn"]
        sp_0 = sp["WIND0"]
        sp_z0 = sp["WINDz0"]
    end

    # Use CartesianIndex to preserve both investment window and equipment dimensions
    idx = findall(!iszero, vals)
    
    if !isempty(idx)
        eq = [i[2] for i in idx]
        win = [i[1] for i in idx]
        # Map investment window number to period index, then to date
        date = [tmDate[tmIW[w]] for w in win]
        type = type[eq]
        cost = cost[eq]
        quantity = [vals[i] for i in idx]
        new = zeros(length(quantity))
        for n in findall(x -> x!="0", sp_yn)
            name = findall(x -> x==sp_0[n], type)
            if !isempty(name)
                new[name] = quantity[name] .- sp_z0[n]
            end
        end
        capex = cost.*new
        return DataFrame(Eq=type,Inv=cost,Qty=quantity,New=new,CAPEX=capex,Date=date)
    else
        return DataFrame(Eq=nothing,Inv=nothing,Qty=nothing,New=nothing,CAPEX=nothing,Date=nothing)
    end

end