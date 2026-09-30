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
        sp_0 = sp["CHP0"]
        sp_z0 = sp["CHPz0"]
    elseif equip=="hvac"
        unit = value.(model[:bHVACty])
        sp_0 = sp["HVAC0"]
        sp_z0 = sp["HVACz0"]
    elseif equip=="abp"
        unit = value.(model[:bABPty])
        sp_0 = sp["ABP0"]
        sp_z0 = sp["ABPz0"]
    elseif equip=="wh"
        unit = value.(model[:bWHty])
        sp_0 = sp["WH0"]
        sp_z0 = sp["WHz0"]
    end

    idx = findall(!iszero, unit)

    if !isempty(idx)
        eq = [i[2] for i in idx]
        win = [i[1] for i in idx]
        unit_type = type[eq]
        cost = cost[eq]

        # Thermal catalogs append a numeric suffix for each physical unit.
        # Reports aggregate those copies back to their catalog equipment type.
        base_type = replace.(unit_type, r"-\d+$" => "")
        rows = DataFrame(Eq=base_type, Inv=cost, Window=win,
            Qty=ones(Int64, length(idx)))
        report = combine(groupby(rows, [:Eq, :Inv, :Window]), :Qty => sum => :Qty)
        sort!(report, [:Eq, :Window])
        report.New = copy(report.Qty)

        for n in eachindex(sp_0)
            sp_0[n] == "0" && continue
            matching = findall(==(sp_0[n]), report.Eq)
            if !isempty(matching)
                first_idx = matching[argmin(report.Window[matching])]
                report.New[first_idx] -= sp_z0[n]
            end
        end
        report.CAPEX = report.Inv .* report.New
        report.Date = [tmDate[tmIW[w]] for w in report.Window]

        return select(report, :Eq, :Inv, :Qty, :New, :CAPEX, :Date)
    else
        return DataFrame(Eq=nothing,Inv=nothing,Qty=nothing,New=nothing,CAPEX=nothing,Date=nothing)
    end

end
