"""
read_econ(model::Model,tm::Dict)

Reads time series and outputs related to economic and emissions values

inputs:
model   optimization model object
tm      dictionary with time series data

returns dataframes of outputs
"""

function read_econ(model::Model,tm::Dict)

    # Read values for dual variables
    if has_duals(model)==true
        Qdual = round.(dual.(model[:eQbal]), digits=4)
        Tdual = zeros(Float64, tm["P"])
        Tdual[1] = round.(dual.(model[:eTbal0]), digits=4)
        for p=2:tm["P"]
            Tdual[p] = abs.(round.(dual.(model[:eTbal][p]), digits=4))
        end
        HWdual = round.(dual.(model[:eHWbal]), digits=4)
    else
        Qdual = zeros(Float64, tm["P"])
        Tdual = zeros(Float64, tm["P"])
        HWdual = zeros(Float64, tm["P"])    
    end

    # Read values for economic variables
    C = round.(value.(model[:COST_VAR]), digits=2)
    Qearn = round.(sum(value.(model[:vQearn][t]) for t=1:tm["P"]), digits=2)
    Qcost = round.(sum(value.(model[:vQcost][t]) for t=1:tm["P"]), digits=2)
    GLcost = round.(sum(value.(model[:vGLcost][t]) for t=1:tm["P"]), digits=2)
    Cvom = round.(value.(model[:COST_VOM]), digits=2)
    Cfom = round.(value.(model[:COST_FOM]), digits=2)
    NSEcost = round.(sum(value.(model[:vNSEcost][t]) for t=1:tm["P"]), digits=2)
    NSTcost = round.(sum(value.(model[:vNSTcost][t]) for t=1:tm["P"]), digits=2)
    NSHWcost = round.(sum(value.(model[:vNSHWcost][t]) for t=1:tm["P"]), digits=2)

    # Read values for direct and indirect emissions
    Bco2 = round.(value.(model[:CO2_B]), digits=2)
    Eco2 = round.(value.(model[:CO2_E]), digits=2)

    df_dual = DataFrame(dualQ=Qdual,dualT=Tdual,dualHW=HWdual)
    df_econ = DataFrame(eq=[C;Qearn;Qcost;GLcost;Cvom;Cfom;NSEcost;NSTcost;NSHWcost;
        Bco2;Eco2])
    
    return df_dual,df_econ

end
