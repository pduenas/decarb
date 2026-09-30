"""
objective_function!(model::Model,cfg::Dict,tm::Dict,chp::Dict,abp::Dict,hvac::Dict,
    wh::Dict,pv::Dict,bess::Dict,ev::Dict,wind::Dict)

Creates variables, expressions and constraints associated with the objective function

inputs:
model   name of core model
cfg     dictionary with configuration input data
tm      dictionary with time series data
chp     dictionary with CHP data
abp     dictionary with absorption chiller data
hvac    dictionary with HVAC data
wh      dictionary with water heater data
pv      dictionary with PV data
bess    dictionary with BESS data
ev      dictionary with EV data
wind    dictionary with wind turbine data

"""
function objective_function!(model::Model,cfg::Dict,tm::Dict,chp::Dict,abp::Dict,hvac::Dict,
    wh::Dict,pv::Dict,bess::Dict,ev::Dict,wind::Dict)

    # cost of non-served electricity [$]
    @expression(model, vNSEcost[t=1:tm["P"]], cfg["NSEcost"]*model[:vNSE_Q][t])
    # cost of non-served temperature [$]
    @expression(model, vNSTcost[t=1:tm["P"]], cfg["NSTcost"]*(model[:vTup][t]+model[:vTlo][t])*tm["TM"][t])
    # cost of non-served hot water [$]
    @expression(model, vNSHWcost[t=1:tm["P"]], cfg["NSHWcost"]*tm["HWdem"][t]*model[:vNShw][t])
    # cost of leaving EV charge below required level [$]
    @expression(model, vNSEVcost[t=1:tm["P"]],
        cfg["NSEVcost"]*sum(model[:vEVlo][t,e] for e=1:ev["N"]))
    # cost of purchasing electricity [$]
    @expression(model, vQcost[t=1:tm["P"]], tm["TM"][t]*tm["QcostBuy"][t]*model[:vQbuy][t])
    # income from selling electricity [$]
    @expression(model, vQearn[t=1:tm["P"]], tm["TM"][t]*tm["QcostSell"][t]*model[:vQsell][t])
    # cost of purchased fuel [$]
    @expression(model, vGLcost[t=1:tm["P"]],
        tm["Gcost"][t]*(model[:vCHP_G][t]+model[:vABP_G][t]+model[:vWH_G][t]) +
        tm["Lcost"][t]*(model[:vCHP_L][t]+model[:vABP_L][t]+model[:vWH_L][t]))

    # penalties for driver type: range anxious (-1), indifferent (0), battery concious (1)
    @expression(model, vEVpen,
        sum(cfg["EVdriver"]*cfg["EVpen"]*model[:vEVsoc][t,e] for t=1:tm["P"],e=1:ev["N"] if ev["mx_eff"][e] > 0))

    # cost of peak capacity subscription [$]
    @expression(model, vQmxCost, sum(tm["QmxCostN"][n]*model[:vQmx][n] for n=1:cfg["QmxTM"]))

    # Existing equipment is encoded as an explicit lower bound on the first
    # investment variable. Binary variables without such a bound are new.
    sunk_stock(vars) = map(v -> has_lower_bound(v) ? lower_bound(v) : 0.0, vars)

    # annualized cost of CHP during time scope [$]
    b = upper_bound.(model[:bCHPty]) .> 0   # for potential investments
    z0 = sunk_stock(model[:bCHPty])         # existing stock (sunk capital)
    @expression(model, vCHPinv,
        sum(b[i,c]*sum(tm["TM"][tm["IW"][i]:tm["P"]])/8760*chp["inv"][c]*cfg["IR"]/
        (1-(1+cfg["IR"])^(-chp["life"][c]))*(model[:bCHPty][i,c]-z0[i,c])
        for i=1:cfg["IT"],c=1:chp["N"]))
    # annualized cost of HVAC during time scope [$]
    b = upper_bound.(model[:bHVACty]) .> 0  # for potential investments
    z0 = sunk_stock(model[:bHVACty])        # existing stock (sunk capital)
    @expression(model, vHVACinv,
        sum(b[i,h]*sum(tm["TM"][tm["IW"][i]:tm["P"]])/8760*hvac["inv"][h]*cfg["IR"]/
        (1-(1+cfg["IR"])^(-hvac["life"][h]))*(model[:bHVACty][i,h]-z0[i,h])
        for i=1:cfg["IT"],h=1:hvac["N"]))
    # annualized cost of absorption chiller during time scope [$]
    b = upper_bound.(model[:bABPty]) .> 0   # for potential investments
    z0 = sunk_stock(model[:bABPty])         # existing stock (sunk capital)
    @expression(model, vABPinv,
        sum(b[i,a]*sum(tm["TM"][tm["IW"][i]:tm["P"]])/8760*abp["inv"][a]*cfg["IR"]/
        (1-(1+cfg["IR"])^(-abp["life"][a]))*(model[:bABPty][i,a]-z0[i,a])
        for i=1:cfg["IT"],a=1:abp["N"]))
    # annualized cost of water heater during time scope [$]
    b = upper_bound.(model[:bWHty]) .> 0    # for potential investments
    z0 = sunk_stock(model[:bWHty])          # existing stock (sunk capital)
    @expression(model, vWHinv,
        sum(b[i,w]*sum(tm["TM"][tm["IW"][i]:tm["P"]])/8760*wh["inv"][w]*cfg["IR"]/
        (1-(1+cfg["IR"])^(-wh["life"][w]))*(model[:bWHty][i,w]-z0[i,w])
        for i=1:cfg["IT"],w=1:wh["N"]))
    # annualized cost of PV during time scope [$]
    b = upper_bound.(model[:zPV]) .> 0      # for potential investments
    z0 = sunk_stock(model[:zPV])            # existing stock (sunk capital)
    @expression(model, vPVinv,
        sum(b[i,v]*sum(tm["TM"][tm["IW"][i]:tm["P"]])/8760*pv["inv"][v]*cfg["IR"]/
        (1-(1+cfg["IR"])^(-pv["life"][v]))*(model[:zPV][i,v]-z0[i,v])
        for i=1:cfg["IT"],v=1:pv["N"]))
    # annualized cost of wind turbine during time scope [$]
    b = upper_bound.(model[:zWIND]) .> 0   # for potential investments
    z0 = sunk_stock(model[:zWIND])         # existing stock (sunk capital)
    @expression(model, vWINDinv,
        sum(b[i,d]*sum(tm["TM"][tm["IW"][i]:tm["P"]])/8760*wind["inv"][d]*cfg["IR"]/
        (1-(1+cfg["IR"])^(-wind["life"][d]))*(model[:zWIND][i,d]-z0[i,d])
        for i=1:cfg["IT"],d=1:wind["N"]))
    # annualized cost of electricity storage during time scope [$]
    b = upper_bound.(model[:zBESS]) .> 0   # for potential investments
    z0 = sunk_stock(model[:zBESS])         # existing stock (sunk capital)
    @expression(model, vBESSinv,
        sum(b[i,s]*sum(tm["TM"][tm["IW"][i]:tm["P"]])/8760*bess["inv"][s]*cfg["IR"]/
        (1-(1+cfg["IR"])^(-bess["life"][s]))*(model[:zBESS][i,s]-z0[i,s])
        for i=1:cfg["IT"],s=1:bess["N"]))

    # total fixed O&M costs (+) [$]
    @expression(model, COST_FOM,
        sum(chp["fom"][c]*model[:bCHPty][i,c]*tm["H"]/8760 for i=1:cfg["IT"],c=1:chp["N"]) +
        sum(abp["fom"][a]*model[:bABPty][i,a]*tm["H"]/8760 for i=1:cfg["IT"],a=1:abp["N"]) +
        sum(hvac["fom"][h]*model[:bHVACty][i,h]*tm["H"]/8760 for i=1:cfg["IT"],h=1:hvac["N"]) +
        sum(wh["fom"][w]*model[:bWHty][i,w]*tm["H"]/8760 for i=1:cfg["IT"],w=1:wh["N"]) +
        sum(pv["fom"][v]*model[:zPV][i,v]*tm["H"]/8760 for i=1:cfg["IT"],v=1:pv["N"]) +
        sum(wind["fom"][d]*model[:zWIND][i,d]*tm["H"]/8760 for i=1:cfg["IT"],d=1:wind["N"]) +
        sum(bess["fom"][s]*model[:zBESS][i,s]*tm["H"]/8760 for i=1:cfg["IT"],s=1:bess["N"]))
    # total variable O&M costs (+) [$]
    @expression(model, COST_VOM,
        sum(chp["vom"][c]*model[:vCHP_Q][t,c] for t=1:tm["P"],c=1:chp["N"]) +
        sum(abp["vom"][a]*model[:vABP_AC][t,a] for t=1:tm["P"],a=1:abp["N"]) +
        sum(hvac["vom"][h]*model[:vHVAC_HT][t,h] for t=1:tm["P"],h=1:hvac["N"]) +
        sum(hvac["vom"][h]*model[:vHVAC_AC][t,h] for t=1:tm["P"],h=1:hvac["N"]) +
        sum(wh["vom"][w]*model[:vWH_HW][t,w] for t=1:tm["P"],w=1:wh["N"]))

    # total variable costs (+) / total incomes (-) [$]
    @expression(model, COST_VAR,
        sum(vQcost[t]-vQearn[t]+vGLcost[t] for t=1:tm["P"]) + vQmxCost + COST_VOM + COST_FOM)
    # total discomfort costs
    @expression(model, COST_NS,
        sum(vNSEcost[t]+vNSTcost[t]+vNSHWcost[t]+vNSEVcost[t] for t=1:tm["P"]))
    # total annualized costs
    @expression(model, COST_INV, vCHPinv+vHVACinv+vABPinv+vWHinv+vPVinv+vWINDinv+vBESSinv)
    # total costs [$]
    @expression(model, COST, COST_VAR + COST_NS + COST_INV + vEVpen)

    # CO2 emissions from building [ton]
    @expression(model, CO2_B,
        sum(cfg["Gco2"]*(model[:vCHP_G][t]+model[:vABP_G][t]+model[:vWH_G][t]) +
            cfg["Lco2"]*(model[:vCHP_L][t]+model[:vABP_L][t]+model[:vWH_L][t])
            for t=1:tm["P"])/1e3)

    # CO2 emissions from grid purchases [ton]
    @expression(model, CO2_E, sum(tm["TM"][t]*model[:vQbuy][t]*tm["Qco2"][t] for t=1:tm["P"])/1e3)

    # Define the objective function
    @objective(model,Min,COST)

end
