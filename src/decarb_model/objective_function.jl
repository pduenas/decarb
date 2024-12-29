"""
objective_function!(model::Model,in::Dict,tm::Dict,chp::Dict,abs::Dict,hvac::Dict,
    wh::Dict,pv::Dict,bess::Dict,ev::Dict,wind::Dict)

Creates variables, expressions and constraints associated with the objective function

inputs:
model   name of core model
in      dictionary with miscellaneous input data
tm      dictionary with time series data
chp     dictionary with CHP data
abs     dictionary with absorption chiller data
hvac    dictionary with HVAC data
wh      dictionary with water heater data
pv      dictionary with PV data
bess    dictionary with BESS data
ev      dictionary with EV data
wind    dictionary with wind turbine data

"""
function objective_function!(model::Model,in::Dict,tm::Dict,chp::Dict,abs::Dict,hvac::Dict,
    wh::Dict,pv::Dict,bess::Dict,ev::Dict,wind::Dict)

    # cost of non-served electricity [$]
    @expression(model, vNSEcost[t=1:tm["P"]], in["NSEcost"]*model[:vNSE_Q][t])
    # cost of non-served temperature [$]
    @expression(model, vNSTcost[t=1:tm["P"]], in["NSTcost"]*(model[:vTup][t]+model[:vTlo][t]))
    # cost of non-served hot water [$]
    @expression(model, vNSHWcost[t=1:tm["P"]], in["NSHWcost"]*tm["HWdem"][t]*model[:vNShw][t])
    # cost of non-served hot water [$]
    @expression(model, vNSEVcost[t=1:tm["P"]],
        in["NSEVcost"]*sum(model[:vEVlo][t,e] for e=1:ev["N"]))
    # cost of purchasing electricity [$]
    @expression(model, vQcost[t=1:tm["P"]], tm["QcostBuy"][t]*model[:vQbuy][t])
    # income from selling electricity [$]
    @expression(model, vQearn[t=1:tm["P"]], tm["QcostSell"][t]*model[:vQsell][t])
    # cost of purchased fuel [$]
    @expression(model, vGLcost[t=1:tm["P"]], 
        tm["Gcost"][t]*(model[:vCHP_G][t]+model[:vABS_G][t]+model[:vWH_G][t]+model[:vTH_G][t]) + 
        tm["Lcost"][t]*(model[:vCHP_L][t]+model[:vABS_L][t]+model[:vWH_L][t]+model[:vTH_L][t]))
    
    # penalties for driver type: range anxious (-1), indifferent (0), battery concious (1)
    @expression(model, vEVpen,
        sum(in["EVdriver"]*in["EVpen"]*model[:vEVsoc][t,e] for t=1:tm["P"],e=1:ev["N"]))
    
    # cost of peak capacity subscription [$]
    @expression(model, vQmxCost, sum(tm["QmxCostN"][n]*model[:vQmx][n] for n=1:in["QmxTM"]))
    
    # annualized cost of CHP during time scope [$]
    b = has_upper_bound.(model[:bCHPty]).==true     # for potential investments
    @expression(model, vCHPinv,
        sum(b[i,c]*(tm["H"]-tm["IW"][i])/8760*chp["inv"][c]*in["IR"]/
        (1-(1+in["IR"])^(-chp["life"][c]))*model[:bCHPty][i,c] for i=1:in["IT"],c=1:chp["N"]))
    # annualized cost of HVAC during time scope [$]
    b = has_upper_bound.(model[:bHVACty]).==true    # for potential investments
    @expression(model, vHVACinv,
        sum(b[i,h]*(tm["H"]-tm["IW"][i])/8760*hvac["inv"][h]*in["IR"]/
        (1-(1+in["IR"])^(-hvac["life"][h]))*model[:bHVACty][i,h] for i=1:in["IT"],h=1:hvac["N"]))
    # annualized cost of absorption chiller during time scope [$]
    b = has_upper_bound.(model[:bABSty]).==true     # for potential investments
    @expression(model, vABSinv, 
        sum(b[i,a]*(tm["H"]-tm["IW"][i])/8760*abs["inv"][a]*in["IR"]/
        (1-(1+in["IR"])^(-abs["life"][a]))*model[:bABSty][i,a] for i=1:in["IT"],a=1:abs["N"]))
    # annualized cost of water heater during time scope [$]
    b = has_upper_bound.(model[:bWHty]).==true     # for potential investments
    @expression(model, vWHinv, 
        sum(b[i,w]*(tm["H"]-tm["IW"][i])/8760*wh["inv"][w]*in["IR"]/
        (1-(1+in["IR"])^(-wh["life"][w]))*model[:bWHty][i,w] for i=1:in["IT"],w=1:wh["N"]))
    # annualized cost of PV during time scope [$]
    b = has_upper_bound.(model[:zPV]).==true       # for potential investments
        @expression(model, vPVinv,
        sum(b[i,v]*(tm["H"]-tm["IW"][i])/8760*pv["inv"][v]*in["IR"]/
        (1-(1+in["IR"])^(-pv["life"][v]))*model[:zPV][i,v] for i=1:in["IT"],v=1:pv["N"]))
    # annualized cost of wind turbine during time scope [$]
    b = has_upper_bound.(model[:zWIND]).==true     # for potential investments
        @expression(model, vWINDinv, 
        sum(b[i,d]*(tm["H"]-tm["IW"][i])/8760*wind["inv"][d]*in["IR"]/
        (1-(1+in["IR"])^(-wind["life"][d]))*model[:zWIND][i,d] for i=1:in["IT"],d=1:wind["N"]))
    # annualized cost of electricity storage during time scope [$]
    b = has_upper_bound.(model[:zBESS]).==true     # for potential investments
        @expression(model, vBESSinv, 
        sum(b[i,s]*(tm["H"]-tm["IW"][i])/8760*bess["inv"][s]*in["IR"]/
        (1-(1+in["IR"])^(-bess["life"][s]))*model[:zBESS][i,s] for i=1:in["IT"],s=1:bess["N"]))

    # total fixed O&M costs (+) [$]
    @expression(model, COST_FOM,
        sum(chp["fom"][c]*model[:bCHPty][i,c]*tm["H"]/8760 for i=1:in["IT"],c=1:chp["N"]) +
    	sum(abs["fom"][a]*model[:bABSty][i,a]*tm["H"]/8760 for i=1:in["IT"],a=1:abs["N"]) +
    	sum(hvac["fom"][h]*model[:bHVACty][i,h]*tm["H"]/8760 for i=1:in["IT"],h=1:hvac["N"]) +
    	sum(wh["fom"][w]*model[:bWHty][i,w]*tm["H"]/8760 for i=1:in["IT"],w=1:wh["N"]) +
    	sum(pv["fom"][v]*model[:zPV][i,v]*tm["H"]/8760 for i=1:in["IT"],v=1:pv["N"]) +
    	sum(wind["fom"][d]*model[:zWIND][i,d]*tm["H"]/8760 for i=1:in["IT"],d=1:wind["N"]) +
    	sum(bess["fom"][s]*model[:zBESS][i,s]*tm["H"]/8760 for i=1:in["IT"],s=1:bess["N"]))
    # total variable O&M costs (+) [$]
    @expression(model, COST_VOM,
        sum(chp["vom"][c]*model[:vCHP_Q][t,c] for t=1:tm["P"],c=1:chp["N"]) +
        sum(abs["vom"][a]*model[:vABS_AC][t,a] for t=1:tm["P"],a=1:abs["N"]) +
        sum(hvac["vom"][h]*model[:vHVAC_HTAC][t,h] for t=1:tm["P"],h=1:hvac["N"]) +
        sum(wh["vom"][w]*model[:vWH_HW][t,w] for t=1:tm["P"],w=1:wh["N"]))
    
    # total variable costs (+) / total incomes (-) [$]
    @expression(model, COST_VAR, 
        sum(vNSEcost[t]+vNSTcost[t]+vNSHWcost[t]+vNSEVcost[t]+vQcost[t]-vQearn[t]+vGLcost[t]
            for t=1:tm["P"]) + vQmxCost + COST_VOM)
    # total annualized costs
    @expression(model, COST_INV, vCHPinv+vHVACinv+vABSinv+vWHinv+vPVinv+vWINDinv+vBESSinv + COST_FOM)
    # total costs [$]
    @expression(model, COST, COST_VAR + COST_INV + vEVpen)

    # CO2 emissions from building [ton]
    @expression(model, CO2_B, 
        sum(in["Gco2"]*(model[:vCHP_G][t]+model[:vABS_G][t]+model[:vWH_G][t]+model[:vTH_G][t]) + 
            in["Lco2"]*(model[:vCHP_L][t]+model[:vABS_L][t]+model[:vWH_L][t]+model[:vTH_L][t]) 
            for t=1:tm["P"])/1e6)

    # CO2 emissions from grid purchases [kg]
    @expression(model, CO2_E, sum(model[:vQbuy][t]*tm["Qco2"][t] for t=1:tm["P"]))

    # Define the objective function
    @objective(model,Min,COST)

end