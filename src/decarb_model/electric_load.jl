"""
electric_load!(model::Model,cfg::Dict,tm::Dict,chp::Dict,hvac::Dict,wh::Dict,pv::Dict,
    bess::Dict,ev::Dict)

Creates variables, expressions and constraints associated with electric load balance

inputs:
model   name of core model
cfg     dictionary with configuration input data
tm      dictionary with time series data
chp     dictionary with CHP data
hvac    dictionary with HVAC data
wh      dictionary with water heater data
pv      dictionary with PV data
wind    dictionary with wind turbine data
bess    dictionary with BESS data
ev      dictionary with EV data

"""
function electric_load!(model::Model,cfg::Dict,tm::Dict,chp::Dict,hvac::Dict,wh::Dict,
    pv::Dict,wind::Dict,bess::Dict,ev::Dict)

    # unitary non-served electricity [0,1]
    @variable(model, 1 >= vNSEq[t=1:tm["P"]] >= 0)
    # peak power demand [kW]
    @variable(model, vQmx[n=1:cfg["QmxTM"]] >= 0)
    # purchased electricity [kW]
    @variable(model, cfg["QmxBuy"] >= vQbuy[t=1:tm["P"]] >= 0)
    # sold electricity [kW]
    @variable(model, cfg["QmxSell"] >= vQsell[t=1:tm["P"]] >= 0)
    # purchase/sale electricity mode {0,1}
    @variable(model, bQbs[t=1:tm["P"]], Bin)

    # disable non-served energy if indicated
    cfg["NSEcost"]==0 ? fix.(vNSEq,0; force=true) : println("   \u2139  Non-served energy allowed")

    # non-served electricity [kWh]
    @expression(model, vNSE_Q[t=1:tm["P"]], tm["TM"][t]*(tm["Qlight"][t]+tm["Qequip"][t])*vNSEq[t])
    # electricity demand [kWh]
    @expression(model, vQdem[t=1:tm["P"]],
        tm["TM"][t]*(tm["Qlight"][t]+tm["Qequip"][t]) + 
        sum(model[:vHVAC_HT][t,h]+model[:vHVAC_AC][t,h] for h=1:hvac["N"]; init=0.0) + 
        sum(model[:vBESS_UP][t,s] for s=1:bess["N"] if bess["zmx0"][s]>0) + 
        sum(model[:vEV_UP][t,e] for e=1:ev["N"] if ev["mx_eff"][e]>0) + 
        sum(model[:vWH_Q][t,w] for w=1:wh["N"] if wh["mx_eff"][w]>0 && wh["fuel"][w]=="0"))
    # electricity generated [kWh]
    @expression(model, vQgen[t=1:tm["P"]],
        sum(model[:vCHP_Q][t,c] for c=1:chp["N"] if chp["mx_eff"][c]>0) + 
        sum(model[:vPV_Q][t,v] for v=1:pv["N"] if pv["zmx0"][v]>0) + 
        sum(model[:vWIND_Q][t,d] for d=1:wind["N"] if wind["zmx0"][d]>0) + 
        sum(model[:vBESS_DN][t,s] for s=1:bess["N"] if bess["zmx0"][s]>0) + 
        sum(model[:vEV_DN][t,e] for e=1:ev["N"] if ev["mx_eff"][e]>0))

    # electricity balance [kWh]
    @constraint(model, eQbal[t=1:tm["P"]], vQgen[t]+tm["TM"][t]*(vQbuy[t]-vQsell[t])+vNSE_Q[t] == vQdem[t])
    # peak power load over time scope [kW]
    @constraint(model, eDmx[t=1:tm["P"],n=1:cfg["QmxTM"]; tm["Qmx"][t]==n && tm["QmxCost"][t]>0],
        vQmx[n] >= vQbuy[t])
    # purchase electricity mode [kW]
    @constraint(model, eQbuy[t=1:tm["P"]], vQbuy[t] <= cfg["QmxBuy"]*bQbs[t])
    # sell electricity mode [kW]
    @constraint(model, eQsell[t=1:tm["P"]], vQsell[t] <= cfg["QmxSell"]*(1-bQbs[t]))

end