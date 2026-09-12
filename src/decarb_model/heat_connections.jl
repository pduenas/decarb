"""
heat_connections!(model::Model,topo::Dict,tm::Dict,chp::Dict,abp::Dict)

Creates variables, expressions and constraints associated to heat transfer connections

inputs:
model   name of core model
topo    dictionary with topology of thermal connections
tm      dictionary with time series data
chp     dictionary with CHP data
abp     dictionary with absorption chiller data

"""
function heat_connections!(model::Model,topo::Dict,tm::Dict,chp::Dict,abp::Dict)

    # energy --heating-- from CHP to building: hot air, hot water [0,efficiency]
    @variable(model, (!iszero).(topo["chp_bdg"][c,l]) >= 
        vCHPbdg[t=1:tm["P"],c=1:chp["N"],l=1:2] >= 0)
    # energy --heating-- from CHP to absorption chiller [0,efficiency]
    @variable(model, (!iszero).(topo["chp_abp"][c,a]) >= 
        vCHPabp[t=1:tm["P"],c=1:chp["N"],a=1:abp["N"]] >= 0)
    # active link CHP to absorption chiller {0,1}
    @variable(model, bCHPabp[c=1:chp["N"],a=1:abp["N"]], Bin)
    # heating mode in building {0,1}
    @variable(model, bBDGht[t=1:tm["P"]], Bin)
    # cooling mode in building {0,1}
    @variable(model, bBDGac[t=1:tm["P"]], Bin)

    # heating allowed to building from CHP [0,1]
    @constraint(model, eBDGhtchp[t=1:tm["P"],c=1:chp["N"]; (!iszero).(topo["chp_bdg"][c,1])],
        vCHPbdg[t,c,1] <= bBDGht[t])
    # heating/cooling mode in building {0,1}
    @constraint(model, eBDGhtac[t=1:tm["P"]], bBDGht[t]+bBDGac[t] <= 1)
    # active link from CHP to absorption chiller {0,1}
    @constraint(model, eCHPabp[t=1:tm["P"],c=1:chp["N"],a=1:abp["N"]; (!iszero).(topo["chp_abp"][c,a])],
        vCHPabp[t,c,a] <= bCHPabp[c,a])
    # unique active link from CHP to absorption chiller, and viceversa {0,1}
    if abp["N"]>0
        @constraint(model, eCHPabpC[c=1:chp["N"]; any((!iszero).(topo["chp_abp"][c,:]))],
            sum(bCHPabp[c,a] for a=1:abp["N"]) <= 1)
    end
    if chp["N"]>0
        @constraint(model, eCHPabpA[a=1:abp["N"]; any((!iszero).(topo["chp_abp"][:,a]))],
            sum(bCHPabp[c,a] for c=1:chp["N"]) <= 1)
    end

end