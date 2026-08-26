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
    @variable(model, topo["chp_bdg"][c,l] >= 
        vCHPbdg[t=1:tm["P"],c=1:chp["N"],l=1:2] >= 0)
    # energy --heating-- from CHP to absorption chiller [0,efficiency]
    @variable(model, topo["chp_abp"][c,a] >= 
        vCHPabp[t=1:tm["P"],c=1:chp["N"],a=1:abp["N"]] >= 0)
    # supplemental gaseous or liquid fuel firing 'f' from CHP to absorption chiller
    #   or hor air, hot water [0,1]
    @variable(model, (!iszero).(topo["chp_fire"][c,l,f]) >= 
        vCHPfire[t=1:tm["P"],c=1:chp["N"],l=1:(abp["N"]+2),f=1:2] >= 0)
    # active link CHP to absorption chiller {0,1}
    @variable(model, bCHPabp[c=1:chp["N"],a=1:abp["N"]], Bin)
    # firing fuel in CHP link {0,1} f=1:Gas, f=2:Liquid
    @variable(model, bCHPfire[c=1:chp["N"],f=1:2], Bin)
    # heating mode in building {0,1}
    @variable(model, bBDGht[t=1:tm["P"]], Bin)
    # cooling mode in building {0,1}
    @variable(model, bBDGac[t=1:tm["P"]], Bin)

    # gaseous fuel purchased by supplemental firing [kWh]
    @expression(model, vTH_G[t=1:tm["P"]],
        tm["TM"][t]*sum(topo["chp_fire"][c,l,1]*vCHPfire[t,c,l,1] for c=1:chp["N"],l=1:abp["N"]+2))
    # liquid fuel purchased by supplemental firing [kWh]
    @expression(model, vTH_L[t=1:tm["P"]],
        tm["TM"][t]*sum(topo["chp_fire"][c,l,2]*vCHPfire[t,c,l,2] for c=1:chp["N"],l=1:abp["N"]+2))

    # heating allowed to building from CHP [0,1]
    @constraint(model, eBDGhtchp[t=1:tm["P"],c=1:chp["N"]; (!iszero).(topo["chp_bdg"][c,1])],
        vCHPbdg[t,c,1] <= bBDGht[t])
    # heating/cooling mode in building {0,1}
    @constraint(model, eBDGhtac[t=1:tm["P"]], bBDGht[t]+bBDGac[t] <= 1)
    # active link from CHP to absorption chiller {0,1}
    @constraint(model, eCHPabp[t=1:tm["P"],c=1:chp["N"],a=1:abp["N"]; (!iszero).(topo["chp_abp"][c,a])],
        vCHPabp[t,c,a] <= bCHPabp[c,a])
    # supplemental firing allowed when link from CHP to absorption chiller {0,1}
    @constraint(model, eCHPfireA[t=1:tm["P"],c=1:chp["N"],a=1:abp["N"],f=1:2; (!iszero).(topo["chp_fire"][c,a,f])],
        vCHPfire[t,c,a,f] <= bCHPabp[c,a])
    # unique type of firing fuel in CHP {0,1} f=1:Gas, f=2:Liquid
    @constraint(model, eCHPfire[t=1:tm["P"],c=1:chp["N"],a=1:abp["N"]+2,f=1:2; (!iszero).(topo["chp_fire"][c,a,f])],
        vCHPfire[t,c,a,f] <= bCHPfire[c,f])
    @constraint(model, eCHPfireF[c=1:chp["N"]; any(!iszero, topo["chp_fire"][c,:,:])],
        sum(bCHPfire[c,f] for f=1:2) <= 1)
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