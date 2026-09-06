"""
abp_chillers!(model::Model,cfg::Dict,tm::Dict,bdg::Dict,topo::Dict,sp::Dict,chp::Dict,
    abp::Dict)

Creates variables, expressions and constraints associated to absorption chillers

inputs:
model   name of core model
cfg     dictionary with configuration input data
tm      dictionary with time series data
bdg     dictionary with building data
topo    dictionary with topology of thermal connections
sp      dictionary with equipment selection data
chp     dictionary with CHP data
abp     dictionary with absorption chiller data

"""
function abp_chillers!(model::Model,cfg::Dict,tm::Dict,bdg::Dict,topo::Dict,sp::Dict,
    chp::Dict,abp::Dict)

    # unitary fuel consumption by absorption chiller [0,1]
    @variable(model, 1 >= vABPq[t=1:tm["P"],a=1:abp["N"]] >= 0)    
    # unitary cooling generated in absorption chiller (0,1)
    @variable(model, 1 >= vABPac[t=1:tm["P"],a=1:abp["N"]] >= 0)
    # investment in absorption chiller at investment window {0,1}
    @variable(model, bABPty[i=1:cfg["IT"],a=1:abp["N"]], Bin)
    # existing absorption chiller along simulation {0,1}
    @variable(model, bABP_u[t=1:tm["P"],a=1:abp["N"]], Bin)

    # define bounds of binary investment variable
    set_upper_bound.(bABPty[1,:],0)
    for a in findall(sp["ABP0"].!="0")
        # fix already installed absorption chillers
        for u=1:sp["ABPz0"][a]
            set_upper_bound.(bABPty[1,abp["ty"].==string(sp["ABP0"][a],'-',u)],1)
            set_lower_bound.(bABPty[1,abp["ty"].==string(sp["ABP0"][a],'-',u)],1)
        end
        # release potential installed absorption chillers
        if sp["ABPyn"][a] == "YES"
            for u=(sp["ABPz0"][a]+1):bdg["Babp"]
                set_upper_bound.(bABPty[1,abp["ty"].==string(sp["ABP0"][a],'-',u)],1)
            end
        end
    end
    # correct maximum capacity for non-existing or disabled absorption chillers
    abp["mx"][iszero.(upper_bound.(bABPty[1,:]))] .= 0

    # zero fuel consumption when disabled
    set_upper_bound.(vABPq[:,abp["fuel"].=="0"],0)

    # fuel consumption for cooling [kWh]
    @expression(model, vABP_Q[t=1:tm["P"],a=1:abp["N"]; abp["ac"][a]>0],
        tm["TM"][t]*abp["mx"][a]*vABPq[t,a]/abp["ac"][a])
    # cooling provided by absorption chiller [kWh]
    @expression(model, vABP_AC[t=1:tm["P"],a=1:abp["N"]], tm["TM"][t]*abp["mx"][a]*vABPac[t,a])

    # gaseous fuel purchased by absorption chillers [kWh]
    if any(abp["fuel"].=="G")
        @expression(model, vABP_G[t=1:tm["P"]],
            sum(vABP_Q[t,a]/abp["fcf"][a] for a=1:abp["N"] if abp["fcf"][a]>0 && abp["fuel"][a]=="G"))
    else
        @variable(model, vABP_G[t=1:tm["P"]] == 0)
    end

    # liquid fuel purchased by absorption chillers [kWh]
    if any(abp["fuel"].=="L")
        @expression(model, vABP_L[t=1:tm["P"]],
            sum(vABP_Q[t,a]/abp["fcf"][a] for a=1:abp["N"] if abp["fcf"][a]>0 && abp["fuel"][a]=="L"))
    else
        @variable(model, vABP_L[t=1:tm["P"]] == 0)
    end

    # maximum available space for absorption chillers {0,Babp}
    @constraint(model, eABPbdg,
        sum(bABPty[i,a] for i=1:cfg["IT"],a=1:abp["N"]) <= bdg["Babp"])
    # only one investment per time window for absorption chillers {0,1}
    @constraint(model, eABPw[a=1:abp["N"]], sum(bABPty[i,a] for i=1:cfg["IT"]) <= 1)
    # maximum cooling provided by absorption chiller (0,1)
     @constraint(model, eABPac[t=1:tm["P"],a=1:abp["N"]], vABPac[t,a] <= bABP_u[t,a])
    # investment in absorption chiller {0,1}
    for i1=1:cfg["IT"]
    	 @constraint(model, eABPb[t=tm["IW"][i1]:tm["P"],a=1:abp["N"]],
    		bABP_u[t,a] == sum(bABPty[i2,a] for i2=1:i1))
    end
    # allowed cooling to building from absorption chiller {0,1}
    @constraint(model, eABPacbdg[t=1:tm["P"],a=1:abp["N"]], vABPac[t,a] <= model[:bBDGac][t])
    # heat consumption by absorption chiller [0,1]
    @constraint(model, eABPq[t=1:tm["P"],a=1:abp["N"]; abp["mx"][a]>0],
        vABPac[t,a] <= vABPq[t,a] + 
        sum(topo["chp_abp"][c,a]*chp["mx"][c]*model[:vCHPabp][t,c,a]/abp["mx"][a] for c=1:chp["N"] if topo["chp_abp"][c,a]>0) +
        sum(topo["chp_fire"][c,a,f]*model[:vCHPfire][t,c,a,f]/abp["mx"][a] for c=1:chp["N"],f=1:2 if topo["chp_fire"][c,a,f]>0))

end