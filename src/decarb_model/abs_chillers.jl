"""
abs_chillers!(model::Model,in::Dict,tm::Dict,bdg::Dict,topo::Dict,sp::Dict,chp::Dict,
    abs::Dict)

Creates variables, expressions and constraints associated to absorption chillers

inputs:
model   name of core model
in      dictionary with miscellaneous input data
tm      dictionary with time series data
bdg     dictionary with building data
topo    dictionary with topology of thermal connections
sp      dictionary with equipment selection data
chp     dictionary with CHP data
abs     dictionary with absorption chiller data

"""
function abs_chillers!(model::Model,in::Dict,tm::Dict,bdg::Dict,topo::Dict,sp::Dict,
    chp::Dict,abs::Dict)

    # unitary fuel consumption by absorption chiller [0,1]
    @variable(model, 1 >= vABSq[t=1:tm["P"],a=1:abs["N"]] >= 0)    
    # unitary cooling generated in absorption chiller (0,1)
    @variable(model, 1 >= vABSac[t=1:tm["P"],a=1:abs["N"]] >= 0)
    # investment in absorption chiller at investment window {0,1}
    @variable(model, bABSty[i=1:in["IT"],a=1:abs["N"]], Bin)
    # existing absorption chiller along simulation {0,1}
    @variable(model, bABS_u[t=1:tm["P"],a=1:abs["N"]], Bin)

    # define bounds of binary investment variable
    set_upper_bound.(bABSty[1,:],0)
    for a in findall(sp["ABS0"].!="0")
        # fix already installed absorption chillers
        for u=1:sp["ABSz0"][a]
            set_upper_bound.(bABSty[1,abs["ty"].==string(sp["ABS0"][a],'-',u)],1)
            set_lower_bound.(bABSty[1,abs["ty"].==string(sp["ABS0"][a],'-',u)],1)
        end
        # release potential installed absorption chillers
        if sp["ABSyn"][a] == "YES"
            for u=(sp["ABSz0"][a]+1):bdg["Babs"]
                set_upper_bound.(bABSty[1,abs["ty"].==string(sp["ABS0"][a],'-',u)],1)
            end
        end
    end
    # correct maximum capacity for non-existing or disabled absorption chillers
    abs["mx"][iszero.(upper_bound.(bABSty[1,:]))] .= 0

    # zero fuel consumption when disabled
    set_upper_bound.(vABSq[:,abs["fuel"].=="0"],0)

    # fuel consumption for cooling [kWh]
    @expression(model, vABS_Q[t=1:tm["P"],a=1:abs["N"]], tm["TM"][t]*abs["mx"][a]*vABSq[t,a]/abs["ac"][a])
    # cooling provided by absorption chiller [kWh]
    @expression(model, vABS_AC[t=1:tm["P"],a=1:abs["N"]], tm["TM"][t]*abs["mx"][a]*vABSac[t,a])

    # gaseous fuel purchased by absorption chillers [MMBtu]
    if any(abs["fuel"].=="G")
        @expression(model, vABS_G[t=1:tm["P"]],
            sum(abs["hr"][a]*vABS_Q[t,a] for a=1:abs["N"] if abs["hr"][a]>0 && abs["fuel"][a]=="G"))
    else
        @variable(model, vABS_G[t=1:tm["P"]] == 0)
    end

    # liquid fuel purchased by absorption chillers [MMBtu]
    if any(abs["fuel"].=="L")
        @expression(model, vABS_L[t=1:tm["P"]],
            sum(abs["hr"][a]*vABS_Q[t,a] for a=1:abs["N"] if abs["hr"][a]>0 && abs["fuel"][a]=="L"))
    else
        @variable(model, vABS_L[t=1:tm["P"]] == 0)
    end

    # maximum available space for absorption chillers {0,Babs}
    @constraint(model, eABSbdg,
        sum(bABSty[i,a] for i=1:in["IT"],a=1:abs["N"]) <= bdg["Babs"])
    # only one investment per time window for absorption chillers {0,1}
    @constraint(model, eABSw[a=1:abs["N"]], sum(bABSty[i,a] for i=1:in["IT"]) <= 1)
    # maximum cooling provided by absorption chiller (0,1)
     @constraint(model, eABSac[t=1:tm["P"],a=1:abs["N"]], vABSac[t,a] <= bABS_u[t,a])
    # investment in absorption chiller {0,1}
    for i1=1:in["IT"]
    	 @constraint(model, eABSb[t=tm["IW"][i1]:tm["P"],a=1:abs["N"]],
    		bABS_u[t,a] == sum(bABSty[i2,a] for i2=1:i1))
    end
    # allowed cooling to building from absorption chiller {0,1}
    @constraint(model, eABSacbdg[t=1:tm["P"],a=1:abs["N"]], vABSac[t,a] <= model[:bBDGac][t])
    # heat consumption by absorption chiller [0,1]
    @constraint(model, eABSq[t=1:tm["P"],a=1:abs["N"]; abs["mx"][a]>0],
        vABSac[t,a] <= vABSq[t,a] + 
        sum(topo["chp_abs"][c,a]*chp["mx"][c]*vCHPabs[t,c,a]/abs["mx"][a] for c=1:chp["N"] if topo["chp_abs"][c,a]>0) +
        sum(topo["chp_fire"][c,a,f]*vCHPfire[t,c,a,f]/abs["mx"][a] for c=1:chp["N"],f=1:2 if topo["chp_fire"][c,a,f]>0))

end