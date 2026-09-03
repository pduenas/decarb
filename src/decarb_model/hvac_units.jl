"""
hvac_units!(model::Model,cfg::Dict,tm::Dict,bdg::Dict,topo::Dict,sp::Dict,hvac::Dict)

Creates variables, expressions and constraints associated to HVAC units

inputs:
model   name of core model
cfg     dictionary with configuration input data
tm      dictionary with time series data
bdg     dictionary with building data
sp      dictionary with equipment selection data
hvac    dictionary with HVAC data

"""
function hvac_units!(model::Model,cfg::Dict,tm::Dict,bdg::Dict,sp::Dict,hvac::Dict)

    # heating mode of HVAC unit along simulation {0,1}
    @variable(model, bHVACht[t=1:tm["P"],h=1:hvac["N"]], Bin)
    # cooling mode of HVAC unit along simulation {0,1}
    @variable(model, bHVACac[t=1:tm["P"],h=1:hvac["N"]], Bin)
    # investment in HVAC unit at investment window {0,1}
    @variable(model, bHVACty[i=1:cfg["IT"],h=1:hvac["N"]], Bin)
    # existing HVAC along simulation {0,1}
    @variable(model, bHVAC_u[t=1:tm["P"],h=1:hvac["N"]], Bin)

    # define bounds of binary investment variable
    set_upper_bound.(bHVACty[1,:],0)
    for h in findall(sp["HVAC0"].!="0")
        # fix already installed HVAC units
        for u=1:sp["HVACz0"][h]
            set_upper_bound.(bHVACty[1,hvac["ty"].==string(sp["HVAC0"][h],'-',u)],1)
            set_lower_bound.(bHVACty[1,hvac["ty"].==string(sp["HVAC0"][h],'-',u)],1)
        end
        # release potential installed HVAC units
        if sp["HVACyn"][h] == "YES"
            for u=(sp["HVACz0"][h]+1):bdg["Bhvac"]
                set_upper_bound.(bHVACty[1,hvac["ty"].==string(sp["HVAC0"][h],'-',u)],1)
            end
        end
    end
    # correct maximum capacity for non-existing or disabled HVAC units
    hvac["HVmx"][iszero.(upper_bound.(bHVACty[1,:]))] .= 0
    hvac["ACmx"][iszero.(upper_bound.(bHVACty[1,:]))] .= 0

    # load temperature variables
    vTin = model[:vTin]

    # electricity unitary consumption for heating [0,1]
    @variable(model, 1 >= vHVACht[t=1:tm["P"], h=1:hvac["N"]] >= 0)
    # electricity unitary consumption for cooling [0,1]
    @variable(model, 1 >= vHVACac[t=1:tm["P"], h=1:hvac["N"]] >= 0)

    # auxiliary variables for heating/cooling derating capacity against indoor temperature
    @variable(model, vTdiffHT[t=1:tm["P"]] >= 0)
    @variable(model, vTdiffAC[t=1:tm["P"]] >= 0)
    @constraint(model, [t=1:tm["P"]], vTdiffHT[t] >= vTin[t] - tm["Tout"][t])
    @constraint(model, [t=1:tm["P"]], vTdiffAC[t] >= tm["Tout"][t] - vTin[t])

    # maximum heat provided by HVAC (0,1)
    @constraint(model, eHVACmxHT[t=1:tm["P"], h=1:hvac["N"]; hvac["HVmx"][h]>0],
        1 - hvac["HVmx_"][h]*vTdiffHT[t] >= vHVACht[t,h])
    # maximum cold provided by HVAC (0,1)
    @constraint(model, eHVACmxAC[t=1:tm["P"], h=1:hvac["N"]; hvac["ACmx"][h]>0],
        1 - hvac["ACmx_"][h]*vTdiffAC[t] >= vHVACac[t,h])

    # electricity consumption for heating [kWh]
    @expression(model, vHVAC_HT[t=1:tm["P"],h=1:hvac["N"]],
        tm["TM"][t]*hvac["HVmx"][h]*vHVACht[t,h]/hvac["HVeff_k"][t,h])
    # electricity consumption for cooling [kWh]
    @expression(model, vHVAC_AC[t=1:tm["P"],h=1:hvac["N"]],
        tm["TM"][t]*hvac["ACmx"][h]*vHVACac[t,h]/hvac["ACeff_k"][t,h])    
    # heat(+)/cool(-) provided by HVAC [kWh]
    @expression(model, vHVAC_HTAC[t=1:tm["P"],h=1:hvac["N"]],
        tm["TM"][t]*(hvac["HVmx"][h]*vHVACht[t,h]-hvac["ACmx"][h]*vHVACac[t,h]))
    
    # maximum available space for HVAC units [0,Bhvac]
    @constraint(model, eHVACbdg,
    	sum(bHVACty[i,h] for i=1:cfg["IT"],h=1:hvac["N"] if hvac["HVmx"][h]>0 || hvac["ACmx"][h]>0) <= bdg["Bhvac"])
    # only one investment per time window for HVAC units {0,1}
    @constraint(model, eHVACw[h=1:hvac["N"]; hvac["HVmx"][h]>0 || hvac["ACmx"][h]>0],
        sum(bHVACty[i,h] for i=1:cfg["IT"]) <= 1)
    # heating/cooling mode of HVAC unit {0,1}
    @constraint(model, eHVAChtac[t=1:tm["P"],h=1:hvac["N"]; hvac["HVmx"][h]>0 || hvac["ACmx"][h]>0],
        bHVACht[t,h]+bHVACac[t,h] <= bHVAC_u[t,h])
    # investment in HVAC unit {0,1}
    for i1=1:cfg["IT"]
    	 @constraint(model, eHVACb[t=tm["IW"][i1]:tm["P"],h=1:hvac["N"]; hvac["HVmx"][h]>0 || hvac["ACmx"][h]>0],
    		bHVAC_u[t,h] == sum(bHVACty[i2,h] for i2=1:i1))
    end
    # maximum heat provided by HVAC (0,1)
    @constraint(model, eHVACht[t=1:tm["P"],h=1:hvac["N"]; hvac["HVmx"][h]>0],
        vHVACht[t,h] <= bHVACht[t,h])
    # maximum cold provided by HVAC (0,1)
    @constraint(model, eHVACac[t=1:tm["P"],h=1:hvac["N"]; hvac["ACmx"][h]>0],
        vHVACac[t,h] <= bHVACac[t,h])
end