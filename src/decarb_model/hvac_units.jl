"""
hvac_units!(model::Model,in::Dict,tm::Dict,bdg::Dict,topo::Dict,sp::Dict,hvac::Dict)

Creates variables, expressions and constraints associated to HVAC units

inputs:
model   name of core model
in      dictionary with miscellaneous input data
tm      dictionary with time series data
bdg     dictionary with building data
sp      dictionary with equipment selection data
hvac    dictionary with HVAC data

"""
function hvac_units!(model::Model,in::Dict,tm::Dict,bdg::Dict,sp::Dict,hvac::Dict)

    # electricity unitary consumption for heating [0,1]
    @variable(model, (!iszero).(hvac["HVmx"][h]) >= vHVACht[t=1:tm["P"],h=1:hvac["N"]] >= 0)
    # electricity unitary consumption for cooling [0,1]
    @variable(model, (!iszero).(hvac["ACmx"][h]) >= vHVACac[t=1:tm["P"],h=1:hvac["N"]] >= 0)
    # heating mode of HVAC unit along simulation {0,1}
    @variable(model, bHVACht[t=1:tm["P"],h=1:hvac["N"]], Bin)
    # cooling mode of HVAC unit along simulation {0,1}
    @variable(model, bHVACac[t=1:tm["P"],h=1:hvac["N"]], Bin)
    # investment in HVAC unit at investment window {0,1}
    @variable(model, bHVACty[i=1:in["IT"],h=1:hvac["N"]], Bin)
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

    # electricity consumption for heating [kWh]
    @expression(model, vHVAC_HT[t=1:tm["P"],h=1:hvac["N"]],
        tm["TM"][t]*hvac["HVmx_k"][t,h]*vHVACht[t,h]/hvac["HVeff_k"][t,h])
    # electricity consumption for cooling [kWh]
    @expression(model, vHVAC_AC[t=1:tm["P"],h=1:hvac["N"]],
        tm["TM"][t]*hvac["ACmx_k"][t,h]*vHVACac[t,h]/hvac["ACeff_k"][t,h])
    # heat(+)/cool(-) provided by HVAC [kWh]
    @expression(model, vHVAC_HTAC[t=1:tm["P"],h=1:hvac["N"]],
        hvac["HVmx_k"][t,h]*vHVACht[t,h]-hvac["ACmx_k"][t,h]*vHVACac[t,h])
    
    # maximum available space for HVAC units [0,Bhvac]
    @constraint(model, eHVACbdg,
    	sum(bHVACty[i,h] for i=1:in["IT"],h=1:hvac["N"] if hvac["HVmx"][h]>0 || hvac["ACmx"][h]>0) <= bdg["Bhvac"])
    # only one investment per time window for HVAC units {0,1}
    @constraint(model, eHVACw[h=1:hvac["N"]; hvac["HVmx"][h]>0 || hvac["ACmx"][h]>0],
        sum(bHVACty[i,h] for i=1:in["IT"]) <= 1)
    # heating/cooling mode of HVAC unit {0,1}
    @constraint(model, eHVAChtac[t=1:tm["P"],h=1:hvac["N"]; hvac["HVmx"][h]>0 || hvac["ACmx"][h]>0],
        bHVACht[t,h]+bHVACac[t,h] <= bHVAC_u[t,h])
    # investment in HVAC unit {0,1}
    for i1=1:in["IT"]
    	eHVACb = @constraint(model, [t=tm["IW"][i1]:tm["P"],h=1:hvac["N"]; hvac["HVmx"][h]>0 || hvac["ACmx"][h]>0],
    		bHVAC_u[t,h] <= sum(bHVACty[i2,h] for i2=1:i1))
    end
    # maximum heat provided by HVAC (0,1)
    @constraint(model, eHVACht[t=1:tm["P"],h=1:hvac["N"]; hvac["HVmx"][h]>0],
        vHVACht[t,h] <= bHVACht[t,h])
    # maximum cold provided by HVAC (0,1)
    @constraint(model, eHVACac[t=1:tm["P"],h=1:hvac["N"]; hvac["ACmx"][h]>0],
        vHVACac[t,h] <= bHVACac[t,h])

end