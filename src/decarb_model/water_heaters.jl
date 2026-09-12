"""
water_heaters!(model::Model,cfg::Dict,tm::Dict,bdg::Dict,sp::Dict,wh::Dict)

Creates variables, expressions and constraints associated to water heaters

inputs:
model   name of core model
cfg     dictionary with configuration input data
tm      dictionary with time series data
bdg     dictionary with building data
sp      dictionary with equipment selection data
wh      dictionary with WH data

"""
function water_heaters!(model::Model,cfg::Dict,tm::Dict,bdg::Dict,sp::Dict,wh::Dict)

    # energy consumption for water heating [0,1]
    @variable(model, (!iszero).(wh["mx"][w]) >= vWHq[t=1:tm["P"],w=1:wh["N"]] >= 0)
    # tank level in water heater [0,1]
    @variable(model, (!iszero).(wh["tank"][w]) >= vWHsoc[t=0:tm["P"],w=1:wh["N"]] >= 0)
    # hot water demand from water heater [0,1]
    @variable(model, (!iszero).(wh["mx"][w]) >= vWHhw[t=1:tm["P"],w=1:wh["N"]] >= 0)
    # investment in water heater [0,1]
    @variable(model, bWHty[i=1:cfg["IT"],w=1:wh["N"]], Bin)
    # existing water heater during period {0,1}
    @variable(model, bWH_u[t=1:tm["P"],w=1:wh["N"]], Bin)

    # define bounds of binary investment variable
    set_upper_bound.(bWHty,0)
    for w in findall(sp["WH0"].!="0")
        # fix already installed water heaters
        for u=1:sp["WHz0"][w]
            set_upper_bound.(bWHty[1,wh["ty"].==string(sp["WH0"][w],'-',u)],1)
            set_lower_bound.(bWHty[1,wh["ty"].==string(sp["WH0"][w],'-',u)],1)
        end
        # release potential installed CHP units
        if sp["WHyn"][w] == "YES"
            for u=(sp["WHz0"][w]+1):bdg["Bwh"]
                set_upper_bound.(bWHty[:,wh["ty"].==string(sp["WH0"][w],'-',u)],1)
            end
        end
    end

    # correct maximum capacity for non-existing or disabled water heaters
    wh["mx_eff"] = copy(wh["mx"])
    wh["tank_eff"] = copy(wh["tank"])
    wh["mx_eff"][iszero.(upper_bound.(bWHty[1,:]))] .= 0
    wh["tank_eff"][iszero.(upper_bound.(bWHty[1,:]))] .= 0

    # energy consumption for water heating [kWh]
    @expression(model, vWH_Q[t=1:tm["P"],w=1:wh["N"]], tm["TM"][t]*wh["mx_eff"][w]*vWHq[t,w])
    # hot water demand from water heater [kWh]
    @expression(model, vWH_HW[t=1:tm["P"],w=1:wh["N"]], tm["TM"][t]*wh["mx_eff"][w]*vWHhw[t,w])

    # gaseous fuel purchased by water heaters [kWh]
    if any(wh["fuel"].=="G")
        @expression(model, vWH_G[t=1:tm["P"]], sum(vWH_Q[t,w]/wh["fcf"][w] for w=1:wh["N"]
            if wh["fcf"][w]>0 && wh["fuel"][w]=="G"))
    else
        @variable(model, vWH_G[t=1:tm["P"]] == 0)
    end
    # liquid fuel purchased by water heaters [kWh]
    if any(wh["fuel"].=="L")
        @expression(model, vWH_L[t=1:tm["P"]], sum(vWH_Q[t,w]/wh["fcf"][w] for w=1:wh["N"]
            if wh["fcf"][w]>0 && wh["fuel"][w]=="L"))
    else
        @variable(model, vWH_L[t=1:tm["P"]] == 0)
    end

    # maximum available space for water heaters {0,Bwh}
    @constraint(model, eWHbdg,
        sum(bWHty[i,w] for i=1:cfg["IT"],w=1:wh["N"] if wh["mx_eff"][w]>0) <= bdg["Bwh"])
    # only one investment per time window in water heaters {0,1}
    @constraint(model, eWHw[w=1:wh["N"]; wh["mx_eff"][w]>0],
        sum(bWHty[i,w] for i=1:cfg["IT"]) <= 1)
    # maximum heat provided by water heater [0,1]
    @constraint(model, eWHmx[t=1:tm["P"],w=1:wh["N"]; wh["mx_eff"][w]>0],
        vWHq[t,w] <= bWH_u[t,w])
    # investment in water heater {0,1}
    for i1=1:cfg["IT"]
    	@constraint(model, eWHb[t=tm["IW"][i1]:tm["P"],w=1:wh["N"]; wh["mx_eff"][w]>0],
    		bWH_u[t,w] == sum(bWHty[i2,w] for i2=1:i1))
    end
    # maximum water stored by water heater [0,1]
    @constraint(model, eWHsoc[t=1:tm["P"],w=1:wh["N"]; wh["tank_eff"][w]>0],
        vWHsoc[t,w] <= bWH_u[t,w])
    # water heater balance [0,1]
    @constraint(model, eWHbal[t=1:tm["P"],w=1:wh["N"]; wh["mx_eff"][w]>0],
    	wh["tank_eff"][w]/(tm["TM"][t]*wh["mx_eff"][w])*(vWHsoc[t,w] - vWHsoc[t-1,w]) ==
        wh["eff"][w]*vWHq[t,w] - vWHhw[t,w])
    # fix initial tank level [0,1]
    @constraint(model, eWHi[w=1:wh["N"]; wh["tank_eff"][w]>0],
        vWHsoc[0,w] == cfg["WHsto0"]*bWH_u[1,w])
    # fix final tank level [0,1]
    @constraint(model, eWHf[w=1:wh["N"]; wh["tank_eff"][w]>0],
        vWHsoc[tm["P"],w] == cfg["WHstof"]*bWH_u[tm["P"],w])

end