"""
electric_vehicles!(model::Model,in::Dict,tm::Dict,bdg::Dict,sp::Dict,ev::Dict)

Creates variables, expressions and constraints associated to EV modules

inputs:
model   name of core model
in      dictionary with miscellaneous input data
tm      dictionary with time series data
ev      dictionary with EV data

"""
function electric_vehicles!(model::Model,in::Dict,tm::Dict,ev::Dict)

    # unitary energy consumption for EV charging [0,1]
    @variable(model, 1 >= vEVup[t=1:tm["P"],e=1:ev["N"]] >= 0)
    # unitary energy production from EV discharging [0,1]
    @variable(model, 1 >= vEVdn[t=1:tm["P"],e=1:ev["N"]] >= 0)
    # charge/discharge mode of electric vehicle {0,1}
    @variable(model, bEV[t=1:tm["P"],e=1:ev["N"]], Bin)
    # unitary state-of-charge of electric vehicle [0,1]    
    @variable(model, 1 >= vEVsoc[t=1:tm["P"],e=1:ev["N"]] >= 0)    
    # excursion below minimum state-of-charge of electric vehicle [0,1]
    @variable(model, in["EVmnsoc"] >= vEVlo[t=1:tm["P"],e=1:ev["N"]] >= 0)    

    # correct maximum capacity for non-existing or disabled CHP units
    for e = 1:ev["N"]
        if all(ev["time"][:,e].==0)
            ev["mx"][e] = 0
        end
    end

    # electricity charged in electric vehicle [kWh]
    @expression(model, vEV_UP[t=1:tm["P"],e=1:ev["N"]], ev["mx"][e]*vEVup[t,e])
    # electricity discharged from electric vehicle [kWh]
    @expression(model, vEV_DN[t=1:tm["P"],e=1:ev["N"]], ev["mx"][e]*vEVdn[t,e])

    # maximum charging of electric vehicle [0,1]
    @constraint(model, eEVup[t=1:tm["P"],e=1:ev["N"]; ev["mx"][e]>0],
        vEVup[t,e] <= tm["TM"][t]*ev["up"][e]/ev["mx"][e]*bEV[t,e])
    # maximum discharging from electric vehicle [0,1]
    @constraint(model, eEVdn[t=1:tm["P"],e=1:ev["N"]; ev["mx"][e]>0],
        vEVdn[t,e] <= tm["TM"][t]*ev["dn"][e]/ev["mx"][e]*(1-bEV[t,e]))
    # electricity stored balance in electric vehicle [0,1]
    @constraint(model, eEVbal[t=1:tm["P"],e=1:ev["N"]; ev["mx"][e]>0],
        vEVsoc[t,e] - (t==1 ? in["EVsoc0"] : vEVsoc[t-1,e]) == 
        vEVup[t,e]*ev["effu"][e]-vEVdn[t,e]/ev["effd"][e]-ev["kwh"][t,e]/ev["mx"][e])
    # charging allowed when electric vehicle plugged in
    @constraint(model, eEVupok[t=1:tm["P"],e=1:ev["N"]; ev["time"][t,e]==0],
        vEVup[t,e] == 0)
    # discharging allowed when electric vehicle plugged in and enabled  
    @constraint(model, eEVdnok[t=1:tm["P"],e=1:ev["N"]; ev["time"][t,e]==0 || in["EVv2g"]==0],
        vEVdn[t,e] == 0)
    # allowed excursion of minimum state-of-charge for electric vehicle
    @constraint(model, eEVlo[t=1:tm["P"],e=1:ev["N"]; ev["mx"][e]>0],
        vEVsoc[t,e] >= in["EVmnsoc"]-vEVlo[t,e])

end