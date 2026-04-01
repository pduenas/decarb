"""
thermal_load!(model::Model,in::Dict,tm::Dict,bdg::Dict,topo::Dict,chp::Dict,hvac::Dict,
    abs::Dict,wh::Dict)

Creates variables, expressions and constraints associated with thermal load balance

inputs:
model   name of core model
in      dictionary with miscellaneous input data
tm      dictionary with time series data
bdg     dictionary with building data
topo    dictionary with topology of thermal connections
chp     dictionary with CHP data
hvac    dictionary with HVAC data
abs     dictionary with absorption chiller data
wh      dictionary with water heater data

"""
function thermal_load!(model::Model,in::Dict,tm::Dict,bdg::Dict,topo::Dict,chp::Dict,
    hvac::Dict,abs::Dict,wh::Dict)

    # load temperature variables
    vTin = model[:vTin]
    vTup = model[:vTup]
    vTlo = model[:vTlo]

    # disable discomfort temperature if indicated
    if in["NSTcost"]==0
        fix.(vTup,0; force=true)
        fix.(vTlo,0; force=true)
    else
        println("   \u2139  Discomfort temperature allowed")
    end

    # minimum indoor temperature [°C]
    @constraint(model, eTmn[t=1:tm["P"]; tm["Ton"][t]==1], vTin[t]+vTlo[t] >= tm["Tmn"][t])
    # maximum indoor temperature [°C]
    @constraint(model, eTmx[t=1:tm["P"]; tm["Ton"][t]==1], vTin[t]-vTup[t] <= tm["Tmx"][t])

    # Au = calculate_conductivity_ventilation(bdg["Balt"],bdg["Bventy"],bdg["Bvent"],tm["P"],
    #     tm["Tout"],tm["Bppl"])

    # thermal gain from active equipment [kWh]
    @expression(model, vQ_HTAC[t=1:tm["P"]],
        sum(model[:vCHP_HT][t,c] for c=1:chp["N"] if (!iszero).(topo["chp_bdg"][c,1])) + 
        sum(model[:vHVAC_HTAC][t,h] for h=1:hvac["N"] if (hvac["HVmx"][h]>0 || hvac["ACmx"][h]>0)) - 
        sum(model[:vABS_AC][t,a] for a=1:abs["N"]))

    # calculate internal heat gains from occupancy, lighting and electrical equipment
    Q_IHG = tm["Qihg_P"] + tm["Qihg_L"] + tm["Qihg_E"]

    # calculate total radiation on roof and external facades
    Q_R = tm["Q_R"]

    # # calculate capacitance of RC model
    # B_C = bdg["Bfoot"]*bdg["Bslab"]*bdg["Btslab"]*bdg["Bpslab"]*bdg["Bcpslab"]
    # # calculate resistance '1' of RC model
    # B_R1 = 1/bdg["Bfoot"]/bdg["Bslab"]/bdg["Bkslab"]
    # # calculate resistance '2' of RC model
    # B_R2 = 1/sum(bdg["Bwall"].*bdg["Bkwall"],dims=1)[1]
    # B_R2 = 1 ./ (1/B_R2.+Au)    # correct resistance due to ventilation
    # replace!(B_R2,Inf=>1)

    # # temperature balance [°C]
    # if B_C>0.01     # mass in floor slabs
    #     @constraint(model, eTbal0,      # initial period
    #         vQ_HTAC[1]+Q_IHG[1] == -1/B_R2[1]*tm["Tout"][1] +
    #         (B_R1+B_R2[1])/(B_R1*B_R2[1])*(vTin[1]-in["Tin0"]-(tm["TM"][1]/B_C*Q_R[1])))
    #     @constraint(model, eTbal[t=2:tm["P"]],
    #         vQ_HTAC[t]+Q_IHG[t] == 
    #         (1-tm["TM"][t]/(B_R1*B_C))*(vQ_HTAC[t-1]+Q_IHG[t-1]) -
    #         1/B_R2[t]*(tm["Tout"][t]-tm["Tout"][t-1]) +
    #         (B_R1+B_R2[t])/(B_R1*B_R2[t])*(vTin[t]-vTin[t-1]-(tm["TM"][t]/B_C*Q_R[t])) +
    #         tm["TM"][t]/(B_R1*B_R2[t]*B_C)*(vTin[t-1]-tm["Tout"][t-1]))
    # else            # mass in walls
    #     @constraint(model, eTbal0,      # initial period
    #         vQ_HTAC[1]+Q_IHG[1]== (1/B_R1)*(vTin[1]-in["Tin0"]) - 
    #         tm["TM"][1]/(B_R1*B_C)*Q_R[1])
    #     @constraint(model, eTbal[t=2:tm["P"]],
    #         vQ_HTAC[t]+Q_IHG[t] == 
    #         (1-tm["TM"][t]*B_R1*B_R2[t]/(B_R1*B_R2[t]*B_C))*(vQ_HTAC[t-1]+Q_IHG[t-1]) +
    #         tm["TM"][t]/(B_R1*B_R2[t]*B_C)*(vTin[t-1]-tm["Tout"][t-1]) +
    #         (1/B_R1)*(vTin[t]-vTin[t-1]) - tm["TM"][t]/(B_R1*B_C)*Q_R[t])
    # end

    @constraint(model, eTbal0,      # initial period
        vTin[1] == 
        in["Tin0"] + bdg["Bk1"]*(tm["Tout"][1]-in["Tin0"]) + bdg["Bk2"]*Q_R[1] + bdg["Bk3"]*vQ_HTAC[1])
    @constraint(model, eTbal[t=2:tm["P"]],
        vTin[t] == 
        vTin[t-1] + bdg["Bk1"]*(tm["Tout"][t-1]-vTin[t-1]) + bdg["Bk2"]*Q_R[t] + bdg["Bk3"]*vQ_HTAC[t])
    
    # non-served hot water [0,1]
    @variable(model, 1 >= vNShw[t=1:tm["P"]] >= 0)

    # disable discomfort hot water if indicated
    in["NSHWcost"]==0 ? fix.(vNShw,0; force=true) : println("   \u2139  Discomfort hot water allowed")

    # domestic hot water balance [kWh]
    @constraint(model, eHWbal[t=1:tm["P"]], 
        sum(model[:vWH_HW][t,w] for w=1:wh["N"] if wh["mx"][w]>0) + 
        sum(model[:vCHP_HW][t,c] for c=1:chp["N"] if (!iszero).(topo["chp_bdg"][c,2])) == 
        tm["HWdem"][t]*(1-vNShw[t]))

end


# function calculate_conductivity_ventilation(alt,venty,vent,nP,Tout,ppl)

#     # air density variation with temperature [kg/m3]
#     Ap_T = [1.4224,1.3943,1.3673,1.3413,1.3163,1.2922,1.269,1.2466,1.225,1.2041,1.1839,1.1644,1.1455]
#     A_T  = [   -25,   -20,   -15,   -10,    -5,     0,    5,    10,   15,    20,    25,    30,    35]

#     # air density correction by altitude [kg/m3]
#     Ap_T = Ap_T.*exp(-alt/10400)

#     # air volumetric heat capacity [kWh/m3-°C]
#     Avhc_T = Ap_T.*0.0002793

#     # conductivity due to ventilation per person [kW/°C/person]
#     Au_T = Avhc_T.*vent
#     Au = Array{Float64}(undef,nP)       # initialize conductivity vector
#     for t=1:nP
#         lb = diff([Tout[t].>=A_T;0],dims=1).!=0     # lower bound for interpolation
#         ub = diff([0;Tout[t].<=A_T],dims=1).!=0     # upper bound for interpolation
#         # identify temperature interval for interpolation
#         xin = (1:length(A_T)).*(lb.|ub)
#         xin_T = Array{Float64}(undef,sum(xin.!=0))
#         xin_T = xin[xin.!=0]
#         # if coincident upper and lower bound, no need for interpolation
#         length(xin_T)==1 ? Au[t]=Au_T[xin_T[1]] : 
#         # else, interpolate within interval of temperatures
#             Au[t]=(Au_T[xin_T[2]]-Au_T[xin_T[1]])/(A_T[xin_T[2]]-A_T[xin_T[1]])*(Tout[t]-A_T[xin_T[1]])+Au_T[xin_T[1]]
#     end

#     # conductivity due to ventilation [kW/°C]
#     if venty==0         # forced ventilation varies with level of occupancy
#         return Au.*ppl
#     elseif venty==1     # forced ventilation fixed at maximum occupancy
#         return Au.*maximum(ppl)
#     end

# end