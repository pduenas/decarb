"""
thermal_load!(model::Model,cfg::Dict,tm::Dict,bdg::Dict,topo::Dict,chp::Dict,hvac::Dict,
    abp::Dict,wh::Dict)

Creates variables, expressions and constraints associated with thermal load balance

inputs:
model   name of core model
cfg     dictionary with configuration input data
tm      dictionary with time series data
bdg     dictionary with building data
topo    dictionary with topology of thermal connections
chp     dictionary with CHP data
hvac    dictionary with HVAC data
abp     dictionary with absorption chiller data
wh      dictionary with water heater data

"""
function thermal_load!(model::Model,cfg::Dict,tm::Dict,bdg::Dict,topo::Dict,chp::Dict,
    hvac::Dict,abp::Dict,wh::Dict)

    # load temperature variables
    vTin = model[:vTin]
    vTup = model[:vTup]
    vTlo = model[:vTlo]

    # disable discomfort temperature if indicated
    if cfg["NSTcost"]==0
        fix.(vTup,0; force=true)
        fix.(vTlo,0; force=true)
    else
        println("   \u2139  Discomfort temperature allowed")
    end

    # minimum indoor temperature [°C]
    @constraint(model, eTmn[t=1:tm["P"]; tm["Ton"][t]==1], vTin[t]+vTlo[t] >= tm["Tmn"][t])
    # maximum indoor temperature [°C]
    @constraint(model, eTmx[t=1:tm["P"]; tm["Ton"][t]==1], vTin[t]-vTup[t] <= tm["Tmx"][t])

    # average thermal power from active equipment [kW]
    @expression(model, vQ_HTAC[t=1:tm["P"]],
        (sum(model[:vCHP_HT][t,c] for c=1:chp["N"] if (!iszero).(topo["chp_bdg"][c,1])) +
         sum(model[:vHVAC_HTAC][t,h] for h=1:hvac["N"] if (hvac["HVmx_eff"][h]>0 || hvac["ACmx_eff"][h]>0)) -
         sum(model[:vABP_AC][t,a] for a=1:abp["N"]))/tm["TM"][t])

    # calculate internal heat gains from occupancy, lighting and electrical equipment
    Q_IHG = tm["Qihg_P"] + tm["Qihg_L"] + tm["Qihg_E"]

    # calculate total radiation on roof and external facades
    Q_R = tm["Q_R"]

    # The data-driven coefficients describe a 15-minute transition. Resample
    # that transition to each model period, assuming weather and thermal power
    # are constant within the period.
    0 <= bdg["Bk1"] <= 1 ||
        error("❗  bdg_ii.pBk1 must be in [0,1] for timestep resampling")
    calibration_hours = 0.25
    period_ratio = tm["TM"] ./ calibration_hours
    Bk1_t = 1 .- (1 - bdg["Bk1"]) .^ period_ratio
    response_scale = iszero(bdg["Bk1"]) ? period_ratio : Bk1_t ./ bdg["Bk1"]
    Bk2_t = bdg["Bk2"] .* response_scale
    Bk3_t = bdg["Bk3"] .* response_scale

    # thermal model based on data-driven parameters:
    #   1- indoor to outdoor temperature difference
    #   2- solar radiation on building envelope
    #   3- heating/cooling balance and internal heat gains
    @constraint(model, eTbal0,      # initial period
        vTin[1] ==
        cfg["Tin0"] + Bk1_t[1]*(tm["Tout"][1]-cfg["Tin0"]) + Bk2_t[1]*Q_R[1] + Bk3_t[1]*(vQ_HTAC[1]+Q_IHG[1]))
    @constraint(model, eTbal[t=2:tm["P"]],
        vTin[t] ==
        vTin[t-1] + Bk1_t[t]*(tm["Tout"][t]-vTin[t-1]) + Bk2_t[t]*Q_R[t] + Bk3_t[t]*(vQ_HTAC[t]+Q_IHG[t]))

    # non-served hot water [0,1]
    @variable(model, 1 >= vNShw[t=1:tm["P"]] >= 0)

    # disable discomfort hot water if indicated
    cfg["NSHWcost"]==0 ? fix.(vNShw,0; force=true) : println("   \u2139  Discomfort hot water allowed")

    # domestic hot water balance [kWh]
    @constraint(model, eHWbal[t=1:tm["P"]],
        sum(model[:vWH_HW][t,w] for w=1:wh["N"] if wh["mx_eff"][w]>0) +
        sum(model[:vCHP_HW][t,c] for c=1:chp["N"] if (!iszero).(topo["chp_bdg"][c,2])) ==
        tm["HWdem"][t]*(1-vNShw[t]))

end
