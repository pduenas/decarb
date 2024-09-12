"""
read_electric(model::Model,tm::Dict,n_chp::Int64,n_hvac::Int64,n_wh::Int64,
    n_pv::Int64,n_bess::Int64,n_wind::Int64,BESSmx::Vector{Float64})

Reads time series outputs related to electric demands and generations and
    load them into dataframe

inputs:
model   optimization model object
tm      dictionary with time series data
n_chp   number of CHP types
n_hvac  number of HVAC types
n_wh    number of water heater types
n_pv    number of PV panel types
n_bess  number of BESS module types
n_wind  number of wind turbine types
BESSmx  maximum battery capacity

returns dataframes of outputs
"""

function read_electric(model::Model,tm::Dict,n_chp::Int64,n_hvac::Int64,n_wh::Int64,
    n_pv::Int64,n_bess::Int64,n_wind::Int64,BESSmx::Vector{Float64})

    Qdem = tm["Qlight"] + tm["Qequip"]
    Qsell = round.(value.(model[:vQsell])./tm["TM"], digits=2)
    Qbuy = round.(value.(model[:vQbuy])./tm["TM"], digits=2)
    NSE_Q = round.(value.(model[:vNSE_Q])./tm["TM"], digits=2)

    # outputs from CHP units
    if n_chp>0
        CHP_Q = round.(sum(value.(model[:vCHP_Q][:,c]) for c=1:n_chp)./tm["TM"], digits=2)
    else
        oCHP_Q = zeros(Float64, tm["P"])
    end

    # outputs from HVAC units
    if n_hvac>0
        HVAC_Qht = round.(sum(value.(model[:vHVAC_HT][:,h]) for h=1:n_hvac)./tm["TM"], digits=2)
        HVAC_Qac = round.(sum(value.(model[:vHVAC_AC][:,h]) for h=1:n_hvac)./tm["TM"], digits=2)
    else
        HVAC_Qht = zeros(Float64, tm["P"])
        HVAC_Qac = zeros(Float64, tm["P"])
    end

    # outputs from water heaters
    if n_wh > 0
        WH_Q = round.(sum(value.(model[:vWH_Q][:,w]) for w=1:n_wh)./tm["TM"], digits=2)
    else
        WH_Q=zeros(Float64, tm["P"])
    end

    # outputs from solar panels
    if n_pv > 0
        PV_Q = round.(sum(value.(model[:vPV_Q][:,p]) for p=1:n_pv)./tm["TM"], digits=2)
    else
        PV_Q = zeros(Float64, tm["P"])
    end

    # outputs from battery modules
    if n_bess > 0
        BESS_UP = round.(sum(value.(model[:vBESS_UP][:,s]) for s=1:n_bess)./tm["TM"], digits=2)
        BESS_DN = round.(sum(value.(model[:vBESS_DN][:,s]) for s=1:n_bess)./tm["TM"], digits=2)
        BESSsoc = zeros(Float64, tm["P"])										 
        for p=1:tm["P"]
            BESSsoc[p] = round.(sum(BESSmx[s]*value.(model[:vBESSsoc][p,s]) for s=1:n_bess), digits=2)
        end
    else
        BESS_UP = zeros(Float64, tm["P"])
        BESS_DN = zeros(Float64, tm["P"])
        BESSsoc = zeros(Float64, tm["P"])
    end

    # outputs from wind turbines
    if n_wind > 0
        WIND_Q = round.(sum(value.(model[:vWIND_Q][:,p]) for p=1:n_wind)./tm["TM"], digits=2)
    else
        WIND_Q = zeros(Float64, tm["P"])
    end

    df = DataFrame(dem=Qdem,sell=Qsell,hvac_HT=HVAC_Qht,hvac_AC=HVAC_Qac,Qwh=WH_Q,bess_UP=BESS_UP,
        buy=Qbuy,pv=PV_Q,wind=WIND_Q,chp=CHP_Q,bess_DN=BESS_DN,nse=NSE_Q,bess_SOC=BESSsoc)

    balance = Qdem+Qsell+HVAC_Qht+HVAC_Qac+WH_Q+BESS_UP-Qbuy-PV_Q-WIND_Q-CHP_Q-BESS_DN-NSE_Q
    
    return df,balance

end
