"""
read_fuel(model::Model,tm::Dict,n_chp::Int64,n_abp::Int64,n_wh::Int64,n_topo::Int64)

Reads time series outputs related to fuel consumptions and load them into dataframe

inputs:
model   optimization model object
tm      dictionary with time series data
n_chp   number of CHP types
n_abp   number of absorption chiller types
n_wh    number of water heater types
n_topo  number of thermal links

returns dataframes of outputs
"""

function read_fuel(model::Model,tm::Dict,n_chp::Int64,n_abp::Int64,n_wh::Int64,n_topo::Int64)

    # outputs from CHP units
    if n_chp > 0
        CHP_G = round.(value.(model[:vCHP_G])./tm["TM"], digits=2)
        CHP_L = round.(value.(model[:vCHP_L])./tm["TM"], digits=2)
    else
        CHP_G = zeros(Float64, tm["P"])
        CHP_L = zeros(Float64, tm["P"])
    end

    # outputs from absorption chiller
    if n_abp > 0
        ABS_G = round.(value.(model[:vABS_G])./tm["TM"], digits=2)
        ABS_L = round.(value.(model[:vABS_L])./tm["TM"], digits=2)
    else
        ABS_G = zeros(Float64, tm["P"])
        ABS_L = zeros(Float64, tm["P"])
    end

    # outputs from water heaters
    if n_wh > 0
        WH_G = round.(value.(model[:vWH_G])./tm["TM"], digits=2)
        WH_L = round.(value.(model[:vWH_L])./tm["TM"], digits=2)
    else
        WH_G = zeros(Float64, tm["P"])
        WH_L = zeros(Float64, tm["P"])
    end

    # outputs from thermal links
    if n_topo > 0
        TH_G = round.(value.(model[:vTH_G])./tm["TM"], digits=2)
        TH_L = round.(value.(model[:vTH_L])./tm["TM"], digits=2)
    else
        TH_G = zeros(Float64, tm["P"])
        TH_L = zeros(Float64, tm["P"])
    end

    df = DataFrame(Gchp=CHP_G,Gabs=ABS_G,Gwh=WH_G,Gth=TH_G,Lchp=CHP_L,Labs=ABS_L,Lwh=WH_L,Lth=TH_L)
    
    return df

end
