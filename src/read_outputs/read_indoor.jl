"""
read_indoor(model::Model,tm::Dict,n_chp::Int64,n_abp::Int64,n_hvac::Int64,n_wh::Int64,
    HVmx::Matrix{Float64},ACmx::Matrix{Float64},WHtank::Vector{Float64},CHPbdg::Matrix{Float64})

Reads time series outputs related to indoor demands for temperature and hot water and
    load them into dataframe

inputs:
model   optimization model object
tm      dictionary with time series data
n_chp   number of CHP types
n_abp   number of absorption chiller types
n_hvac  number of HVAC types
n_wh    number of water heater types
WHtank  capacity of water heater
CHPbdg  CHP unit connected to building

returns dataframes of outputs
"""

function read_indoor(model::Model,tm::Dict,n_chp::Int64,n_abp::Int64,n_hvac::Int64,n_wh::Int64,
    WHtank::Vector{Float64},CHPbdg::Matrix{Float64})

    # indoor temperature
    Tin = round.(value.(model[:vTin]), digits=2)

    # outputs from HVAC units
    if n_hvac>0
        HVAC_HTAC = round.(sum(value.(model[:vHVAC_HTAC][:,h]) for h in 1:n_hvac),digits=2)
        HVAC_HT = max.(HVAC_HTAC, 0)
        HVAC_AC = max.(-HVAC_HTAC, 0)
    else
        HVAC_HT = zeros(Float64, tm["P"])
        HVAC_AC = zeros(Float64, tm["P"])
    end
    
    # outputs from CHP units
    if n_chp>0
        CHP_HT = zeros(tm["P"])
        CHP_HW = zeros(tm["P"])
        for t=1:tm["P"]
			for c=1:n_chp
				if !iszero(CHPbdg[c,1])
                    CHP_HT[t] += round(value(model[:vCHP_HT][t,c]),digits=2)
                end
				if !iszero(CHPbdg[c,2])
                    CHP_HW[t] += round(value(model[:vCHP_HW][t,c]),digits=2)
				end
			end
		end
    else
        CHP_HT = zeros(Float64, tm["P"])
        CHP_HW = zeros(Float64, tm["P"])
    end

    # outputs from absorption chillers
    if n_abp>0
        ABP_AC = round.(sum(value.(model[:vABP_AC][:,a]) for a=1:n_abp), digits=2)
    else
        ABP_AC = zeros(Float64, tm["P"])
    end

    # solar heat gains
    Qihg_P = round.(tm["Qihg_P"], digits=2)
    Qihg_L = round.(tm["Qihg_L"], digits=2)
    Qihg_E = round.(tm["Qihg_E"], digits=2)
    Qshg = round.(tm["Q_R"], digits=2)

    # hot water
    HWdem = round.(tm["HWdem"], digits=2)
    NSHW = round.(tm["HWdem"].*value.(model[:vNShw]), digits=2)
    WHhw = round.(sum(value.(model[:vWH_HW][:,w]) for w=1:n_wh), digits=2)

    # outputs from water heaters
    if n_wh>0
        WHsoc = zeros(Float64, tm["P"])
        for p=1:tm["P"]
            WHsoc[p] = round.(sum(WHtank[w]*value.(model[:vWHsoc][p,w]) for w=1:n_wh), digits=2)
        end
    else
        WHsoc = zeros(Float64, tm["P"])
    end

    df = DataFrame(Temp=Tin,HVACht=HVAC_HT,HVACac=HVAC_AC,CHPht=CHP_HT,ABPac=ABP_AC,
        IHGp=Qihg_P,IHGl=Qihg_L,IHGe=Qihg_E,SHG=Qshg,HWdem=HWdem,HWns=NSHW,WHsoc=WHsoc,
        WHhw=WHhw,HWchp=CHP_HW)

    return df

end
