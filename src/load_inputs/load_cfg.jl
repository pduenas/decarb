"""
load_cfg(path::AbstractString)

Loads cfg.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory

returns configuration inputs in dictionary object
"""

function load_cfg(path::AbstractString)

    # declare dictionary object to store parameters
    cfg = Dict()

    path2file = joinpath(path,"cfg.csv")
    
    # load file into dataframe with predefined types
    df_cfg = CSV.File(path2file;delim=',',types=[DateTime,Bool,Float64,Float64,Float64,
                      Float64,Float64,Float64,Float64,Float64,Float64,UInt16,Float64,
                      Float64,Float64,Float64,Float64,UInt8,Float64,Float64,Float64,
                      Float64,Int8,Float64,Bool,UInt8,Float64],transpose=true) |> DataFrame

    cfg["P0"] = df_cfg.pP0[1]               # initial date - datetime
    cfg["Tmode"] = df_cfg.pTmode[1]         # temperature control mode {0,1}
    cfg["BESSsoc0"] = df_cfg.pBESSsoc0[1]   # initial state-of-charge of BESS
    cfg["BESSsocf"] = df_cfg.pBESSsocf[1]   # final state-of-charge of BESS
    cfg["EVsoc0"] = df_cfg.pEVsoc0[1]       # initial state-of-charge of EV
    cfg["EVmnsoc"] = df_cfg.pEVmnsoc[1]     # minimum state-of-charge of EV
    cfg["WHsto0"] = df_cfg.pWHsto0[1]       # initial state-of-tank WH
    cfg["WHstof"] = df_cfg.pWHstof[1]       # final state-of-tank WH
    cfg["Tin0"] = df_cfg.pTin0[1]           # initial indoor temperature [°C]
    cfg["QmxBuy"] = df_cfg.pQmxBuy[1]       # maximum purchase of electricity [kW]
    cfg["QmxSell"] = df_cfg.pQmxSell[1]     # maximum sale of electricity [kW]
    cfg["ST"] = df_cfg.pST[1]               # survival time [h]
    cfg["Gco2"] = df_cfg.pGco2[1]		    # CO2 emissions rate of gaseous fuel [kg/kWh]
    cfg["Lco2"] = df_cfg.pLco2[1]    	    # CO2 emissions rate of liquid fuel [kg/kWh]
    cfg["CO2mx"] = df_cfg.pCO2mx[1]    	    # CO2 emission limit [ton]
    cfg["CO2cost"] = df_cfg.pCO2cost[1]	    # CO2 price offset [$/ton]
    cfg["CO2ton"] = df_cfg.pCO2ton[1]		# CO2 amount offset [ton]
    cfg["QmxTM"] = df_cfg.pQmxTM[1]	        # peak capacity charge periods {0,12}
    cfg["NSEcost"] = df_cfg.pNSEcost[1]     # cost of non-served electricity [$/kWh]
    cfg["NSTcost"] = df_cfg.pNSTcost[1]     # cost of temperature discomfort [$/°C-h]
    cfg["NSHWcost"] = df_cfg.pNSHWcost[1]   # cost of non-served hot water [$/kWh]
    cfg["NSEVcost"] = df_cfg.pNSEVcost[1]   # cost of non-served EV state-of-charge [$/p.u.]
    cfg["EVdriver"] = df_cfg.pEVdriver[1]	# type of driver {-1,0,1}
    cfg["EVpen"] = df_cfg.pEVpen[1]	        # penalty on type of driver [$/p.u.]
    cfg["EVv2g"] = df_cfg.pEVv2g[1]	        # vehicle to grid allowed {0,1}
    cfg["IT"] = df_cfg.pIT[1]		        # investment windows [0,...,n]
    cfg["IR"] = df_cfg.pIR[1]               # annual interest rate [p.u.]

    # enable investments
    cfg["b_inv"] = true
    # when no investment windows are defined
    if cfg["IT"] == 0
        # disable investments
        cfg["b_inv"] = false
        # allow initial free installation
        cfg["IT"] = 1
    end

    # enable or disable temperature control
    cfg["b_temp"] = cfg["Tmode"]

    return cfg

end