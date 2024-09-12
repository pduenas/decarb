"""
load_in(path::AbstractString)

Loads in.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory

returns in-type inputs in dictionary object
"""

function load_in(path::AbstractString)

    # declare dictionary object to store parameters
    in = Dict()

    path2file = joinpath(path,"in.csv")
    
    # load file into dataframe with predefined types
    df_in = CSV.File(path2file;delim=',',types=[DateTime,Bool,Float64,Float64,Float64,
                    Float64,Float64,Float64,Float64,UInt16,Float64,Float64,Float64,
                    Float64,Float64,UInt8,Float64,Float64,Float64,UInt8,Float64],
                    transpose=true) |> DataFrame

    in["P0"] = df_in.pP0[1]             # initial date - datetime
    in["Tmode"] = df_in.pTmode[1]       # temperature control mode {0,1}
    in["BESSsoc0"] = df_in.pBESSsoc0[1] # initial state-of-charge of BESS [%]
    in["BESSsocf"] = df_in.pBESSsocf[1] # final state-of-charge of BESS [%]
    in["WHsto0"] = df_in.pWHsto0[1]     # initial state-of-tank WH [%]
    in["WHstof"] = df_in.pWHstof[1]     # final state-of-tank WH [%]
    in["Tin0"] = df_in.pTin0[1]         # initial indoor temperature [°C]
    in["QmxBuy"] = df_in.pQmxBuy[1]     # maximum purchase of electricity [kW]
    in["QmxSell"] = df_in.pQmxSell[1]   # maximum sale of electricity [kW]
    in["ST"] = df_in.pST[1]             # survival time [h]
    in["Gco2"] = df_in.pGco2[1]		    # CO2 emissions rate of gaseous fuel [kg/kWh]
    in["Lco2"] = df_in.pLco2[1]    	    # CO2 emissions rate of liquid fuel [kg/kWh]
    in["CO2mx"] = df_in.pCO2mx[1]    	# CO2 emission limit [ton]
    in["CO2cost"] = df_in.pCO2cost[1]	# CO2 price offset [$/ton]
    in["CO2ton"] = df_in.pCO2ton[1]		# CO2 amount offset [ton]
    in["QmxTM"] = df_in.pQmxTM[1]	    # peak capacity charge periods {0,12}
    in["NSEcost"] = df_in.pNSEcost[1]   # cost of non-served electricity [$/kWh]
    in["NSTcost"] = df_in.pNSTcost[1]   # cost of temperature discomfort [$/°C-h]
    in["NSHWcost"] = df_in.pNSHWcost[1] # cost of non-served hot water [$/kWh]
    in["IT"] = df_in.pIT[1]		        # investment windows [0,...,n]
    in["IR"] = df_in.pIR[1]             # annual interest rate [%]

    # enable or disable investments in equipment
    in["IT"]>0 ? in["b_inv"]=true : in["b_inv"]=false
    # enable or disable temperature control
    in["b_temp"] = in["Tmode"]

    # interest rate corrected by frequency of payments during year
    in["IR"] = in["IR"]*1

    return in

end