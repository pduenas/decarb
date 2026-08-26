"""
load_sp(path::AbstractString,in::Dict)

Loads sp.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
b_inv   boolean flag to reveal investment option

returns sp-type inputs in dictionary object
"""

function load_sp(path::AbstractString,in::Dict)

    # declare dictionary object to store parameters
    sp = Dict()

    path2file = joinpath(path,"sp.csv")

    # load file into dataframe with predefined types
    df_sp = CSV.File(path2file;delim=',',types=[String,UInt8,String,String,UInt8,String,
                    String,UInt8,String,String,UInt8,String,String,UInt8,String,String,
                    UInt8,String,String,UInt8,String,String,UInt8,String]) |> DataFrame

    sp["CHP0"] = collect(skipmissing(df_sp.pCHP0))      # CHP equipment
    sp["CHPz0"] = collect(skipmissing(df_sp.pCHPz0))    # CHP existing units [0,...,n]
    sp["CHPyn"] = collect(skipmissing(df_sp.pCHPyn))    # enable investment in CHP
    sp["ABP0"] = collect(skipmissing(df_sp.pABP0))      # absorption chiller equipment
    sp["ABPz0"] = collect(skipmissing(df_sp.pABPz0))    # absorption chiller existing units [0,...,n]
    sp["ABPyn"] = collect(skipmissing(df_sp.pABPyn))    # enable investment in absorption chiller
    sp["HVAC0"] = collect(skipmissing(df_sp.pHVAC0))    # HVAC equipment
    sp["HVACz0"] = collect(skipmissing(df_sp.pHVACz0))  # HVAC existing units [0,...,n]
    sp["HVACyn"] = collect(skipmissing(df_sp.pHVACyn))  # enable investment in HVAC
    sp["WH0"] = collect(skipmissing(df_sp.pWH0))        # water heater equipment
    sp["WHz0"] = collect(skipmissing(df_sp.pWHz0))      # water heater existing units [0,...,n]
    sp["WHyn"] = collect(skipmissing(df_sp.pWHyn))      # enable investment in water heater
    sp["PV0"] = collect(skipmissing(df_sp.pPV0))        # PV equipment
    sp["PVz0"] = collect(skipmissing(df_sp.pPVz0))      # PV existing panels [0,...,n]
    sp["PVyn"] = collect(skipmissing(df_sp.pPVyn))      # enable investment in PV
    sp["WIND0"] = collect(skipmissing(df_sp.pWIND0))    # PV equipment
    sp["WINDz0"] = collect(skipmissing(df_sp.pWINDz0))  # PV existing panels [0,...,n]
    sp["WINDyn"] = collect(skipmissing(df_sp.pWINDyn))  # enable investment in PV
    sp["BESS0"] = collect(skipmissing(df_sp.pBESS0))    # BESS equipment
    sp["BESSz0"] = collect(skipmissing(df_sp.pBESSz0))  # BESS existing modules [0,...,n]
    sp["BESSyn"] = collect(skipmissing(df_sp.pBESSyn))  # enable investment in BESS
    sp["EV0"] = collect(skipmissing(df_sp.pEV0))        # EV types
    sp["EVz0"] = collect(skipmissing(df_sp.pEVz0))      # EV existing types [0,...,n]

    # For each equipment group, filter rows if all three corresponding columns are zero
    maskCHP = .!((sp["CHP0"].=="0") .& (sp["CHPz0"].==0) .& (sp["CHPyn"].=="0"))
    sp["CHP0"] = sp["CHP0"][maskCHP]
    sp["CHPz0"] = sp["CHPz0"][maskCHP]
    sp["CHPyn"] = sp["CHPyn"][maskCHP]

    maskABP = .!((sp["ABP0"].=="0") .& (sp["ABPz0"].==0) .& (sp["ABPyn"].=="0"))
    sp["ABP0"] = sp["ABP0"][maskABP]
    sp["ABPz0"] = sp["ABPz0"][maskABP]
    sp["ABPyn"] = sp["ABPyn"][maskABP]

    maskHVAC = .!((sp["HVAC0"].=="0") .& (sp["HVACz0"].==0) .& (sp["HVACyn"].=="0"))
    sp["HVAC0"] = sp["HVAC0"][maskHVAC]
    sp["HVACz0"] = sp["HVACz0"][maskHVAC]
    sp["HVACyn"] = sp["HVACyn"][maskHVAC]

    maskWH = .!((sp["WH0"].=="0") .& (sp["WHz0"].==0) .& (sp["WHyn"].=="0"))
    sp["WH0"] = sp["WH0"][maskWH]
    sp["WHz0"] = sp["WHz0"][maskWH]
    sp["WHyn"] = sp["WHyn"][maskWH]

    maskPV = .!((sp["PV0"].=="0") .& (sp["PVz0"].==0) .& (sp["PVyn"].=="0"))
    sp["PV0"] = sp["PV0"][maskPV]
    sp["PVz0"] = sp["PVz0"][maskPV]
    sp["PVyn"] = sp["PVyn"][maskPV]

    maskWIND = .!((sp["WIND0"].=="0") .& (sp["WINDz0"].== 0) .& (sp["WINDyn"].=="0"))
    sp["WIND0"] = sp["WIND0"][maskWIND]
    sp["WINDz0"] = sp["WINDz0"][maskWIND]
    sp["WINDyn"] = sp["WINDyn"][maskWIND]

    maskBESS = .!((sp["BESS0"].=="0") .& (sp["BESSz0"].==0) .& (sp["BESSyn"].=="0"))
    sp["BESS0"] = sp["BESS0"][maskBESS]
    sp["BESSz0"] = sp["BESSz0"][maskBESS]
    sp["BESSyn"] = sp["BESSyn"][maskBESS]

    maskEV = .!((sp["EV0"].=="0") .& (sp["EVz0"].==0))
    sp["EV0"] = sp["EV0"][maskEV]
    sp["EVz0"] = sp["EVz0"][maskEV]

    # disable potential for investment when investment windows do not exist
    if in["b_inv"]==false
        sp["CHPyn"] .= "NO"
        sp["ABPyn"] .= "NO"
        sp["HVACyn"] .= "NO"
        sp["WHyn"] .= "NO"
        sp["PVyn"] .= "NO"
        sp["WINDyn"] .= "NO"
        sp["BESSyn"] .= "NO"
    end

    return sp

end
