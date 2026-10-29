"""
load_sp(path::AbstractString,cfg::Dict)

Loads sp.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
cfg     dictionary with configuration data

returns sp-type inputs in dictionary object
"""

function load_sp(path::AbstractString,cfg::Dict)

    # declare dictionary object to store parameters
    sp = Dict()

    path2file = joinpath(path,"sp.csv")

    # load file into dataframe with predefined types
    df_sp = CSV.File(path2file;delim=',',types=Dict{String,Type}(
                    ["p$(k)$(s)"=>t for k in ("CHP","ABP","HVAC","WH","PV","WIND","BESS","EV")
                     for (s,t) in (("0",String),("z0",Int),("yn",String))])) |> DataFrame

    # columns are ragged, but each row of a group must be either blank or complete
    for prefix in ("CHP","ABP","HVAC","WH","PV","WIND","BESS","EV")
        group = ["p$(prefix)0","p$(prefix)z0"]
        prefix == "EV" || push!(group,"p$(prefix)yn")
        require_columns(df_sp,"sp.csv",group)
        for row in 1:nrow(df_sp)
            blanks = count(c -> ismissing(df_sp[row,c]),group)
            0 < blanks < length(group) && throw(ArgumentError(
                "sp.csv row $row has a missing value among columns $(join(group,", "))"))
        end
    end

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
    sp["WIND0"] = collect(skipmissing(df_sp.pWIND0))    # wind equipment
    sp["WINDz0"] = collect(skipmissing(df_sp.pWINDz0))  # existing wind turbines [0,...,n]
    sp["WINDyn"] = collect(skipmissing(df_sp.pWINDyn))  # enable investment in wind
    sp["BESS0"] = collect(skipmissing(df_sp.pBESS0))    # BESS equipment
    sp["BESSz0"] = collect(skipmissing(df_sp.pBESSz0))  # BESS existing modules [0,...,n]
    sp["BESSyn"] = collect(skipmissing(df_sp.pBESSyn))  # enable investment in BESS
    sp["EV0"] = collect(skipmissing(df_sp.pEV0))        # EV types
    sp["EVz0"] = collect(skipmissing(df_sp.pEVz0))      # EV existing types [0,...,n]

    # For each equipment group, filter rows if all three corresponding columns are zero
    for prefix in ("CHP","ABP","HVAC","WH","PV","WIND","BESS")
        mask = .!((sp["$(prefix)0"].=="0") .& (sp["$(prefix)z0"].==0) .& (sp["$(prefix)yn"].=="0"))
        sp["$(prefix)0"] = sp["$(prefix)0"][mask]
        sp["$(prefix)z0"] = sp["$(prefix)z0"][mask]
        sp["$(prefix)yn"] = sp["$(prefix)yn"][mask]
        # unmodified yn values and their sp.csv data rows, kept for validation
        sp["$(prefix)yn_raw"] = copy(sp["$(prefix)yn"])
        sp["$(prefix)rows"] = collect(1:length(mask))[mask]
    end

    maskEV = .!((sp["EV0"].=="0") .& (sp["EVz0"].==0))
    sp["EV0"] = sp["EV0"][maskEV]
    sp["EVz0"] = sp["EVz0"][maskEV]

    # disable potential for investment when investment windows do not exist
    if cfg["b_inv"]==false
        for prefix in ("CHP","ABP","HVAC","WH","PV","WIND","BESS")
            sp["$(prefix)yn"] .= "NO"
        end
    end

    return sp

end
