"""
load_hvac(path::AbstractString,tm::Dict,sp::Dict,bdg::Dict)

Loads hvac.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
tm      dictionary with time series data
sp      dictionary with equipment selection
bdg     dictionary with building data

returns hvac-type inputs in dictionary object
"""

function load_hvac(path::AbstractString,tm::Dict,sp::Dict,bdg::Dict)

    # declare dictionary object to store parameters
    hvac = Dict()

    path2file = joinpath(path,"hvac.csv")
    
    # load file into dataframe with predefined types
    df_hvac = CSV.File(path2file;delim=',',types=[String,Float64,Float64,Float64,
                    Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,
                    Float64,Float64]) |> DataFrame

    # delete non-existing rows
    filter!(row -> !ismissing(row.ty), df_hvac)
    # delete all zero rows
    filter!(row -> !all(x -> x==0, row), df_hvac)
    # delete zero-capacity elemenets
    filter!(row -> !(iszero(row.pHVmx) && iszero(row.pACmx)), df_hvac)
    # delete repeated rows
    unique!(df_hvac)

    df_hvac = extend_catalog(df_hvac,sp["HVAC0"],sp["HVACz0"],sp["HVACyn"],bdg["Bhvac"])

    hvac["ty"] = df_hvac.ty             # name of HVAC unit
    hvac["HVmx"] = df_hvac.pHVmx        # maximum heating capacity [kW]
    hvac["ACmx"] = df_hvac.pACmx        # maximum cooling capacity [kW]
    hvac["HVeff"] = df_hvac.pHVeff      # efficiency --heating--
    hvac["ACeff"] = df_hvac.pACeff      # efficiency --cooling--
    hvac["temp"] = df_hvac.temp         # design temperature [°C]
    hvac["HVmx_"] = df_hvac.pHVmx_      # heating capacity loss [kW/°C]
    hvac["ACmx_"] = df_hvac.pACmx_      # cooling capacity loss [kW/°C]
    hvac["HVeff_"] = df_hvac.pHVeff_    # efficiency loss --heating-- [/°C]
    hvac["ACeff_"] = df_hvac.pACeff_    # efficiency loss --cooling-- [/°C]
    hvac["inv"] = df_hvac.inv           # capital cost [$]
    hvac["fom"] = df_hvac.fom           # fixed O&M cost [$/year]
    hvac["vom"] = df_hvac.vom           # variable O&M cost [$/kWh]
    hvac["life"] = df_hvac.life         # lifetime [years]

    hvac["N"] = size(hvac["ty"],1)      # number of HVAC units

    # correct derating coefficients to avoid infeasibility
    MIN_CAP = 0.02
    hvac["HVmx_"] = min.(hvac["HVmx_"],MIN_CAP)
    hvac["ACmx_"] = min.(hvac["ACmx_"],MIN_CAP)

    hvac["HVeff_k"] = zeros(tm["P"], hvac["N"])
    hvac["ACeff_k"] = zeros(tm["P"], hvac["N"])
    
    for h=1:hvac["N"]
        # correct heating efficiency with temperature
        hvac["HVeff_k"][:,h] = correct_heating_efficiency(hvac["HVeff"][h],hvac["temp"][h],
            hvac["HVeff_"][h],tm["Tout"])
        # correct cooling efficiency with temperature
        hvac["ACeff_k"][:,h] = correct_cooling_efficiency(hvac["ACeff"][h],hvac["temp"][h],
            hvac["ACeff_"][h],tm["Tout"])
    end

    return hvac

end

function correct_heating_efficiency(Qeff,Thvac,Coeff,Tout)

    return max.(Qeff .- Coeff.*(max.(Thvac.-Tout,0)), 1)

end

function correct_cooling_efficiency(Qeff,Thvac,Coeff,Tout)

    return max.(Qeff .- Coeff.*(max.(Tout.-Thvac,0)), 1)

end
