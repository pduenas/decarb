"""
load_tm(path::AbstractString,cfg::Dict,bdg::Dict)

Loads tm.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
cfg     dictionary with configuration data
bdg     dictionary with building data

returns tm-type inputs in dictionary object
"""

function load_tm(path::AbstractString,cfg::Dict,bdg::Dict)

    # declare dictionary object to store parameters
    tm = Dict()

    path2file = joinpath(path,"tm.csv")

    # load file into dataframe with predefined types
    df_tm = CSV.File(path2file;delim=',',types=[DateTime,Float64,Float64,UInt8,Float64,
                    Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,
                    UInt8,Float64,Float64,Float64,Float64,Float64,Float64,UInt8,
                    Float64]) |> DataFrame

    # delete non-existing rows
    filter!(row -> !ismissing(row.pP), df_tm)
    # delete all zero rows
    filter!(row -> !all(iszero, collect(row)[2:end]), df_tm)
    
    tm["Date"] = df_tm.pP               # date - datetime
    tm["QcostBuy"] = df_tm.pQcostBuy    # price of electricity purchase [$/kWh]
    tm["QcostSell"] = df_tm.pQcostSell  # price of electricity sale [$/kWh]
    tm["Qmx"] = df_tm.pQmx		        # peak capacity charge period type {0,12}
    tm["QmxCost"] = df_tm.pQmxCost		# peak capacity charge [$/kW]
    tm["Qco2"] = df_tm.pQco2            # CO2 emissions rate of power system [kg/kWh]
    tm["Gcost"] = df_tm.pGcost          # price of gaseous fuel [$/kWh]
    tm["Lcost"] = df_tm.pLcost          # price of liquid fuel [$/kWh]
    tm["Qlight"] = df_tm.pQlight        # lighting electric load [kW]
    tm["Qequip"] = df_tm.pQequip        # equipment electric load [kW]
    tm["Tout"] = df_tm.pTout            # outdoor temperature [°C]
    tm["Tmx"] = df_tm.pTmx              # maximum indoor temperature [°C]
    tm["Tmn"] = df_tm.pTmn              # minimum indoor temperature [°C]
    tm["Ton"] = df_tm.pTon              # forced comfort temperature {0,1}
    tm["HWdem"] = df_tm.pHWdem          # hot water demand [kWh]
    tm["Sdni"] = df_tm.pSdni            # direct normal irradiance [kW/m2]
    tm["Sdhi"] = df_tm.pSdhi            # diffuse horizontal irradiance [kW/m2]
    tm["Wms"] = df_tm.pWms              # wind speed [m/s]
    tm["EVdem"] = df_tm.pEVdem          # electric vehicle demand [km]
    tm["EVtm"] = df_tm.pEVtm            # electric vehicle charging window [h]
    tm["EVtype"] = df_tm.pEVtype        # electric vehicle type [0,...,n]
    tm["Bppl"] = df_tm.pBppl            # building occupancy per time period [°]
    
    # convert date times into UTC
    tm["Putc"] = convert_to_utc(tm,bdg)

    prepend!(tm["Date"],[cfg["P0"]])     # add initial date

    # calculate period durations in hours
    tm["TM"] = calculate_durations(tm)
    # calculate number of hours
    tm["H"] = calculate_nhours(tm)

    deleteat!(tm["Date"],1)             # remove initial date
    tm["P"] = length(tm["Date"])        # number of periods

    # enable or disable temperature control
    cfg["b_temp"]==true ? tm["Ton"]=tm["Ton"] : tm["Ton"].=0

    # allow initial free installation
    tm["IW"] = Int32[1]
    # when investments are allowed
    if cfg["b_inv"]==true
        # create investment windows
        tm["IW"] = investment_windows(cfg["IT"],tm["H"],tm["TM"])
    end

    if cfg["QmxTM"]==0
        tm["QmxCostN"] = zeros(Float64, cfg["QmxTM"])
    else
        tm["QmxCostN"] = calculate_peak_charge(cfg["QmxTM"],tm["Qmx"],tm["QmxCost"],tm["P"])
    end

    tm["Qihg_P"],tm["Qihg_L"],tm["Qihg_E"] = calculate_internal_heat_gain(tm["Bppl"],
        tm["Qlight"],tm["Qequip"],bdg["Bihgp"],bdg["Bihgl"],bdg["Bihge"])

    tm["ele"],tm["azi"] = calculate_sun_position(tm["Putc"],bdg["Balt"],bdg["Blat"],
        bdg["Blon"])

    tm["Q_R"] = calculate_solar_radiation(bdg,tm["Sdni"],tm["ele"],tm["azi"])

    return tm

end


function convert_to_utc(tm::Dict,bdg::Dict)
    tz = TimeZone(bdg["Btz"])
    return map(tm["Date"]) do dt
        try
            DateTime(ZonedDateTime(dt, tz, 1), UTC)
        catch e
            e isa TimeZOnes.NonExistentTimeError || rethrow()
            @warn "local time does not exist (DST gap); resolved via preceding minute" datetime=dt
            DateTime(ZonedDateTime(dt - Minute(1), tz, 1), UTC) + Minute(1)
        end
    end
end

function calculate_durations(tm::Dict)
    return Dates.value.(tm["Date"][2:end]-tm["Date"][1:(end-1)])/1000/60/60
end

function calculate_nhours(tm::Dict)
    return Dates.value.(tm["Date"][end]-tm["Date"][1])/1000/60/60
end

function investment_windows(IT,H,TM)
    
    IW = Int32[]    # investment windows hours

    # obtain simulation hour for investment in reverse order
    IH = floor(Int32,H/IT).*(1:IT)
    TM_cum = reverse(cumsum(reverse(TM)))

    for i in IH
        push!(IW,argmin(abs.(TM_cum.-i)))
    end

    # investment always happening in first simulation hour
    replace!(IW,minimum(IW)=>1)
    # undo reverse order for investment windows
    reverse!(IW)

    return IW

end

function calculate_peak_charge(QmxTM,Qmx,QmxCost,nP)

    QmxCostN = zeros(Float64, QmxTM)
    for t=1:nP
        for n=1:QmxTM
            if Qmx[t]==n
                QmxCostN[n] = QmxCost[t]
                break
            end
        end
    end

    return QmxCostN

end

function calculate_internal_heat_gain(occupancy,lighting,equipment,ihg_people,
    ihg_light,ihg_equipment)

    return ihg_people.*occupancy, ihg_light.*lighting, ihg_equipment.*equipment

end

function calculate_sun_position(utc,alt,lat,lon)

    # calculate the Sun position: elevation/azimuth
    jd = datetime2julian.(utc)                  # julian date
    ra,dec = sunpos(jd)                         # equatorial coordinates
    out = eq2hor.(ra,dec,jd,lat,lon,alt)        # horizontal coordinates
    sun_alt = [t[1] for t in out]
    sun_az  = [t[2] for t in out]
    ele = max.(sun_alt*pi/180,0)                # elevation [rad]
    azi = (ele.!=0).*sun_az*pi/180              # azimuth [rad]

    return ele,azi

end

function calculate_solar_radiation(bdg,radiation,elevation,azimuth)

    Q_Rroof  = bdg["Bkroof"] *bdg["Bfoot"] *sin.(elevation).*radiation
    Q_Rwest  = bdg["Bkwest"] *bdg["Bwest"] *cos.(elevation).*max.(-sin.(azimuth),0).*radiation
    Q_Reast  = bdg["Bkeast"] *bdg["Beast"] *cos.(elevation).*max.( sin.(azimuth),0).*radiation
    Q_Rnorth = bdg["Bknorth"]*bdg["Bnorth"]*cos.(elevation).*max.( cos.(azimuth),0).*radiation
    Q_Rsouth = bdg["Bksouth"]*bdg["Bsouth"]*cos.(elevation).*max.(-cos.(azimuth),0).*radiation

    return Q_Rroof+Q_Rwest+Q_Reast+Q_Rnorth+Q_Rsouth

end