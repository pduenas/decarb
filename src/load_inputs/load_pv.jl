"""
load_pv(path::AbstractString,tm::Dict,bdg::Dict)

Loads pv.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory
tm      dictionary with time series data
bdg     dictionary with building data

returns pv-type inputs in dictionary object
"""

function load_pv(path::AbstractString,tm::Dict,bdg::Dict)

    # declare dictionary object to store parameters
    pv = Dict()

    path2file = joinpath(path,"pv.csv")

    # load file into dataframe with predefined types
    df_pv = CSV.File(path2file;delim=',',types=[String,Int8,Float64,Float64,Float64,
                    Float64,Float64,Float64,Float64,Float64,Float64]) |> DataFrame

    # delete non-existing rows
    filter!(row -> !ismissing(row.ty), df_pv)
    # delete all zero rows
    filter!(row -> !all(x -> x==0, row), df_pv)
    # delete zero-capacity elemenets
    filter!(row -> !iszero(row.mx), df_pv)
    # delete repeated rows
    unique!(df_pv)

    pv["ty"] = df_pv.ty         # name of PV panel
    pv["tech"] = df_pv.tech     # technology of PV power
    pv["mx"] = df_pv.mx         # module PV capacity [kW]
    pv["ar"] = df_pv.ar         # module PV area [m2]
    pv["eff"] = df_pv.eff       # DC/AC efficiency [p.u.]
    pv["loss"] = df_pv.loss     # PV system losses [p.u.]
    pv["fail"] = df_pv.fail     # PV failure rate [p.u.]
    pv["inv"] = df_pv.inv       # capital cost [$]
    pv["lc"] = df_pv.lc         # learning curve [%/year]
    pv["fom"] = df_pv.fom       # fixed O&M cost [$/year]
    pv["life"] = df_pv.life     # lifetime [years]

    pv["N"] = size(pv["ty"],1)  # number of PV panels

    # maximum potential number of PV panels [0,z]
    pv["zmx"] = floor.(bdg["Bpv"]./pv["ar"])

    # PV panel capacity factor [kW/m2]
    pv["pv_cf"] = maximum_solar_potential(pv["tech"],bdg["Btilt"],bdg["Bazi"],bdg["Btck"],
        tm["ele"],tm["azi"],tm["Sdni"],tm["Sdhi"],bdg["Balb"],tm["P"],tm["Tout"])

    return pv

end

"""
same PV solar model as renewables.ninja: Huld 2010, Pfenninger 2016
"""

function maximum_solar_potential(tech,pvtilt,pvazi,pvtck,ele,azi,
    sdni,sdhi,albedo,np,tout)

    # Solar projections into x,y,z coordinates
    sunx = sin.(azi).*cos.(ele)
    suny = cos.(azi).*cos.(ele)
    sunz = sin.(ele)

    # Fixed panel: normal is defined by the panel tilt and azimuth
    if pvtck==0
        nx = sin.(pvtilt).*sin.(pvazi)
        ny = sin.(pvtilt).*cos.(pvazi)
        nz = cos.(pvtilt)
        cosinc = sunx.*nx .+ suny.*ny .+ sunz.*nz
    # Single-axis tracker: axis is defined by (pvtilt, pvazi)
    elseif pvtck==1
        ax = sin.(pvtilt).*sin.(pvazi)
        ay = sin.(pvtilt).*cos.(pvazi)
        az = cos.(pvtilt)
        # panel normal is sun vector projected onto plane perpendicular to that axis
        d = sunx.*ax .+ suny.*ay .+ sunz.*az                # projection of sun onto axis
        projx = sunx .- d.*ax
        projy = suny .- d.*ay
        projz = sunz .- d.*az
        projn = sqrt.(projx.^2 .+ projy.^2 .+ projz.^2)     # magnitude of projected sun vector
        cosinc = projn
    # Dual-axis tracker: panel normal matches the sun vector for daylight hours.
    elseif pvtck==2
        cosinc = ones(Float64, np)
    else
        error("❗  Invalid PV tracking type. Choose 0=fixed, 1=single-axis, or 2=dual-axis.")
    end

    incid = acos.(clamp.(cosinc, -1, 1))

    # calculate the normalized flat irradiance based on real DNI and DHI
    dni = max.(sdni.*cos.(incid), 0)            # avoid negative values
    dhi = max.(sdhi, 0)                         # avoid negative values
    ghi = max.(sdni.*sin.(ele), 0).+dhi         # obtain the global horizontal irradiance

    sky = dhi.*(1 .+ cos.(pvtilt))./2
    ground = albedo.*ghi.*(1 .- cos.(pvtilt))./2 
    
    ir = dni .+ sky .+ ground
    ir[ir.<0] .= 0

    # input: efficiency parameters for each type of PV technology
    k = [-0.017162	-0.040289	-0.004681	0.000148	0.000169	0.000005;
        -0.005554	-0.038724	-0.003723	-0.000905	-0.001256	0.000001;
        -0.046689 	-0.072844	-0.002262	0.000276	0.000159	-0.000006]

    # calculate PV panel efficiency depending on technology and active irradiance periods
    idx = ir .> 0
    eff = zeros(Float64, length(ir), length(tech))
    G_ = transpose(log.(ir[idx]))
    T_ = transpose(tout[idx].+0.035*1000*(ir[idx]).-25)
    eff_active = 1 .+ k[tech,1].*G_ .+ k[tech,2].*G_.^2 .+
        T_.*(k[tech,3].+k[tech,4].*G_.+k[tech,5].*G_.^2) .+ k[tech,6].*T_.^2
    eff[idx, :] = transpose(clamp.(eff_active, 0, 1))

    return eff.*ir

end