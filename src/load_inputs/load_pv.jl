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
    pv["eff"] = df_pv.eff       # DC/AC efficiency [%]
    pv["loss"] = df_pv.loss     # PV system losses [%]
    pv["fail"] = df_pv.fail     # PV failure rate [%]
    pv["inv"] = df_pv.inv       # capital cost [$]
    pv["lc"] = df_pv.lc         # learning curve [%/year]
    pv["fom"] = df_pv.fom       # fixed O&M cost [$/year]
    pv["life"] = df_pv.life     # lifetime [years]

    pv["N"] = size(pv["ty"],1)  # number of PV panels

    # maximum potential number of PV panels [0,z]
    pv["zmx"] = floor.(bdg["Bpv"]./pv["ar"])

    # maximum PV panel potential [kW/m2]
    pv["Sdnimx"] = maximum_solar_potential(pv["tech"],bdg["Btilt"],bdg["Bazi"],
        bdg["Btck"],tm["ele"],tm["azi"],tm["Sdni"],tm["Sdhi"],tm["P"],tm["Tout"])

    return pv

end

"""
same PV solar model as renewables.ninja: Huld 2010, Pfenninger 2016
"""

function maximum_solar_potential(tech,pvtilt,pvazi,pvtck,ele,azi,
    sdni,sdhi,np,tout)

    # calculate the incident angle and correct the tilt angle
    if pvtck==0			# fixed axis
        incid = acos.(sin.(ele).*cos(pvtilt)+cos.(ele).*sin(pvtilt).*cos.(pvazi.-azi))
    elseif pvtck==1		# single-axis
        incid = acos.(sqrt.(1 .- (cos.(ele.+pvtilt)-cos(pvtilt).*cos.(ele).*(1 .- cos.(azi.-pvazi))).^2))
        pvtilt = atan.(cos.(ele).*sin.(azi.-pvazi)./(sin.(ele.-pvtilt).+sin(pvtilt).*cos.(ele).*  (1 .- cos.(azi.-pvazi))))
    elseif pvtck==2		# dual-axis
        incid = zeros(Float64, np)
        pvtilt = pi/2 .- ele
    end

    # calculate the normalized flat irradiance based on real DNI and DHI
    dni = sdni.*cos.(incid)
    dhi = sdhi.*(1 .+ cos.(pvtilt))/2 .+ 0.3.*(sdni.+sdhi).*(1 .- cos.(pvtilt))/2
    ir = dni+dhi
    ir[ir.<0] .= 0

    # input: efficiency parameters for each type of PV technology
    k = [-0.017162	-0.040289	-0.004681	0.000148	0.000169	0.000005;
        -0.005554	-0.038724	-0.003723	-0.000905	-0.001256	0.000001;
        -0.046689 	-0.072844	-0.002262	0.000276	0.000159	-0.000006]

    # calculate PV panel efficieny depending on type
    G_ = transpose(log.(ir))                            # natural logarithm
    T_ = transpose(tout.+0.035*1000*(ir).-25)           # temperature of PV panel
    eff = 1 .+ k[tech,1].*G_ .+ k[tech,2].*G_.^2 .+     # empirical formula
        T_.*(k[tech,3].+k[tech,4].*G_.+k[tech,5].*G_.^2) .+ k[tech,6].*T_.^2
    eff[eff.<0] .= 0
    eff[eff.>1] .= 1
    replace!(eff,NaN=>0)
    
    return transpose(eff).*ir

end