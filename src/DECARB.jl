"""
DECARB: Distributed Energy Consumption in Responsive Buildings
"""

module DECARB

export case_runner
export load_inputs
export configure_solver
export decarb_model
export solve_model
export read_outputs
export write_outputs

export run_decarb!

using CSV
using DataFrames
using JuMP
using Dates
using Libdl
using AstroLib
using TimeZones

# uncomment your solver
using Gurobi
#using CPLEX

# Launch DECARB process
include(joinpath("case_runner","case_runner.jl"))

# Load input data
include(joinpath("load_inputs","load_in.jl"))
include(joinpath("load_inputs","load_sp.jl"))
include(joinpath("load_inputs","load_tm.jl"))
include(joinpath("load_inputs","load_bdgi.jl"))
include(joinpath("load_inputs","load_bdgii.jl"))
include(joinpath("load_inputs","load_extended_catalog.jl"))
include(joinpath("load_inputs","load_chp.jl"))
include(joinpath("load_inputs","load_abs.jl"))
include(joinpath("load_inputs","load_hvac.jl"))
include(joinpath("load_inputs","load_topo.jl"))
include(joinpath("load_inputs","load_wh.jl"))
include(joinpath("load_inputs","load_pv.jl"))
include(joinpath("load_inputs","load_wind.jl"))
include(joinpath("load_inputs","load_bess.jl"))
include(joinpath("load_inputs","load_ev.jl"))

# Configure solver settings
include(joinpath("configure_solver","configure_gurobi.jl"))
include(joinpath("configure_solver","configure_cplex.jl"))

# Create DECARB model
include(joinpath("decarb_model","heat_connections.jl"))
include(joinpath("decarb_model","temperature_variables.jl"))
include(joinpath("decarb_model","chp_units.jl"))
include(joinpath("decarb_model","hvac_units.jl"))
include(joinpath("decarb_model","abs_chillers.jl"))
include(joinpath("decarb_model","water_heaters.jl"))
include(joinpath("decarb_model","pv_panels.jl"))
include(joinpath("decarb_model","wind_turbines.jl"))
include(joinpath("decarb_model","bess_modules.jl"))
include(joinpath("decarb_model","electric_vehicles.jl"))
include(joinpath("decarb_model","electric_load.jl"))
include(joinpath("decarb_model","thermal_load.jl"))
include(joinpath("decarb_model","objective_function.jl"))

# Solve DECARB model
include(joinpath("solve_model","solve_model.jl"))

# Read output data
include(joinpath("read_outputs","read_thermal.jl"))
include(joinpath("read_outputs","read_der.jl"))
include(joinpath("read_outputs","read_electric.jl"))
include(joinpath("read_outputs","read_fuel.jl"))
include(joinpath("read_outputs","read_indoor.jl"))
include(joinpath("read_outputs","read_econ.jl"))

# Write output files
include(joinpath("write_outputs","write_outputs.jl"))

end

""" THERMAL TOPOLOGY ... pending
    oCHPbdg_HT = zeros(Float64, P,0)
    s_cb_ht = String[]
    if isempty(vCHP_HT)==false
    	cb_ht = dropdims(any(value.(vCHPbdg[:,:,1]).!=0,dims=1); dims=1)
    	ncb_ht = sum(cb_ht)
    	s_cb_ht = string.(pCHPty[findall(cb_ht)]," >> ",pB," :heat")
    	itc = findall(cb_ht)
    	oCHPbdg_HT = zeros(Float64, P,ncb_ht)
        for p=1:P
            for c=1:ncb_ht
                oCHPbdg_HT[p,c] = round.(pTM[p]*pCHPmx[itc[c]]*pCHPbdg[itc[c],1]* value.(vCHPbdg[p,itc[c],1]), digits=2)
            end
        end
    end
    oCHPabs = zeros(Float64, P,0)
    s_ca = String[]
    if isempty(vCHPabs)==false
        ca = dropdims(any(value.(vCHPabs).!=0,dims=1); dims=1)
        nca = sum(ca)
        itca = findall(ca)
        oCHPabs = zeros(Float64, P,nca)
        s_ca = Array{String}(undef,nca)
        for ii=1:nca
            s_ca[ii] = join([pCHPty[itca[ii][1]]," >> ",pABSty[itca[ii][2]]])
            for p=1:P
                oCHPabs[p,ii] = round.(pTM[p]*pCHPabs[itca[ii]]*pCHPmx[findall(any(ca,dims=1))[ii]]* value.(vCHPabs[p,itca[ii]]), digits=2)
            end
        end
    end
    oCHPfire = zeros(Float64, P,0)
    s_cf = String[]
    if isempty(vCHPfire)==false
        cf = dropdims(any(value.(vCHPfire).!=0,dims=1); dims=1)
        ncf = sum(cf)
        itcf = findall(cf)
        oCHPfire = zeros(Float64, P,ncf)
        s_cf = Array{String}(undef,ncf)
        for ii=1:ncf
            itcf[ii][2]>A ? a=pB : a=pABSty[itcf[ii][2]]
            s_cf[ii] = join([pCHPty[itcf[ii][1]]," >> ",a])
            for p=1:P
                oCHPfire[p,ii] = round.(pCHPfire[itcf[ii]]*value.(vCHPfire[p,itcf[ii]]), digits=2)
            end
        end
    end
    oCHPbdg_HW = zeros(Float64, P,0)
    s_cb_hw = String[]
    if isempty(vCHP_HW)==false
        cb_hw = dropdims(any(value.(vCHPbdg[:,:,2]).!=0,dims=1); dims=1)
        ncb_hw = sum(cb_hw)
        s_cb_hw = string.(pCHPty[findall(cb_hw)]," >> ",pB," :DHW")
        itc = findall(cb_hw)
        oCHPbdg_HW = zeros(Float64, P,ncb_hw)
        for p=1:P
            for c=1:ncb_hw
                oCHPbdg_HW[p,c] = round.(pTM[p]*pCHPmx[itc[c]]*pCHPbdg[itc[c],2]* value.(vCHPbdg[p,itc[c],2]), digits=2)
            end
        end
    end
    dfHT = hcat(DataFrame(Date=pP),DataFrame(oCHPbdg_HT,s_cb_ht),DataFrame(oABSbdg_HT,s_ab_ht), DataFrame(oCHPbdg_HW,s_cb_hw),DataFrame(oABSbdg_HW,s_ab_hw),DataFrame(oCHPabs,s_ca), DataFrame(oCHPfire,s_cf),DataFrame(oABSfire,s_af))
    CSV.write(joinpath(p,"out","ht.csv"),dfHT,dateformat="mm/dd/yyyy HH:MM")
"""
