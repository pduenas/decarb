"""
DECARB: Distributed Energy Consumption in Responsive Buildings
"""

module DECARB

export run_decarb!

using CSV
using DataFrames
using JuMP
using Dates
using AstroLib
using TimeZones
using PrecompileTools

using HiGHS

# Launch DECARB process
include(joinpath("case_runner","case_runner.jl"))

# Load input data
include(joinpath("load_inputs","load_cfg.jl"))
include(joinpath("load_inputs","load_sp.jl"))
include(joinpath("load_inputs","load_tm.jl"))
include(joinpath("load_inputs","load_bdgi.jl"))
include(joinpath("load_inputs","load_bdgii.jl"))
include(joinpath("load_inputs","load_extended_catalog.jl"))
include(joinpath("load_inputs","load_chp.jl"))
include(joinpath("load_inputs","load_abp.jl"))
include(joinpath("load_inputs","load_hvac.jl"))
include(joinpath("load_inputs","load_topo.jl"))
include(joinpath("load_inputs","load_wh.jl"))
include(joinpath("load_inputs","load_pv.jl"))
include(joinpath("load_inputs","load_wind.jl"))
include(joinpath("load_inputs","load_bess.jl"))
include(joinpath("load_inputs","load_ev.jl"))
include(joinpath("load_inputs","validate_inputs.jl"))

# Configure solver settings
include(joinpath("configure_solver","configure_gurobi.jl"))
include(joinpath("configure_solver","configure_highs.jl"))
include(joinpath("configure_solver","configure_solver.jl"))

# Create DECARB model
include(joinpath("decarb_model","heat_connections.jl"))
include(joinpath("decarb_model","temperature_variables.jl"))
include(joinpath("decarb_model","chp_units.jl"))
include(joinpath("decarb_model","hvac_units.jl"))
include(joinpath("decarb_model","abp_chillers.jl"))
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

@setup_workload begin
    case = mktempdir()
    try
        input_dir = joinpath(case, "in")
        shipped_inputs = normpath(joinpath(@__DIR__, "..", "examples", "minimal", "in"))
        if isdir(shipped_inputs)
            cp(shipped_inputs, input_dir; force=true)
        else
            mkpath(input_dir)
            fallback_inputs = Dict(
                "cfg.csv" => """
                    pP0,2019-01-01T05:00:00
                    pTmode,true
                    pBESSsoc0,0.5
                    pBESSsocf,0.5
                    pEVsoc0,0.8
                    pEVmnsoc,0.3
                    pWHsto0,0.5
                    pWHstof,0.5
                    pTin0,20.0
                    pQmxBuy,100.0
                    pQmxSell,100.0
                    pST,0
                    pGco2,0.18
                    pLco2,0.25
                    pCO2mx,0.0
                    pCO2cost,0.0
                    pCO2ton,0.0
                    pQmxTM,0
                    pNSEcost,5.0
                    pNSTcost,1.0
                    pNSHWcost,1.0
                    pNSEVcost,1.0
                    pEVdriver,0
                    pEVpen,0.0
                    pEVv2g,false
                    pIT,0
                    pIR,0.06
                    """,
                "sp.csv" => """
                    pCHP0,pCHPz0,pCHPyn,pABP0,pABPz0,pABPyn,pHVAC0,pHVACz0,pHVACyn,pWH0,pWHz0,pWHyn,pPV0,pPVz0,pPVyn,pWIND0,pWINDz0,pWINDyn,pBESS0,pBESSz0,pBESSyn,pEV0,pEVz0
                    0,0,0,0,0,0,HP-MINISPLIT,1,NO,WH-TANKLESS-GAS,1,NO,PV-MONO,4,NO,0,0,0,0,0,0,0,0
                    """,
                "bdg_i.csv" => """
                    pBalt,10.0
                    pBlon,-71.06
                    pBlat,42.36
                    pBtz,America/New_York
                    pBihgp,0.1
                    pBihgl,0.9
                    pBihge,0.8
                    pBfoot,120.0
                    pBkroof,0.06
                    pBwest,30.0
                    pBkwest,0.03
                    pBeast,30.0
                    pBkeast,0.03
                    pBnorth,45.0
                    pBknorth,0.01
                    pBsouth,45.0
                    pBksouth,0.15
                    pBchp,0
                    pBhvac,1
                    pBabp,0
                    pBwh,1
                    pBpv,20.0
                    pBtilt,30.0
                    pBazi,180.0
                    pBalb,0.2
                    pBtck,0
                    pBwind,0
                    pBbess,0
                    pBevmx,7.4
                    """,
                "bdg_ii.csv" => """
                    pBk1,pBk2,pBk3,pBtime
                    0.03,0.05,0.15,1
                    """,
                "tm.csv" => """
                    pP,pQcostBuy,pQcostSell,pQmx,pQmxCost,pQco2,pGcost,pLcost,pQlight,pQequip,pTout,pTmx,pTmn,pTon,pHWdem,pSdni,pSdhi,pWms,pEVdem,pEVtm,pEVtype,pBppl
                    2019-01-01T06:00:00,0.18,0.05,0,0.0,0.42,0.05,0.09,0.10,0.40,-1.0,26.0,20.0,1,0.0,0.0,0.0,4.2,0.0,0.0,0,2.0
                    2019-01-01T07:00:00,0.18,0.05,0,0.0,0.42,0.05,0.09,0.10,0.40,-1.5,26.0,20.0,1,0.0,0.0,0.0,4.0,0.0,0.0,0,2.0
                    """,
                "chp.csv" => """
                    ty,mx,mn,fcf,fuel,tank,h2p,sup,inv,fom,vom,life
                    MICROTURB,30.0,0.3,0.28,G,0.0,1.6,5.0,45000.0,900.0,0.012,20.0
                    """,
                "abp.csv" => """
                    ty,mx,fcf,fuel,ac,inv,fom,vom,life
                    ABSCHILL,20.0,0.75,G,0.7,18000.0,350.0,0.004,20.0
                    """,
                "hvac.csv" => """
                    ty,pHVmx,pACmx,pHVeff,pACeff,temp,pHVmx_,pACmx_,pHVeff_,pACeff_,inv,fom,vom,life
                    HP-MINISPLIT,8.0,6.0,3.0,3.5,8.33,0.005,0.005,0.03,0.02,4500.0,50.0,0.002,15.0
                    """,
                "wh.csv" => """
                    ty,mx,fcf,fuel,eff,tank,inv,fom,vom,life
                    WH-TANKLESS-GAS,12.0,0.82,G,0.95,0.0,1200.0,25.0,0.001,12.0
                    """,
                "pv.csv" => """
                    ty,tech,mx,ar,eff,loss,fail,inv,lc,fom,life
                    PV-MONO,1,0.4,1.8,0.96,0.14,0.005,320.0,0.02,4.0,25.0
                    """,
                "wind.csv" => """
                    ty,mx,mn,vmn,vmx,fail,inv,lc,fom,life
                    WT-SMALL,3.0,0.2,3.0,12.0,0.02,9000.0,0.01,150.0,20.0
                    """,
                "bess.csv" => """
                    ty,mx,up,dn,effu,effd,inv,lc,fom,life
                    LI-13KWH,13.5,5.0,5.0,0.95,0.95,9000.0,0.03,20.0,10.0
                    """,
                "ev.csv" => """
                    ty,mx,drv,cold,up,dn,effu,effd
                    EV-SEDAN,60.0,0.18,0.15,7.4,7.4,0.94,0.94
                    """,
                "topo.csv" => """
                    up,lo,in,eff
                    MICROTURB,Building,hot water,0.85
                    """,
            )
            for (file, contents) in fallback_inputs
                write(joinpath(input_dir, file), contents)
            end
        end

        tm_file = joinpath(input_dir, "tm.csv")
        tm_lines = readlines(tm_file)
        write(tm_file, join(tm_lines[1:min(length(tm_lines), 3)], '\n') * "\n")

        @compile_workload begin
            redirect_stderr(devnull) do
                cfg = load_cfg(input_dir)
                sp = load_sp(input_dir, cfg)
                bdg = load_bdgii(input_dir, load_bdgi(input_dir))
                tm = load_tm(input_dir, cfg, bdg)
                pv = load_pv(input_dir, tm, bdg)
                chp, abp, hvac, wh, ev = load_extended_catalog(input_dir, tm, sp, bdg)
                wind = load_wind(input_dir, tm)
                bess = load_bess(input_dir)
                topo = load_topo(input_dir, chp, abp)

                validate_inputs(cfg, sp, bdg, tm, chp, abp, hvac, wh, pv, wind, bess, ev, topo;
                    io=devnull)
                warning_sp = deepcopy(sp)
                warning_sp["PVz0"][1] = 0
                validate_inputs(cfg, warning_sp, bdg, tm, chp, abp, hvac, wh, pv, wind, bess, ev, topo;
                    io=IOBuffer())

                run_decarb!(case; verbose=false)
                run_decarb!(case; verbose=false, duals=false)
                run_decarb!(case; verbose=false, relax_integrality=true)
                redirect_stdout(devnull) do
                    run_decarb!(case; verbose=true)
                end
            end
        end
    finally
        rm(case; recursive=true, force=true)
    end
end

end
