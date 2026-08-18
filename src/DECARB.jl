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