# DECARB — example driver.  Run from the repository root:
#   julia --project=. run.jl [case_directory]

using DECARB
using HiGHS

case      = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "examples", "minimal")
mip_gap   = 1e-4      # relative MIP gap
time_limit = 300.0    # seconds
optimizer = HiGHS.Optimizer  # or another installed JuMP optimizer
relax     = false     # true solves the LP relaxation only

DECARB.run_decarb!(case; mip_gap, time_limit, optimizer,
    relax_integrality=relax)

nothing
