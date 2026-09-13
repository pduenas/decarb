# DECARB — example driver.  Run from the repository root:
#   julia --project=. run.jl [case_directory]

using Pkg; Pkg.instantiate()
using DECARB

case      = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "examples", "minimal")
mip_gap   = 1e-2      # relative MIP gap
time_limit = 300.0    # seconds
solver    = 2         # 1 = Gurobi (licence required), 2 = HiGHS (open source)
relax     = false     # true solves the LP relaxation only

DECARB.run_decarb!(case, mip_gap, time_limit, solver, relax)

nothing
