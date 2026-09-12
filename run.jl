using DECARB

# define solver 
#   1: Gurobi, 2: HiGHS
i_solver = 1
# define MIP gap and time limit
mip_gap = 1e-2
time_limit = 300.0
# define integrality
b_relax_integrality = false

run_decarb!(dirname(@__FILE__),mip_gap,time_limit,i_solver,b_relax_integrality);

nothing
