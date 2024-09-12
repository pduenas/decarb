using DECARB

# define solver 
#   1: Gurobi, 2: CPLEX
i_solver = 1
# define integrality
b_relax_integrality = false

run_decarb!(dirname(@__FILE__),i_solver,b_relax_integrality)
