"""
configure_gurobi(model::Model,mip_gap::Float64,time_limit::Float64)

Applies Gurobi solver attributes to the model. Add new settings if desired

inputs:
model       optimization model object
mip_gap     relative MIP gap tolerance for termination
time_limit  maximum solve time in seconds

"""

struct GurobiBackend end

function gurobi_optimizer(::Any)
    error("Gurobi support is optional. Install and load it with " *
          "`import Pkg; Pkg.add(\"Gurobi\"); using Gurobi, DECARB` before " *
          "selecting the Gurobi solver.")
end

function configure_gurobi(model::Model,mip_gap::Float64,time_limit::Float64)
    set_optimizer_attribute(model, "AggFill", 0)
    set_optimizer_attribute(model, "DisplayInterval", 1)
    set_optimizer_attribute(model, "GomoryPasses", 0)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-9)
    set_optimizer_attribute(model, "Heuristics", 1e-3)
    set_optimizer_attribute(model, "IntFeasTol", 1e-9)
    set_optimizer_attribute(model, "MarkowitzTol", 0.999)
    set_optimizer_attribute(model, "RINS", 0)
    set_optimizer_attribute(model, "Method", -1)
    set_optimizer_attribute(model, "MIPFocus", 1)
    set_optimizer_attribute(model, "MIPGap", mip_gap)
    set_optimizer_attribute(model, "NumericFocus", 1)
    set_optimizer_attribute(model, "OptimalityTol", 1e-9) 
    set_optimizer_attribute(model, "TimeLimit", time_limit)
end
