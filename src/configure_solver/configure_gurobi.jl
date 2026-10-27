"""
configure_gurobi(model::Model,mip_gap::Real,time_limit::Real)

Applies Gurobi solver attributes to the model.

Note: Add new settings if desired.

inputs:
model       optimization model object
mip_gap     relative MIP gap tolerance for termination
time_limit  maximum solve time in seconds

"""

function configure_gurobi(model::Model,mip_gap::Real,time_limit::Real)
    set_attribute(model, "AggFill", 0)
    set_attribute(model, "DisplayInterval", 1)
    set_attribute(model, "GomoryPasses", 0)
    set_attribute(model, "FeasibilityTol", 1e-9)
    set_attribute(model, "Heuristics", 1e-3)
    set_attribute(model, "IntFeasTol", 1e-9)
    set_attribute(model, "MarkowitzTol", 0.999)
    set_attribute(model, "RINS", 0)
    set_attribute(model, "Method", -1)
    set_attribute(model, "MIPFocus", 1)
    set_attribute(model, "MIPGap", Float64(mip_gap))
    set_attribute(model, "NumericFocus", 1)
    set_attribute(model, "OptimalityTol", 1e-9)
    set_attribute(model, "TimeLimit", Float64(time_limit))
end
