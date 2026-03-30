"""
Use this file to configure the Gurobi settings. Add new settings if desired.
"""

function configure_gurobi(model::Model)
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
    set_optimizer_attribute(model, "MIPGap", 1e-2) #  5e-2
    set_optimizer_attribute(model, "NumericFocus", 1)
    set_optimizer_attribute(model, "OptimalityTol", 1e-9) 
    set_optimizer_attribute(model, "TimeLimit", 300)
end
