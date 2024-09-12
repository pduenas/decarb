"""
Use this file to configure the CPLEX settings. Add new settings if desired.
"""

function configure_cplex(model::Model)
    set_optimizer_attribute(model, "CPXPARAM_LPMethod", 4)
    set_optimizer_attribute(model, "CPXPARAM_QPMethod", 4)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Strategy_StartAlgorithm", 4)
    set_optimizer_attribute(model, "CPXPARAM_Emphasis_MIP", 1)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Strategy_RINSHeur", -1)
    set_optimizer_attribute(model, "CPXPARAM_Simplex_Tolerances_Markowitz", 0.999)
    set_optimizer_attribute(model, "CPXPARAM_Simplex_Tolerances_Optimality", 1e-9)
    set_optimizer_attribute(model, "CPXPARAM_Simplex_Tolerances_Feasibility", 1e-9)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Tolerances_Integrality", 1e-9)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Tolerances_MIPGap", 1e-2)
end
