"""
configure_highs(model::Model,mip_gap::Real,time_limit::Real)

Applies HiGHS solver attributes to the model.

Note: Add new settings if desired.

inputs:
model       optimization model object
mip_gap     relative MIP gap tolerance for termination
time_limit  maximum solve time in seconds

"""

function configure_highs(model::Model,mip_gap::Real,time_limit::Real)
    set_optimizer_attribute(model, "mip_rel_gap", Float64(mip_gap))
    set_optimizer_attribute(model, "time_limit", Float64(time_limit))
    set_optimizer_attribute(model, "parallel", "on")
    set_optimizer_attribute(model, "presolve", "on")
end
