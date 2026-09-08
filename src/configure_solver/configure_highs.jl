"""
configure_highs(model::Model,mip_gap::Float64,time_limit::Float64)

Applies HiGHS solver attributes to the model. Add new settings if desired

inputs:
model       optimization model object
mip_gap     relative MIP gap tolerance for termination
time_limit  maximum solve time in seconds

"""

function configure_highs(model::Model,mip_gap::Float64,time_limit::Float64)
    set_optimizer_attribute(model, "mip_rel_gap", mip_gap)
    set_optimizer_attribute(model, "time_limit", time_limit)
    set_optimizer_attribute(model, "parallel", "on")
    set_optimizer_attribute(model, "threads", Sys.CPU_THREADS)
    set_optimizer_attribute(model, "presolve", "on")
end