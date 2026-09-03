"""
Use this file to configure the HiGHS settings. Add new settings if desired.
"""

function configure_highs(model::Model,mip_gap::Float64,time_limit::Float64)
    set_optimizer_attribute(model, "mip_rel_gap", mip_gap)
    set_optimizer_attribute(model, "time_limit", time_limit)
    set_optimizer_attribute(model, "parallel", "on")
    set_optimizer_attribute(model, "threads", Sys.CPU_THREADS)
    set_optimizer_attribute(model, "presolve", "on")
end