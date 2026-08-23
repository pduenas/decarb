"""
Use this file to configure the HiGHS settings. Add new settings if desired.
"""

function configure_highs(model::Model)
    set_optimizer_attribute(model, "mip_rel_gap", 1e-2)
    set_optimizer_attribute(model, "time_limit", 300.0)
    set_optimizer_attribute(model, "parallel", "on")
    set_optimizer_attribute(model, "threads", 10)
    set_optimizer_attribute(model, "presolve", "on")
end