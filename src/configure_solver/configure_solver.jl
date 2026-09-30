"""
configure_solver!(model::Model, mip_gap::Real, time_limit::Real)

Apply defaults for Gurobi and HiGHS, or standard gap and time
limits for other optimizers.

Note: New solvers can be added to this function by including a new 
configure_*.jl file in the configure_solver folder and adding a new elseif clause below.

inputs:
model       optimization model object
mip_gap     relative MIP gap tolerance for termination
time_limit  maximum solve time in seconds

"""

function configure_solver!(model::Model, mip_gap::Real, time_limit::Real)
    name = lowercase(solver_name(model))
    if occursin("gurobi", name)
        configure_gurobi(model, mip_gap, time_limit)
    elseif occursin("highs", name)
        configure_highs(model, mip_gap, time_limit)
    else
        set_attribute(model, MOI.RelativeGapTolerance(), Float64(mip_gap))
        set_attribute(model, MOI.TimeLimitSec(), Float64(time_limit))
    end
    return nothing
end
