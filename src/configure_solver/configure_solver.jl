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
        gap_attr = MOI.RelativeGapTolerance()
        if MOI.supports(backend(model), gap_attr)
            set_attribute(model, gap_attr, Float64(mip_gap))
        else
            @warn "Optimizer does not support MOI.RelativeGapTolerance; skipping DECARB default" maxlog=1
        end

        time_attr = MOI.TimeLimitSec()
        if MOI.supports(backend(model), time_attr)
            set_attribute(model, time_attr, Float64(time_limit))
        else
            @warn "Optimizer does not support MOI.TimeLimitSec; skipping DECARB default" maxlog=1
        end
    end
    return nothing
end
