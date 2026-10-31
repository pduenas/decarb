"""
solve_model!(path::AbstractString,model::Model,relax_int::Bool)

Optimizes the model, checks feasibility, and writes an IIS report if infeasible. If the
model is feasible, duals is true, and relax_int is false, integer/binary variables are fixed
at their rounded solution values and the model is re-solved relaxed to obtain dual
information. If that re-solve returns no feasible point, the original integrality is
restored, the saved MIP solution is used as start values, the MIP is re-optimized, and
duals are reported as unavailable.

inputs:
path            string path to working directory
model           optimization model object
relax_int       boolean flag to relax integrality constraints on integer variables

returns a named tuple with the MIP termination_status, primal_status, objective_value,
objective_bound, relative_gap and solve_time_s (captured before the dual re-solve),
solver_name, solver_version, threads ("unset" if not set), and
duals_available, which is false when duals were not requested or could not be obtained.

"""

function solve_model!(path::AbstractString,model::Model,relax_int::Bool;
    duals::Bool=true, verbose::Bool=true)
    progress_io = verbose ? stdout : devnull

    # relax integrality {true,false}
    if relax_int==true
        relax_integrality(model)
    end

    optimize!(model)    # solve model
    s = termination_status(model)

    ok_status = s in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.TIME_LIMIT,
        MOI.NODE_LIMIT, MOI.SOLUTION_LIMIT, MOI.INTERRUPTED)

    if !(ok_status && primal_status(model) == MOI.FEASIBLE_POINT)
        try
            compute_conflict!(model)
            if get_attribute(model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
                println(progress_io, "\n❗ IIS detected — infeasible constraints:\n")
                open(joinpath(path,"out","iis_constraints.txt"), "w") do io
                    for (F, S) in list_of_constraint_types(model)
                        F == VariableRef && continue
                        for con in all_constraints(model, F, S)
                            cs = get_attribute(con, MOI.ConstraintConflictStatus())
                            if cs == MOI.IN_CONFLICT
                                label = name(con) == "" ? string(con) : name(con)
                                println(io, "[IN_CONFLICT] ", label)
                                println(progress_io, "  ❌ ", label)
                            end
                        end
                    end
                end
            end
        catch e
            @warn "Conflict computation unavailable" exception=e
        end
        error("❗  Model has no feasible solution. Check input data.\n(status = $(s))")
    end

    gap = try relative_gap(model) catch; NaN end
    verbose && @info "termination = $(s), gap = $(round(gap*100, digits=1))%"

    # solution summary captured before any dual re-solve overwrites it
    info = (termination_status = s,
        primal_status = primal_status(model),
        objective_value = objective_value(model),
        objective_bound = try objective_bound(model) catch; NaN end,
        relative_gap = gap,
        solve_time_s = try solve_time(model) catch; NaN end,
        solver_name = try solver_name(model) catch; "unknown" end,
        solver_version = try get_attribute(model, MOI.SolverVersion()) catch; "unknown" end,
        threads = try
            something(get_attribute(model, MOI.NumberOfThreads()), "unset")
        catch
            "unset"
        end,
        duals_available = false)

    dual_resolve_status = "n/a"
    mip_resolved = false
    resolved_objective_value = "n/a"
    resolved_relative_gap = "n/a"
    resolved_solve_time_s = "n/a"

    # fix integer variables at rounded values and re-solve the LP to get dual information
    if duals && !relax_int
        vALL = all_variables(model)
        v_L = value.(vALL)      # saved MIP solution
        undo = fix_discrete_variables(var -> round(value(var)), model)
        optimize!(model)
        if primal_status(model) != MOI.FEASIBLE_POINT
            dual_resolve_status = termination_status(model)
            @warn "relaxed re-solve returned no solution; restoring the MIP solution, duals will be zero" status=dual_resolve_status
            undo()
            set_start_value.(vALL, v_L)
            optimize!(model)
            if primal_status(model) != MOI.FEASIBLE_POINT
                error("❗ MIP could not be re-optimized after failed dual re-solve.\n(status = $(termination_status(model)))")
            end
            mip_resolved = true
            resolved_objective_value = objective_value(model)
            resolved_relative_gap = try relative_gap(model) catch; NaN end
            resolved_solve_time_s = try solve_time(model) catch; NaN end
        end
    end

    info = merge(info, (
        duals_available = duals && has_duals(model),
        dual_resolve_status,
        mip_resolved,
        resolved_objective_value,
        resolved_relative_gap,
        resolved_solve_time_s))
    if duals
        println(progress_io, info.duals_available ? "   ℹ  Dual information available." :
            "   ❗  Dual information unavailable.")
    end

    return info

end
