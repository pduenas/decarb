"""
solve_model!(path::AbstractString,model::Model,b_relax_integrality::Bool)

Optimizes the model, checks feasibility, and writes an IIS report if infeasible. If the
model is feasible and b_relax_integrality is false, integer/binary variables are fixed
at their solution values and the model is re-solved relaxed to obtain dual information

inputs:
path                    string path to working directory
model                   optimization model object
b_relax_integrality     boolean flag to relax integrality constraints on integer variables

"""

function solve_model!(path::AbstractString,model::Model,b_relax_integrality::Bool)
    
    # relax integrality {true,false}
    if b_relax_integrality==true
        relax_integrality(model)
    end

    optimize!(model)    # solve model
    s = termination_status(model)

    if !is_solved_and_feasible(model; allow_local = true)
        try
            compute_conflict!(model)
            if get_attribute(model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
                println("\n❗ IIS detected — infeasible constraints:\n")
                open(joinpath(path,"out","iis_constraints.txt"), "w") do io
                    for (F, S) in list_of_constraint_types(model)
                        F == VariableRef && continue
                        for con in all_constraints(model, F, S)
                            cs = get_attribute(con, MOI.ConstraintConflictStatus())
                            if cs == MOI.IN_CONFLICT
                                label = name(con) == "" ? string(con) : name(con)
                                println(io, "[IN_CONFLICT] ", label)
                                println("  ❌ ", label)
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
    @info "termination = $(s), gap = $(round(gap*100, digits=1))%"

    # relax integrality to get dual information
    if b_relax_integrality==false
        # save values of all variables
        vALL = all_variables(model)
        v_L = value.(vALL)
        # locate and fix integer variables
        vINT = is_integer.(vALL)
        unset_integer.(vALL[vINT])
        fix.(vALL[vINT],v_L[vINT]; force=true)
        # locate and fix binary variables
        vBIN = is_binary.(vALL)
        unset_binary.(vALL[vBIN])
        fix.(vALL[vBIN],v_L[vBIN]; force=true)
        # solve relaxed model
        optimize!(model)
    end

    has_duals(model)==true ? println("   \u2139  Dual information available.") : 
        println("   \u2757  Dual information unavailable.")

end