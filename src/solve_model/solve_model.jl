function solve_model!(model::Model,b_relax_integrality::Bool)
    
    # relax integrality {true,false}
    if b_relax_integrality==true
        relax_integrality(model)
    end

    optimize!(model)    # solve model
    s = termination_status(model)
    println("  \u2139  ", s)

    # Check infeasibility if not optimal
    if s != MOI.OPTIMAL
        CSV.write(joinpath(p,"status.csv"),DataFrame(s); header=false)
        compute_conflict!(model)
        if get_attribute(model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
            iis_model, reference_map = copy_conflict(model)
            print(iis_model)
        end
        error("\u2757  Model is not optimal. Check input data.\n")
    end

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