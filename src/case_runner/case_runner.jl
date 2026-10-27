"""
run_decarb!(path; optimizer=HiGHS.Optimizer, attributes=Dict(),
            mip_gap=1e-4, time_limit=300.0, duals=true, verbose=true,
            relax_integrality=false, return_result=false)

Executes the complete DECARB optimization workflow for a distributed energy system

The function orchestrates all stages of the DECARB model: loading input data, configuring
the optimization model with the specified solver, building the model constraints and
objective function, solving the optimization problem, reading solution outputs, and
writing results to CSV files.

inputs:
path                    string path to working directory containing in/ subdirectory
mip_gap                 MIP gap for the optimization problem
time_limit              time limit for the optimization problem
optimizer               JuMP optimizer constructor, factory, or optimizer_with_attributes
attributes              raw solver names or MOI optimizer attributes; override defaults
duals                   fix integer decisions and re-solve to recover dual information
verbose                 print progress and enable solver output (default: true)
relax_integrality       solve the continuous relaxation instead of the integer model
return_result           return the solved model and input dictionaries (default: false)

returns nothing by default, or a named tuple with the solved model and input dictionaries

"""

function run_decarb!(path::AbstractString;
    optimizer=HiGHS.Optimizer,
    attributes::AbstractDict=Dict{String,Any}(),
    mip_gap::Real=1e-4,
    time_limit::Real=300.0,
    duals::Bool=true,
    verbose::Bool=true,
    relax_integrality::Bool=false,
    return_result::Bool=false)

    io = verbose ? stdout : devnull

    title = " DECARB model v$(pkgversion(DECARB)) "
    println(io, "\u250F" * "\u2501"^length(title) * "\u2513")
    println(io, "\u2503" * title * "\u2503")
    println(io, "\u2517" * "\u2501"^length(title) * "\u251B\n")

    a0 = time()	    # start timer

    println(io, "\u23E9 loading inputs")

    # load inputs file
    path2in = joinpath(path,"in")
    cfg = load_cfg(path2in)
    sp = load_sp(path2in,cfg)
    bdg = load_bdgii(path2in,load_bdgi(path2in))
    tm = load_tm(path2in,cfg,bdg)
    pv = load_pv(path2in,tm,bdg)
    chp,abp,hvac,wh,ev = load_extended_catalog(path2in,tm,sp,bdg)
    wind = load_wind(path2in,tm)
    bess = load_bess(path2in)
    topo = load_topo(path2in,chp,abp)

    # validate only after all input loading is complete
    validate_inputs(cfg,sp,bdg,tm,chp,abp,hvac,wh,pv,wind,bess,ev,topo; io)

    cfg["b_inv"]==false  ? println(io, "   \u2139  investments prevented") : println(io, "   \u2139  investments allowed")

    cfg["b_temp"]==false ? println(io, "   \u2139  temperature disabled")  : println(io, "   \u2139  temperature enabled")

    a1 = time()		# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a1-a0; digits=2), " seconds\n")

    println(io, "\u23E9 configuring model")

    # Define the optimization model and solver
    model = Model(optimizer)
    configure_solver!(model, mip_gap, time_limit)
    verbose || set_silent(model)
    for (key, value) in attributes
        set_optimizer_attribute(model, key, value)
    end

    a2 = time()			# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a2-a1; digits=2), " seconds\n")

    println(io, "\u23E9 building model")

    println(io, "   \u23E9 defining heat connections")
    heat_connections!(model,topo,tm,chp,abp)
    a3 = time()			# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a3-a2; digits=2), " seconds\n")

    println(io, "   \u23E9 defining temperature variables")
    temperature_variables!(model,tm)
    a4 = time()			# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a4-a3; digits=2), " seconds\n")

    println(io, "   \u23E9 defining CHP unit model")
    chp_units!(model,cfg,tm,bdg,topo,sp,chp,abp)
    a5 = time()			# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a5-a4; digits=2), " seconds\n")

    println(io, "   \u23E9 defining HVAC unit model")
    hvac_units!(model,cfg,tm,bdg,sp,hvac)
    a6 = time()			# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a6-a5; digits=2), " seconds\n")

    println(io, "   \u23E9 defining absorption chiller model")
    abp_chillers!(model,cfg,tm,bdg,topo,sp,chp,abp)
    a7 = time()			# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a7-a6; digits=2), " seconds\n")

    println(io, "   \u23E9 defining water heater model")
    water_heaters!(model,cfg,tm,bdg,sp,wh)
    a8 = time()			# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a8-a7; digits=2), " seconds\n")

    println(io, "   \u23E9 defining PV module model")
    pv_panels!(model,cfg,tm,bdg,sp,pv)
    a9 = time()			# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a9-a8; digits=2), " seconds\n")

    println(io, "   \u23E9 defining wind turbine model")
    wind_turbines!(model,cfg,tm,bdg,sp,wind)
    a10 = time()			# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a10-a9; digits=2), " seconds\n")

    println(io, "   \u23E9 defining BESS module model")
    bess_modules!(model,cfg,tm,bdg,sp,bess)
    a11 = time()			# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a11-a10; digits=2), " seconds\n")

    println(io, "   \u23E9 defining EV module model")
    electric_vehicles!(model,cfg,tm,ev)
    a12 = time()		# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a12-a11; digits=2), " seconds\n")

    println(io, "   \u23E9 defining electrical model")
    electric_load!(model,cfg,tm,chp,hvac,wh,pv,wind,bess,ev; io)
    a13 = time()		# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a13-a12; digits=2), " seconds\n")

    println(io, "   \u23E9 defining thermal model")
    thermal_load!(model,cfg,tm,bdg,topo,chp,hvac,abp,wh; io)
    a14 = time()		# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a14-a13; digits=2), " seconds\n")

    println(io, "   \u23E9 defining objective function")
    objective_function!(model,cfg,tm,chp,abp,hvac,wh,pv,bess,ev,wind)
    a15 = time()		# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a15-a14; digits=2), " seconds\n")

    println(io, "\u23E9 solving model")

    mkpath(joinpath(path,"out"))        # create output directory if it does not exist

    # remove stale DECARB outputs from previous runs; other user files are kept
    for f in ("eq1.csv","eq2.csv","ts.csv","balance.csv","iis_constraints.txt","solve_info.csv")
        rm(joinpath(path,"out",f); force=true)
    end

    mip = solve_model!(path,model,relax_integrality; duals, verbose)
    solve_info = (;
        decarb_version = string(pkgversion(DECARB)),
        git_commit = git_commit(),
        julia_version = string(VERSION),
        solver_name = mip.solver_name,
        solver_version = mip.solver_version,
        termination_status = mip.termination_status,
        primal_status = mip.primal_status,
        objective_value = mip.objective_value,
        objective_bound = mip.objective_bound,
        relative_gap = mip.relative_gap,
        solve_time_s = mip.solve_time_s,
        mip_gap,
        time_limit,
        relax_integrality,
        duals,
        duals_available = mip.duals_available,
        threads = mip.threads,
        utc_timestamp = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z")

    a16 = time()		# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a16-a15; digits=2), " seconds\n")

    println(io, "\u23E9 reading outputs")

    # read model outputs
    df_chp = read_thermal(model,sp,"chp",chp,tm["Date"],tm["IW"])
    df_hvac = read_thermal(model,sp,"hvac",hvac,tm["Date"],tm["IW"])
    df_abp = read_thermal(model,sp,"abp",abp,tm["Date"],tm["IW"])
    df_wh = read_thermal(model,sp,"wh",wh,tm["Date"],tm["IW"])
    df_pv = read_der(model,sp,"pv",pv,tm["Date"],tm["IW"])
    df_bess = read_der(model,sp,"bess",bess,tm["Date"],tm["IW"])
    df_wind = read_der(model,sp,"wind",wind,tm["Date"],tm["IW"])
    df_elec,balance = read_electric(model,tm,chp["N"],hvac["N"],wh["N"],pv["N"],
        bess["N"],ev["N"],wind["N"],bess["mx"],bess["zmx0"],pv["zmx0"],wind["zmx0"],
        ev["mx_eff"],wh["fuel"])
    df_fuel = read_fuel(model,tm,chp["N"],abp["N"],wh["N"])
    df_indoor = read_indoor(model,tm,chp["N"],abp["N"],hvac["N"],wh["N"],wh["tank_eff"],
        topo["chp_bdg"])
    df_dual,df_econ = read_econ(model,tm; duals=solve_info.duals_available)

    a17 = time()		# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a17-a16; digits=2), " seconds\n")

    println(io, "\u23E9 writing outputs")

    write_outputs(path,tm["Date"],df_chp,df_hvac,df_abp,df_wh,df_pv,df_bess,
        df_wind,df_elec,balance,df_fuel,df_indoor,df_dual,df_econ; io)
    write_solve_info(path,solve_info)

    a18 = time()		# elapsed time
    println(io, "   \u231B elapsed time ... ", round(a18-a17; digits=2), " seconds\n")

    result = (; model, cfg, tm, bdg, sp, chp, abp, hvac, wh, pv, wind, bess, ev, topo,
        solve_info)
    return return_result ? result : nothing
end

function git_commit()
    try
        return strip(readchomp(pipeline(Cmd(`git rev-parse HEAD`; dir=pkgdir(DECARB)); stderr=devnull)))
    catch
        return "unknown"
    end
end
