"""
run_decarb!(path::AbstractString,i_solver::Int64,b_relax_integrality::Bool)

Executes the complete DECARB optimization workflow for a distributed energy system

The function orchestrates all stages of the DECARB model: loading input data, configuring
the optimization model with the specified solver, building the model constraints and 
objective function, solving the optimization problem, reading solution outputs, and
writing results to CSV files.

inputs:
path                    string path to working directory containing in/ subdirectory
i_solver                solver selection flag: 1=Gurobi, 2=HiGHS
mip_gap                 MIP gap for the optimization problem
time_limit              time limit for the optimization problem
b_relax_integrality     boolean flag to relax integrality constraints on integer variables

"""

function run_decarb!(path::AbstractString,mip_gap::Float64,time_limit::Float64,i_solver::Int64,b_relax_integrality::Bool)

    title = " DECARB model v$(pkgversion(DECARB)) "
    println("\u250F" * "\u2501"^length(title) * "\u2513")
    println("\u2503" * title * "\u2503")
    println("\u2517" * "\u2501"^length(title) * "\u251B\n")

    a0 = time()	    # start timer

    println("\u23E9 loading inputs")
    
    # load inputs file
    path2in = joinpath(path,"in")
    cfg = load_cfg(path2in)
    sp = load_sp(path2in,cfg)
    bdg = load_bdgi(path2in)
    bdg = load_bdgii(path2in,bdg)
    tm = load_tm(path2in,cfg,bdg)
    pv = load_pv(path2in,tm,bdg)
    chp,abp,hvac,wh,ev = load_extended_catalog(path2in,tm,sp,bdg)
    wind = load_wind(path2in,tm)
    bess = load_bess(path2in)
    topo = load_topo(path2in,chp,abp)

    cfg["b_inv"]==false  ? println("   \u2139  investments prevented") : println("   \u2139  investments allowed")

    cfg["b_temp"]==false ? println("   \u2139  temperature disabled")  : println("   \u2139  temperature enabled")

    a1 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a1-a0; digits=2), " seconds\n")

    println("\u23E9 configuring model")

    # Define the optimization model and solver
    model = Model()

    # load configuration options for selected solver
    if i_solver==1
        set_optimizer(model,Gurobi.Optimizer)
        configure_gurobi(model,mip_gap,time_limit)
        println("   \u2139  Gurobi called satisfactorily")
    elseif i_solver==2
        set_optimizer(model,HiGHS.Optimizer)
        configure_highs(model,mip_gap,time_limit)
        println("   \u2139  HiGHS called satisfactorily")
    else
        error("❗  Invalid solver selection. Choose 1=Gurobi or 2=HiGHS.")
    end

    a2 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a2-a1; digits=2), " seconds\n")

    println("\u23E9 building model")

    println("   \u23E9 defining heat connections")
    heat_connections!(model,topo,tm,chp,abp)
    a3 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a3-a2; digits=2), " seconds\n")

    println("   \u23E9 defining temperature variables")
    temperature_variables!(model,tm)
    a4 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a4-a3; digits=2), " seconds\n")

    println("   \u23E9 defining CHP unit model")
    chp_units!(model,cfg,tm,bdg,topo,sp,chp,abp)
    a5 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a5-a4; digits=2), " seconds\n")
    
    println("   \u23E9 defining HVAC unit model")
    hvac_units!(model,cfg,tm,bdg,sp,hvac)
    a6 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a6-a5; digits=2), " seconds\n")

    println("   \u23E9 defining absorption chiller model")
    abp_chillers!(model,cfg,tm,bdg,topo,sp,chp,abp)
    a7 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a7-a6; digits=2), " seconds\n")

    println("   \u23E9 defining water heater model")
    water_heaters!(model,cfg,tm,bdg,sp,wh)
    a8 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a8-a7; digits=2), " seconds\n")

    println("   \u23E9 defining PV module model")
    pv_panels!(model,cfg,tm,bdg,sp,pv)
    a9 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a9-a8; digits=2), " seconds\n")

    println("   \u23E9 defining wind turbine model")
    wind_turbines!(model,cfg,tm,bdg,sp,wind)
    a10 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a10-a9; digits=2), " seconds\n")

    println("   \u23E9 defining BESS module model")
    bess_modules!(model,cfg,tm,bdg,sp,bess)
    a11 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a11-a10; digits=2), " seconds\n")

    println("   \u23E9 defining EV module model")
    electric_vehicles!(model,cfg,tm,ev)
    a12 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a12-a11; digits=2), " seconds\n")

    println("   \u23E9 defining electrical model")
    electric_load!(model,cfg,tm,chp,hvac,wh,pv,wind,bess,ev)
    a13 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a13-a12; digits=2), " seconds\n")

    println("   \u23E9 defining thermal model")
    thermal_load!(model,cfg,tm,bdg,topo,chp,hvac,abp,wh)
    a14 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a14-a13; digits=2), " seconds\n")

    println("   \u23E9 defining objective function")
    objective_function!(model,cfg,tm,chp,abp,hvac,wh,pv,bess,ev,wind)
    a15 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a15-a14; digits=2), " seconds\n")

    println("\u23E9 solving model")

    solve_model!(path,model,b_relax_integrality)

    a16 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a16-a15; digits=2), " seconds\n")

    println("\u23E9 reading outputs")

    # read model outputs
    df_chp = read_thermal(model,sp,"chp",chp,tm["Date"],tm["IW"])
    df_hvac = read_thermal(model,sp,"hvac",hvac,tm["Date"],tm["IW"])
    df_abp = read_thermal(model,sp,"abp",abp,tm["Date"],tm["IW"])
    df_wh = read_thermal(model,sp,"wh",wh,tm["Date"],tm["IW"])
    df_pv = read_der(model,sp,cfg,"pv",pv,tm["Date"],tm["IW"])
    df_bess = read_der(model,sp,cfg,"bess",bess,tm["Date"],tm["IW"])
    df_wind = read_der(model,sp,cfg,"wind",wind,tm["Date"],tm["IW"])
    df_elec,balance = read_electric(model,tm,chp["N"],hvac["N"],wh["N"],pv["N"],
        bess["N"],ev["N"],wind["N"],bess["mx"],ev["mx"],collect(wh["fuel"]))
    df_fuel = read_fuel(model,tm,chp["N"],abp["N"],wh["N"],topo["N"])
    df_indoor = read_indoor(model,tm,chp["N"],abp["N"],hvac["N"],wh["N"],wh["tank"],
        topo["chp_bdg"])
    df_dual,df_econ = read_econ(model,tm)

    a17 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a17-a16; digits=2), " seconds\n")

    println("\u23E9 writing outputs")

    mkpath(joinpath(path,"out"))        # create output directory if it does not exist
    write_outputs(path,tm["Date"],df_chp,df_hvac,df_abp,df_wh,df_pv,df_bess,
        df_wind,df_elec,balance,df_fuel,df_indoor,df_dual,df_econ)

    a18 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a18-a17; digits=2), " seconds\n")

end
