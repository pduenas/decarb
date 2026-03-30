
function run_decarb!(path::AbstractString,i_solver::Int64,b_relax_integrality::Bool)

    println("\u250F\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2513")
    println("\u2503 DECARB model v1.0 \u2503")
    println("\u2517\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u251B\n")

    a0 = time()	    # start timer

    println("\u23E9 loading inputs")
    write_status(path,1)
    
    # load inputs file
    path2in = joinpath(path,"in")
    in = load_in(path2in)
    sp = load_sp(path2in,in)
    bdg = load_bdgi(path2in)
    bdg = load_bdgii(path2in,bdg)
    tm = load_tm(path2in,in,bdg)
    pv = load_pv(path2in,tm,bdg)
    chp,abp,hvac,wh,ev = load_extended_catalog(path2in,tm,sp,bdg)
    wind = load_wind(path2in,tm)
    bess = load_bess(path2in)
    topo = load_topo(path2in,chp,abp)

    in["b_inv"]==false  ? println("   \u2139  investments prevented") : println("   \u2139  investments allowed")

    in["b_temp"]==false ? println("   \u2139  temperature disabled")  : println("   \u2139  temperature enabled")

    a1 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a1-a0; digits=2), " seconds\n")

    println("\u23E9 configuring model")
    write_status(path,2)

    # Define the optimization model and solver
    model = Model()

    # load configuration options for selected solver
    if i_solver==1
        set_optimizer(model,Gurobi.Optimizer)
        configure_gurobi(model)
        println("   \u2139  Gurobi called satisfactorily")
    elseif i_solver==2
        set_optimizer(model,CPLEX.Optimizer)
        configure_cplex(model)
        println("   \u2139  CPLEX called satisfactorily")
    end

    a2 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a2-a1; digits=2), " seconds\n")

    println("\u23E9 building model")
    write_status(path,3)

    println("   \u23E9 defining heat connections")
    heat_connections!(model,topo,tm,chp,abp)
    a3 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a3-a2; digits=2), " seconds\n")

    println("   \u23E9 defining temperature variables")
    temperature_variables!(model,tm)
    a4 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a4-a3; digits=2), " seconds\n")

    println("   \u23E9 defining CHP unit model")
    chp_units!(model,in,tm,bdg,topo,sp,chp,abp)
    a5 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a5-a4; digits=2), " seconds\n")
    
    println("   \u23E9 defining HVAC unit model")
    hvac_units!(model,in,tm,bdg,sp,hvac)
    a6 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a6-a5; digits=2), " seconds\n")

    println("   \u23E9 defining absorption chiller model")
    abs_chillers!(model,in,tm,bdg,topo,sp,chp,abp)
    a7 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a7-a6; digits=2), " seconds\n")

    println("   \u23E9 defining water heater model")
    water_heaters!(model,in,tm,bdg,sp,wh)
    a8 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a8-a7; digits=2), " seconds\n")

    println("   \u23E9 defining PV module model")
    pv_panels!(model,in,tm,bdg,sp,pv)
    a9 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a9-a8; digits=2), " seconds\n")

    println("   \u23E9 defining wind turbine model")
    wind_turbines!(model,in,tm,bdg,sp,wind)
    a10 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a10-a9; digits=2), " seconds\n")

    println("   \u23E9 defining BESS module model")
    bess_modules!(model,in,tm,bdg,sp,bess)
    a11 = time()			# elapsed time
    println("   \u231B elapsed time ... ", round(a11-a10; digits=2), " seconds\n")

    println("   \u23E9 defining EV module model")
    electric_vehicles!(model,in,tm,ev)
    a12 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a12-a11; digits=2), " seconds\n")

    println("   \u23E9 defining electrical model")
    electric_load!(model,in,tm,chp,hvac,wh,pv,wind,bess,ev)
    a13 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a13-a12; digits=2), " seconds\n")

    println("   \u23E9 defining thermal model")
    thermal_load!(model,in,tm,bdg,topo,chp,hvac,abp,wh)
    a14 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a14-a13; digits=2), " seconds\n")

    println("   \u23E9 defining objective function")
    objective_function!(model,in,tm,chp,abp,hvac,wh,pv,bess,ev,wind)
    a15 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a15-a14; digits=2), " seconds\n")

    println("\u23E9 solving model")
    write_status(path,4)

    solve_model!(path,model,b_relax_integrality)

    a16 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a16-a15; digits=2), " seconds\n")

    println("\u23E9 reading outputs")
    write_status(path,5)

    # read model outputs
    df_chp = read_thermal(model,sp,"chp",chp)
    df_hvac = read_thermal(model,sp,"hvac",hvac)
    df_abs = read_thermal(model,sp,"abp",abp)
    df_wh = read_thermal(model,sp,"wh",wh)
    df_pv,df_pviw = read_der(model,sp,in,"pv",pv)
    df_bess,df_bessiw = read_der(model,sp,in,"bess",bess)
    df_wind,df_windiw = read_der(model,sp,in,"wind",wind)
    df_elec,balance = read_electric(model,tm,chp["N"],hvac["N"],wh["N"],pv["N"],
        bess["N"],ev["N"],wind["N"],bess["mx"],ev["mx"],collect(wh["fuel"]))
    df_fuel = read_fuel(model,tm,chp["N"],abp["N"],wh["N"],topo["N"])
    df_indoor = read_indoor(model,tm,chp["N"],abp["N"],hvac["N"],wh["N"],wh["tank"],
        topo["chp_bdg"])
    df_dual,df_econ = read_econ(model,tm)

    a17 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a17-a16; digits=2), " seconds\n")

    println("\u23E9 writing outputs")
    write_status(path,6)

    write_outputs(path,tm["Date"],tm["IW"],df_chp,df_hvac,df_abs,df_wh,df_pv,df_pviw,df_bess,
        df_bessiw,df_wind,df_windiw,df_elec,balance,df_fuel,df_indoor,df_dual,df_econ)

    a18 = time()		# elapsed time
    println("   \u231B elapsed time ... ", round(a18-a17; digits=2), " seconds\n")

end



"""
write_status(path::AbstractString,opt::UInt8)

updates the simulation status

input:
path    string path to working directory
opt     integer option with simulation status
"""
function write_status(path::AbstractString,opt::Int64)

    st = open(joinpath(path,"status.txt"),"w")
    if opt==1
        write(st,"reading inputs")
    elseif opt==2
        write(st,"configuring model")
    elseif opt==3
        write(st,"building model")
    elseif opt==4
        write(st,"solving model")
    elseif opt==5
        write(st,"reading outputs")
    elseif opt==5
        write(st,"writing outputs")
    end
    close(st)

end
