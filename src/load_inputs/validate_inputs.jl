"""
    validate_inputs(cfg, sp, bdg, tm, chp, abp, hvac, wh, pv, wind, bess, ev, topo;
                    io=stdout)

Validate already-loaded input dictionaries before model construction. This
function does not read files, load inputs, or mutate the supplied data. It
returns nothing on success and reports input errors and temperature warnings.
"""
function validate_inputs(cfg::Dict,sp::Dict,bdg::Dict,tm::Dict,chp::Dict,abp::Dict,
    hvac::Dict,wh::Dict,pv::Dict,wind::Dict,bess::Dict,ev::Dict,topo::Dict; io::IO=stdout)
    err = String[]       # fatal
    wrn = String[]       # accepted but suspect

    validate_numeric_missing!(err, cfg, bdg, tm, chp, abp, hvac, wh, pv, wind,
        bess, ev, topo)
    isempty(err) || throw(ArgumentError(join(err, "\n")))

    # ---- loaded thermal values -------------------------------------------
    for key in ("Bk1", "Bk2", "Bk3", "Btime")
        if !haskey(bdg, key)
            push!(err, "bdg_ii.p$key is required")
            continue
        end
        value = bdg[key]
        finite = value isa Real && isfinite(value)
        if key == "Bk1"
            finite && 0 <= value <= 1 ||
                push!(err, "bdg_ii.pBk1 must be in [0,1] for timestep resampling")
        elseif key == "Btime"
            finite && value > 0 ||
                push!(err, "bdg_ii.pBtime must be a positive, finite number")
        else
            finite || push!(err, "bdg_ii.p$key must be finite")
        end
    end
    isempty(err) || throw(ArgumentError(join(err, "\n")))

    # ---- equipment selections in loaded catalogs --------------------------
    for (prefix, catalog) in (("CHP", chp), ("ABP", abp), ("HVAC", hvac),
                              ("WH", wh), ("EV", ev))
        names = catalog["ty"]
        for selected in sp["$(prefix)0"]
            selected == "0" && continue
            any(n -> n == selected || startswith(n, selected * "-"), names) && continue
            push!(err,"equipment \"$selected\" is selected in sp.csv but absent from the " *
                  "catalog. Check the spelling, and that its capacity is non-zero " *
                  "(zero-capacity rows are discarded on load). " *
                  "Available: $(join(unique(names), ", "))")
        end
    end

    # ---- counts (read as Int, so negatives are rejected here) -------------
    for k in ("Bchp","Bhvac","Babp","Bwh","Btck","Bwind","Bbess")
        haskey(bdg,k) && bdg[k] < 0 && push!(err,"bdg_i.p$(k) must be non-negative")
    end
    for (k,v) in sp
        endswith(k,"z0") && any(<(0), v) &&
            push!(err,"sp.csv p$(k) must be non-negative")
    end

    # ---- configuration -----------------------------------------------------
    for k in ("BESSsoc0","BESSsocf","EVsoc0","EVmnsoc","WHsto0","WHstof")
        0 <= cfg[k] <= 1 || push!(err,"cfg.$(k) = $(cfg[k]); expected a fraction in [0,1]")
    end
    0 <= cfg["IR"] < 1 || push!(err,"cfg.IR = $(cfg["IR"]); expected an annual rate in [0,1)")
    cfg["QmxBuy"]  > 0 || push!(err,"cfg.QmxBuy must be positive")
    cfg["QmxSell"] >= 0 || push!(err,"cfg.QmxSell must be non-negative")
    cfg["EVdriver"] in (-1,0,1) ||
        push!(err,"cfg.EVdriver = $(cfg["EVdriver"]); expected -1, 0 or 1")
    for k in ("NSEcost","NSTcost","NSHWcost","NSEVcost","Gco2","Lco2")
        cfg[k] >= 0 || push!(err,"cfg.$(k) must be non-negative")
    end

    # ---- time series -------------------------------------------------------
    tm["P"] > 0 || push!(err,"tm.csv contains no periods")
    all(t -> isfinite(t) && t > 0, tm["TM"]) ||
        push!(err,"tm.csv timestamps are not strictly increasing")
    issorted(tm["Putc"]) || push!(err,"tm.csv UTC timestamps are not sorted")
    isapprox(tm["H"], sum(tm["TM"])) ||
        push!(err,"tm.csv elapsed horizon does not match the sum of period durations")
    all(tm["Tmn"] .<= tm["Tmx"]) ||
        push!(err,"tm.csv has Tmn > Tmx in at least one period")
    all(tm["QcostBuy"] .>= 0) || push!(err,"tm.csv QcostBuy must be non-negative")
    all(tm["HWdem"]   .>= 0) || push!(err,"tm.csv HWdem must be non-negative")
    all(tm["Wms"]     .>= 0) || push!(err,"tm.csv Wms must be non-negative")
    any(tm["QcostSell"] .> tm["QcostBuy"]) &&
        push!(wrn,"sale price exceeds purchase price in some periods; " *
                  "unbounded arbitrage is prevented only by the buy/sell binary")

    # ---- thermal coefficients and forcing ---------------------------------
    isfinite(cfg["Tin0"]) || push!(err,"cfg.Tin0 must be finite")
    for key in ("Tout", "Tmn", "Tmx", "Q_R", "Qihg_P", "Qihg_L", "Qihg_E")
        all(isfinite, tm[key]) || push!(err,"tm.$(key) must contain finite values")
    end

    # ---- comfort reachability ---------------------------------------------
    Tup = maximum([tm["Tout"];tm["Tmx"]]; init=-Inf)
    Tlo = minimum([tm["Tout"];tm["Tmn"]]; init=Inf)
    cfg["Tin0"] <= Tup && cfg["Tin0"] >= Tlo ||
        push!(wrn,"cfg.Tin0 = $(cfg["Tin0"]) lies outside the modeled indoor " *
                  "bounds [$(Tlo), $(Tup)]; the first period's thermal balance " *
                  "determines whether it can return within those bounds")
    if cfg["NSTcost"] > 0 && all(tm["Tmx"] .>= Tup)
        push!(wrn,"upward temperature discomfort is priced but structurally " *
                  "impossible: max(Tmx) equals the indoor upper bound")
    end

    # ---- conversion factors and divide-by-zero guards ---------------------
    # fcf is a conversion efficiency, not a heat-rate multiplier. A zero is
    # accepted only for equipment that does not consume fuel directly.
    for (d,name) in ((chp,"chp"), (abp,"abp"), (wh,"wh"))
        for i in 1:d["N"]
            fcf = d["fcf"][i]
            valid = d["fuel"][i] == "0" ? 0 <= fcf <= 1 : 0 < fcf <= 1
            valid || push!(err,"$(name).csv row $(i) ($(d["ty"][i])): " *
                "fcf must be in (0,1] for fuel-consuming equipment, or " *
                "[0,1] when direct fuel is not used")
        end
    end
    for (d,name,keys) in ((abp,"abp",("ac",)),
                          (wh,"wh",("eff",)), (pv,"pv",("ar",)),
                          (bess,"bess",("mx","effu","effd")),
                          (ev,"ev",("mx","effu","effd")))
        for k in keys, i in 1:d["N"]
            if k in ("effu", "effd")
                value = d[k][i]
                value isa Real && isfinite(value) && 0 < value <= 1 ||
                    push!(err,"$(name).csv row $(i) ($(d["ty"][i])): " *
                        "$(k) must be a finite number in (0,1]")
                continue
            end
            d[k][i] > 0 && continue
            push!(err,"$(name).csv row $(i) ($(d["ty"][i])): $(k) must be positive")
        end
    end
    for i in 1:wind["N"]
        vmn, vmx = wind["vmn"][i], wind["vmx"][i]
        vmn isa Real && vmx isa Real && isfinite(vmn) && isfinite(vmx) &&
            0 <= vmn < vmx || push!(err,"wind.csv row $(i) ($(wind["ty"][i])): " *
                "vmn and vmx must be finite numbers with 0 <= vmn < vmx")
    end
    all(1 .<= pv["tech"] .<= 3) ||
        push!(err,"pv.csv tech must be 1, 2 or 3")
    bdg["Btck"] in (0,1,2) ||
        push!(err,"bdg_i.csv pBtck must be 0 (fixed), 1 (single-axis) or 2 (dual-axis)")

    # ---- installation space vs. existing stock ----------------------------
    for (pfx,space) in (("CHP","Bchp"),("ABP","Babp"),("HVAC","Bhvac"),("WH","Bwh"))
        for (i,n) in enumerate(sp["$(pfx)z0"])
            n <= bdg[space] && continue
            push!(err,"sp.csv declares $(n) existing $(pfx) units but " *
                      "bdg_i.csv p$(space) allows only $(bdg[space])")
        end
    end
    for i in 1:pv["N"]
        pv["zmx"][i] >= 1 && continue
        push!(wrn,"pv.csv $(pv["ty"][i]): roof area $(bdg["Bpv"]) m2 fits zero panels")
    end

    # ---- positive lifetimes ----------------------------------------------
    for (d,name) in ((chp,"chp"),(abp,"abp"),(hvac,"hvac"),(wh,"wh"),
                    (pv,"pv"),(wind,"wind"),(bess,"bess"))
        for i in 1:d["N"]
            life = d["life"][i]
            life isa Real && isfinite(life) && life > 0 ||
                push!(err,"$(name).csv row $(i) ($(d["ty"][i])): life must be a positive, finite number " *
                        "(it is the annuity denominator, even for deselected rows)")
            d["inv"][i] >= 0 || push!(err,"$(name).csv row $(i): inv must be non-negative")
            d["fom"][i] >= 0 || push!(err,"$(name).csv row $(i): fom must be non-negative")
            haskey(d,"vom") && (d["vom"][i] >= 0 ||
                push!(err,"$(name).csv row $(i): vom must be non-negative"))
        end
    end

    # ---- DHW served by something -----------------------------------------
    if any(tm["HWdem"] .> 0) && wh["N"] == 0 &&
       all(iszero, topo["chp_bdg"][:,2]) && cfg["NSHWcost"] == 0
        push!(err,"hot water is demanded but no water heater or CHP hot-water " *
                  "link exists, and NSHWcost = 0 forbids unserved demand")
    end

    # ---- topology ---------------------------------------------------------
    for i in 1:topo["N"]
        eff = topo["eff"][i]
        eff isa Real && isfinite(eff) && 0 <= eff <= 1 ||
            push!(err,"topo.csv row $(i): eff must be a finite number in [0,1]")
        topo["in"][i] in ("hot air","hot water","cooling") && continue
        push!(wrn,"topo.csv row $(i): unrecognised product \"$(topo["in"][i])\"; " *
                  "the link will be ignored")
    end

    validate_extended_inputs!(err, wrn, cfg, sp, bdg, tm, chp, pv, wind, topo)

    # ---- reserved inputs, accepted but inactive in v1.0 -------------------
    reserved = String[]
    cfg["ST"]      != 0 && push!(reserved,"cfg.ST")
    cfg["CO2mx"]   != 0 && push!(reserved,"cfg.CO2mx")
    cfg["CO2cost"] != 0 && push!(reserved,"cfg.CO2cost")
    cfg["CO2ton"]  != 0 && push!(reserved,"cfg.CO2ton")
    bdg["Bevmx"]   != 0 && push!(reserved,"bdg_i.Bevmx")
    any(!iszero,chp["sup"])   && push!(reserved,"chp.sup")
    any(!iszero,chp["tank"])  && push!(reserved,"chp.tank")
    any(!iszero,pv["loss"])   && push!(reserved,"pv.loss")
    any(!iszero,pv["fail"])   && push!(reserved,"pv.fail")
    any(!iszero,pv["lc"])     && push!(reserved,"pv.lc")
    any(!iszero,wind["fail"]) && push!(reserved,"wind.fail")
    any(!iszero,wind["lc"])   && push!(reserved,"wind.lc")
    any(!iszero,bess["lc"])   && push!(reserved,"bess.lc")
    any(!iszero,ev["cold"])   && push!(reserved,"ev.cold")
    isempty(reserved) ||
        push!(wrn,"the following inputs are read but not active in DECARB v1.0 " *
                  "and have no effect: " * join(reserved,", "))

    # ---- temperature diagnostics -----------------------------------------
    # Run only after structural checks pass. These optimistic thermal checks
    # provide warnings, not a substitute for solving the coupled energy model.
    if isempty(err)
        report = check_temperature(cfg, tm, bdg, sp, chp, abp, hvac, topo)
        free_min, free_max = extrema(report.free_temperature)
        if free_min < Tlo - 1e-6 || free_max > Tup + 1e-6
            push!(wrn,"without heating/cooling, indoor temperature ranges from " *
                "$(round(free_min; digits=2)) to $(round(free_max; digits=2)) C, " *
                "outside modeled bounds [$(Tlo), $(Tup)]. Solar/internal gains " *
                "mean outdoor extrema are not physical indoor limits; " *
                "active conditioning may be required even when comfort is disabled")
        end
        if report.conflict !== nothing
            c = report.conflict
            push!(wrn,"temperature precheck: insufficient $(c.direction) at " *
                "$(tm["Date"][c.period]) (period $(c.period)); the optimistic " *
                "reachable indoor range [$(round(c.lower; digits=3)), " *
                "$(round(c.upper; digits=3))] C does not intersect required " *
                "bounds [$(c.required_lower), $(c.required_upper)] C, even " *
                "with all eligible thermal equipment available. Check capacity, " *
                "comfort limits, and thermal calibration")
        end
        report.skipped && push!(wrn,"temperature capacity precheck skipped: " *
            "it requires nonnegative thermal response, capacities, and heat links")
    end

    # ---- report ----------------------------------------------------------
    for w in wrn
        println(io, "   \u26A0  ", w)
    end
    if !isempty(err)
        throw(ArgumentError("❗  input validation failed:\n" *
              join(["    • " * e for e in err], "\n") * "\n"))
    end
    println(io, "   \u2139  inputs validated (", length(wrn), " warning(s))")

    return nothing
end

# Strip a trailing "-<number>" instance suffix added by extend_catalog.
catalog_base_name(name::AbstractString) =
    (m = match(r"^(.*)-\d+$", name); m === nothing ? String(name) : String(m.captures[1]))

function validate_extended_inputs!(err, wrn, cfg, sp, bdg, tm, chp, pv, wind, topo)
    # ---- time series ------------------------------------------------------
    nQ = cfg["QmxTM"]
    all(q -> 0 <= q <= nQ, tm["Qmx"]) ||
        push!(err,"tm.csv pQmx values must lie in 0:pQmxTM (pQmxTM = $(nQ))")
    for n in 1:nQ
        costs = unique(tm["QmxCost"][tm["Qmx"] .== n])
        length(costs) > 1 &&
            push!(wrn,"tm.csv pQmxCost is inconsistent within peak group $(n): " *
                      "$(join(costs, ", ")); only the last value is used")
    end
    nEV = sum(max.(sp["EVz0"], 0); init=0)
    evtype = tm["EVtype"]
    any(>(nEV), evtype) &&
        push!(err,"tm.csv pEVtype contains values above the number of selected EV " *
                  "instances ($(nEV))")
    for k in 1:nEV
        isodd(count(==(k), evtype)) &&
            push!(err,"tm.csv pEVtype has an odd number of events for EV $(k) " *
                      "(departure without arrival)")
    end
    all(t -> t in (0,1), tm["Ton"]) || push!(err,"tm.csv pTon must be 0 or 1")
    for key in ("Qlight","Qequip","Bppl","Sdni","Sdhi")
        all(>=(0), tm[key]) || push!(err,"tm.csv p$(key) must be non-negative")
    end

    # ---- existing DER stock vs. available space ---------------------------
    area = 0.0
    for (name,n) in zip(sp["PV0"], sp["PVz0"])
        i = findfirst(==(name), pv["ty"])
        i === nothing || (area += n*pv["ar"][i])
    end
    area > bdg["Bpv"] &&
        push!(err,"sp.csv existing PV stock needs $(area) m2 but bdg_i.csv pBpv " *
                  "allows only $(bdg["Bpv"])")
    sum(sp["BESSz0"]; init=0) > bdg["Bbess"] &&
        push!(err,"sp.csv declares $(sum(sp["BESSz0"])) existing BESS modules but " *
                  "bdg_i.csv pBbess allows only $(bdg["Bbess"])")
    sum(sp["WINDz0"]; init=0) > bdg["Bwind"] &&
        push!(err,"sp.csv declares $(sum(sp["WINDz0"])) existing wind turbines but " *
                  "bdg_i.csv pBwind allows only $(bdg["Bwind"])")

    # ---- duplicate equipment names ----------------------------------------
    for prefix in ("CHP","ABP","HVAC","WH","PV","WIND","BESS","EV")
        names = filter(!=("0"), sp["$(prefix)0"])
        dups = unique(filter(n -> count(==(n), names) > 1, names))
        isempty(dups) ||
            push!(err,"sp.csv $(prefix) lists duplicate equipment: $(join(dups, ", "))")
    end

    # ---- topology names ---------------------------------------------------
    selected = Set{String}()
    for prefix in ("CHP","ABP","HVAC","WH")
        union!(selected, catalog_base_name.(filter(!=("0"), sp["$(prefix)0"])))
    end
    for i in 1:topo["N"], (col,name) in (("up",topo["up"][i]), ("lo",topo["lo"][i]))
        (col == "lo" && name == "Building") && continue
        catalog_base_name(name) in selected && continue
        push!(wrn,"topo.csv row $(i): $(col) \"$(name)\" matches no selected " *
                  "catalog equipment; the link will be ignored")
    end

    # ---- catalog parameter ranges -----------------------------------------
    for i in 1:wind["N"]
        wind["mn"][i] > wind["mx"][i] &&
            push!(err,"wind.csv row $(i) ($(wind["ty"][i])): mn must not exceed mx")
    end
    for i in 1:chp["N"]
        0 <= chp["mn"][i] <= 1 ||
            push!(err,"chp.csv row $(i) ($(chp["ty"][i])): mn must be in [0,1]")
        chp["h2p"][i] >= 0 ||
            push!(err,"chp.csv row $(i) ($(chp["ty"][i])): h2p must be non-negative")
    end
    for i in 1:pv["N"]
        0 < pv["eff"][i] <= 1 ||
            push!(err,"pv.csv row $(i) ($(pv["ty"][i])): eff must be in (0,1]")
    end

    # ---- investment windows -----------------------------------------------
    cfg["IT"] > tm["P"] &&
        push!(err,"cfg.pIT = $(cfg["IT"]) exceeds the number of periods ($(tm["P"]))")
    length(unique(tm["IW"])) == length(tm["IW"]) ||
        push!(err,"investment windows have duplicate start periods; reduce cfg.pIT " *
                  "or lengthen the horizon")
    return nothing
end

function validate_numeric_missing!(err, cfg, bdg, tm, chp, abp, hvac, wh, pv,
    wind, bess, ev, topo)
    sources = (
        ("cfg.csv", cfg, (
            "BESSsoc0" => "pBESSsoc0", "BESSsocf" => "pBESSsocf",
            "EVsoc0" => "pEVsoc0", "EVmnsoc" => "pEVmnsoc",
            "WHsto0" => "pWHsto0", "WHstof" => "pWHstof",
            "Tin0" => "pTin0", "QmxBuy" => "pQmxBuy", "QmxSell" => "pQmxSell",
            "ST" => "pST", "Gco2" => "pGco2", "Lco2" => "pLco2",
            "CO2mx" => "pCO2mx", "CO2cost" => "pCO2cost", "CO2ton" => "pCO2ton",
            "QmxTM" => "pQmxTM", "NSEcost" => "pNSEcost",
            "NSTcost" => "pNSTcost", "NSHWcost" => "pNSHWcost",
            "NSEVcost" => "pNSEVcost", "EVdriver" => "pEVdriver",
            "EVpen" => "pEVpen", "IT" => "pIT", "IR" => "pIR")),
        ("bdg_i.csv", bdg, (
            "Balt" => "pBalt", "Blon" => "pBlon", "Blat" => "pBlat",
            "Bihgp" => "pBihgp", "Bihgl" => "pBihgl", "Bihge" => "pBihge",
            "Bfoot" => "pBfoot", "Bkroof" => "pBkroof", "Bwest" => "pBwest",
            "Bkwest" => "pBkwest", "Beast" => "pBeast", "Bkeast" => "pBkeast",
            "Bnorth" => "pBnorth", "Bknorth" => "pBknorth", "Bsouth" => "pBsouth",
            "Bksouth" => "pBksouth", "Bchp" => "pBchp", "Bhvac" => "pBhvac",
            "Babp" => "pBabp", "Bwh" => "pBwh", "Bpv" => "pBpv",
            "Btilt" => "pBtilt", "Bazi" => "pBazi", "Balb" => "pBalb",
            "Btck" => "pBtck", "Bwind" => "pBwind", "Bbess" => "pBbess",
            "Bevmx" => "pBevmx")),
        ("bdg_ii.csv", bdg, (
            "Bk1" => "pBk1", "Bk2" => "pBk2", "Bk3" => "pBk3",
            "Btime" => "pBtime")),
        ("tm.csv", tm, (
            "QcostBuy" => "pQcostBuy", "QcostSell" => "pQcostSell",
            "Qmx" => "pQmx", "QmxCost" => "pQmxCost", "Qco2" => "pQco2",
            "Gcost" => "pGcost", "Lcost" => "pLcost", "Qlight" => "pQlight",
            "Qequip" => "pQequip", "Tout" => "pTout", "Tmx" => "pTmx",
            "Tmn" => "pTmn", "Ton" => "pTon", "HWdem" => "pHWdem",
            "Sdni" => "pSdni", "Sdhi" => "pSdhi", "Wms" => "pWms",
            "EVdem" => "pEVdem", "EVtm" => "pEVtm", "EVtype" => "pEVtype",
            "Bppl" => "pBppl")),
        ("chp.csv", chp, (
            "mx" => "mx", "mn" => "mn", "fcf" => "fcf", "tank" => "tank",
            "h2p" => "h2p", "sup" => "sup", "inv" => "inv", "fom" => "fom",
            "vom" => "vom", "life" => "life")),
        ("abp.csv", abp, (
            "mx" => "mx", "fcf" => "fcf", "ac" => "ac", "inv" => "inv",
            "fom" => "fom", "vom" => "vom", "life" => "life")),
        ("hvac.csv", hvac, (
            "HVmx" => "pHVmx", "ACmx" => "pACmx", "HVeff" => "pHVeff",
            "ACeff" => "pACeff", "temp" => "temp", "HVmx_" => "pHVmx_",
            "ACmx_" => "pACmx_", "HVeff_" => "pHVeff_", "ACeff_" => "pACeff_",
            "inv" => "inv", "fom" => "fom", "vom" => "vom", "life" => "life")),
        ("wh.csv", wh, (
            "mx" => "mx", "fcf" => "fcf", "eff" => "eff", "tank" => "tank",
            "inv" => "inv", "fom" => "fom", "vom" => "vom", "life" => "life")),
        ("pv.csv", pv, (
            "tech" => "tech", "mx" => "mx", "ar" => "ar", "eff" => "eff",
            "loss" => "loss", "fail" => "fail", "inv" => "inv", "lc" => "lc",
            "fom" => "fom", "life" => "life")),
        ("wind.csv", wind, (
            "mx" => "mx", "mn" => "mn", "vmn" => "vmn", "vmx" => "vmx",
            "fail" => "fail", "inv" => "inv", "lc" => "lc", "fom" => "fom",
            "life" => "life")),
        ("bess.csv", bess, (
            "mx" => "mx", "up" => "up", "dn" => "dn", "effu" => "effu",
            "effd" => "effd", "inv" => "inv", "lc" => "lc", "fom" => "fom",
            "life" => "life")),
        ("ev.csv", ev, (
            "mx" => "mx", "drv" => "drv", "cold" => "cold", "up" => "up",
            "dn" => "dn", "effu" => "effu", "effd" => "effd")),
        ("topo.csv", topo, ("eff" => "eff",)),
    )

    for (filename, input_values, fields) in sources
        for field in fields
            key, column = field.first, field.second
            value = input_values[key]
            row = value isa AbstractArray ? findfirst(ismissing, value) :
                (ismissing(value) ? 1 : nothing)
            isnothing(row) || push!(err,
                "$filename column $column has a missing value at row $row")
        end
    end
    return nothing
end

"""
    check_temperature(cfg, tm, bdg, sp, chp, abp, hvac, topo)

Compute the no-conditioning trajectory and an optimistic reachable temperature
interval without a solver or input mutation. Use the same timestep coefficients
and indoor bounds as the model, including hard comfort only when Ton = 1 and
NSTcost = 0. HVAC capacity is derated against end-of-period indoor temperature.

All eligible existing/candidate units are assumed available from the start;
shared installation space, fuel/electricity supply, CHP heat allocation, and
absorption-chiller heat supply are relaxed. An empty interval identifies a
thermal capacity shortfall; a nonempty interval does not prove full feasibility.
"""
function check_temperature(cfg::Dict, tm::Dict, bdg::Dict, sp::Dict,
    chp::Dict, abp::Dict, hvac::Dict, topo::Dict)
    k1, k2, k3 = thermal_response_coefficients(bdg, tm["TM"])
    forcing = k1 .* tm["Tout"] + k2 .* tm["Q_R"] +
        k3 .* (tm["Qihg_P"] + tm["Qihg_L"] + tm["Qihg_E"])
    free_temperature = zeros(tm["P"])
    previous = cfg["Tin0"]
    for t in 1:tm["P"]
        previous = (1-k1[t])*previous + forcing[t]
        free_temperature[t] = previous
    end

    hv = eligible_thermal_units(sp, "HVAC", hvac["ty"], bdg["Bhvac"])
    cp = eligible_thermal_units(sp, "CHP", chp["ty"], bdg["Bchp"])
    ac = eligible_thermal_units(sp, "ABP", abp["ty"], bdg["Babp"])
    heat = hvac["HVmx"][hv]
    cold = hvac["ACmx"][hv]
    heat_derate = hvac["HVmx_"][hv]
    cold_derate = hvac["ACmx_"][hv]
    valid = bdg["Bk3"] >= 0 && all(values -> all(x -> isfinite(x) && x >= 0, values),
        (heat, cold, chp["mx"][cp], chp["h2p"][cp], topo["chp_bdg"][cp,1], abp["mx"][ac])) &&
        all(isfinite, heat_derate) && all(isfinite, cold_derate)
    valid || return (; free_temperature, conflict=nothing, skipped=true)

    # Generous upper bounds: every eligible CHP can devote all heat to the
    # building, and every eligible absorption chiller has sufficient supply.
    extra_heat = sum((chp["mx"][c] * topo["chp_bdg"][c,1] *
        (chp["h2p"][c] > 0 ? chp["h2p"][c] : 1) for c in cp); init=0.0)
    extra_cold = sum(abp["mx"][ac]; init=0.0)
    bound_lower = minimum([tm["Tout"]; tm["Tmn"]])
    bound_upper = maximum([tm["Tout"]; tm["Tmx"]])
    lower = upper = cfg["Tin0"]
    for t in 1:tm["P"]
        lower = temperature_at_capacity((1-k1[t])*lower + forcing[t], k3[t],
            tm["Tout"][t], cold, cold_derate, extra_cold; cooling=true)
        upper = temperature_at_capacity((1-k1[t])*upper + forcing[t], k3[t],
            tm["Tout"][t], heat, heat_derate, extra_heat; cooling=false)
        hard_comfort = tm["Ton"][t] == 1 && cfg["NSTcost"] == 0
        required_lower = hard_comfort ? max(bound_lower, tm["Tmn"][t]) : bound_lower
        required_upper = hard_comfort ? min(bound_upper, tm["Tmx"][t]) : bound_upper
        if lower > required_upper + 1e-6 || upper < required_lower - 1e-6
            direction = lower > required_upper ? "cooling" : "heating"
            conflict = (; period=t, direction, lower, upper, required_lower, required_upper)
            return (; free_temperature, conflict, skipped=false)
        end
        lower = max(lower, required_lower)
        upper = min(upper, required_upper)
    end
    return (; free_temperature, conflict=nothing, skipped=false)
end

"""Find the catalog copies enabled by the same selection rules as thermal equipment."""
function eligible_thermal_units(sp::Dict, prefix, names, space)
    enabled = Set{String}()
    for i in eachindex(sp["$(prefix)0"])
        name = sp["$(prefix)0"][i]
        name == "0" && continue
        count = sp["$(prefix)yn"][i] == "YES" ? max(space, sp["$(prefix)z0"][i]) : sp["$(prefix)z0"][i]
        for unit in 1:count
            push!(enabled, string(name, '-', unit))
        end
    end
    return findall(name -> name in enabled, names)
end

# Solve the monotone, piecewise-linear temperature balance by bisection. The
# capacity loss uses the resulting temperature, as in eHVACmxHT/eHVACmxAC.
function temperature_at_capacity(rhs, response, outdoor, capacities, derates, extra; cooling)
    nominal = sum(capacities; init=0.0) + extra
    (iszero(response) || iszero(nominal)) && return rhs
    lo, hi = cooling ? (rhs-response*nominal, rhs) : (rhs, rhs+response*nominal)
    for _ in 1:48
        temperature = (lo+hi)/2
        difference = max(cooling ? outdoor-temperature : temperature-outdoor, 0)
        capacity = extra + sum((capacities[h] *
            clamp(1-max(derates[h],0)*difference, 0, 1) for h in eachindex(capacities)); init=0.0)
        residual = cooling ? temperature+response*capacity-rhs : temperature-response*capacity-rhs
        if residual > 0
            hi = temperature
        else
            lo = temperature
        end
    end
    # Round outwards so the computed interval remains optimistic.
    return cooling ? lo : hi
end
