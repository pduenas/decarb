"""
validate_selection(sp::Dict,names::Vector,prefix::String)

Checks that every equipment name selected in sp.csv exists in the corresponding
catalog. Called from extend_catalog, where a missing name would otherwise raise
an opaque DimensionMismatch.

inputs:
sp          dictionary with equipment selection
names       vector of catalog names available for this equipment class
prefix      equipment class label used in the error message

"""

function validate_selection(selected,names,prefix::String)
    for s in selected
        s == "0" && continue
        any(n -> n == s || startswith(n, s * "-"), names) && continue
        error("❗  $(prefix) \"$(s)\" is selected in sp.csv but absent from the " *
              "catalog. Check the spelling, and that its capacity is non-zero " *
              "(zero-capacity rows are discarded on load).\n" *
              "    available: $(join(unique(names), ", "))")

    end
    return nothing
end


"""
validate_inputs(cfg::Dict,sp::Dict,bdg::Dict,tm::Dict,chp::Dict,abp::Dict,
    hvac::Dict,wh::Dict,pv::Dict,wind::Dict,bess::Dict,ev::Dict,topo::Dict)

Validates loaded inputs before the optimization model is built. Raises an error
for conditions that would make the model infeasible or ill-posed, and warns for
inputs that are accepted but inactive in this release

inputs:
cfg     dictionary with configuration input data
sp      dictionary with equipment selection data
bdg     dictionary with building data
tm      dictionary with time series data
chp     dictionary with CHP data
abp     dictionary with absorption chiller data
hvac    dictionary with HVAC data
wh      dictionary with water heater data
pv      dictionary with PV data
wind    dictionary with wind turbine data
bess    dictionary with BESS data
ev      dictionary with EV data
topo    dictionary with topology of thermal connections

"""
function validate_inputs(cfg::Dict,sp::Dict,bdg::Dict,tm::Dict,chp::Dict,abp::Dict,
    hvac::Dict,wh::Dict,pv::Dict,wind::Dict,bess::Dict,ev::Dict,topo::Dict)

    err = String[]       # fatal
    wrn = String[]       # accepted but suspect

    # ---- configuration -----------------------------------------------------
    for k in ("BESSsoc0","BESSsocf","EVsoc0","EVmnsoc","WHsto0","WHstof")
        0 <= cfg[k] <= 1 || push!(err,"cfg.$(k) = $(cfg[k]); expected a fraction in [0,1]")
    end
    0 < cfg["IR"] < 1 || push!(err,"cfg.IR = $(cfg["IR"]); expected an annual rate in (0,1)")
    cfg["QmxBuy"]  > 0 || push!(err,"cfg.QmxBuy must be positive")
    cfg["QmxSell"] >= 0 || push!(err,"cfg.QmxSell must be non-negative")
    cfg["EVdriver"] in (-1,0,1) ||
        push!(err,"cfg.EVdriver = $(cfg["EVdriver"]); expected -1, 0 or 1")
    for k in ("NSEcost","NSTcost","NSHWcost","NSEVcost","Gco2","Lco2")
        cfg[k] >= 0 || push!(err,"cfg.$(k) must be non-negative")
    end

    # ---- time series -------------------------------------------------------
    tm["P"] > 0 || push!(err,"tm.csv contains no periods")
    all(tm["TM"] .> 0) ||
        push!(err,"tm.csv timestamps are not strictly increasing")
    issorted(tm["Date"]) || push!(err,"tm.csv dates are not sorted")
    all(tm["Tmn"] .<= tm["Tmx"]) ||
        push!(err,"tm.csv has Tmn > Tmx in at least one period")
    all(tm["QcostBuy"] .>= 0) || push!(err,"tm.csv QcostBuy must be non-negative")
    all(tm["HWdem"]   .>= 0) || push!(err,"tm.csv HWdem must be non-negative")
    all(tm["Wms"]     .>= 0) || push!(err,"tm.csv Wms must be non-negative")
    any(tm["QcostSell"] .> tm["QcostBuy"]) &&
        push!(wrn,"sale price exceeds purchase price in some periods; " *
                  "unbounded arbitrage is prevented only by the buy/sell binary")

    # ---- comfort reachability ---------------------------------------------
    Tup = maximum([tm["Tout"];tm["Tmx"]])
    Tlo = minimum([tm["Tout"];tm["Tmn"]])
    cfg["Tin0"] <= Tup && cfg["Tin0"] >= Tlo ||
        push!(err,"cfg.Tin0 = $(cfg["Tin0"]) lies outside the reachable indoor " *
                  "range [$(Tlo), $(Tup)] implied by Tout/Tmn/Tmx")
    if cfg["NSTcost"] > 0 && all(tm["Tmx"] .>= Tup)
        push!(wrn,"upward temperature discomfort is priced but structurally " *
                  "impossible: max(Tmx) equals the indoor upper bound")
    end

    # ---- divide-by-zero guards --------------------------------------------
    for (d,name,keys) in ((chp,"chp",("fcf",)), (abp,"abp",("fcf","ac")),
                          (wh,"wh",("fcf","eff")), (pv,"pv",("ar",)),
                          (bess,"bess",("mx","effu","effd")),
                          (ev,"ev",("mx","effu","effd")))
        for k in keys, i in 1:d["N"]
            d[k][i] > 0 && continue
            k == "fcf" && d["fuel"][i] == "0" && continue   # electric: fcf unused
            push!(err,"$(name).csv row $(i) ($(d["ty"][i])): $(k) must be positive")
        end
    end
    for i in 1:wind["N"]
        wind["vmx"][i] > wind["vmn"][i] ||
            push!(err,"wind.csv row $(i) ($(wind["ty"][i])): vmx must exceed vmn")
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

    # ---- DHW served by something -----------------------------------------
    if any(tm["HWdem"] .> 0) && wh["N"] == 0 &&
       all(iszero, topo["chp_bdg"][:,2]) && cfg["NSHWcost"] == 0
        push!(err,"hot water is demanded but no water heater or CHP hot-water " *
                  "link exists, and NSHWcost = 0 forbids unserved demand")
    end

    # ---- topology ---------------------------------------------------------
    for i in 1:topo["N"]
        topo["in"][i] in ("hot air","hot water","cooling") && continue
        push!(wrn,"topo.csv row $(i): unrecognised product \"$(topo["in"][i])\"; " *
                  "the link will be ignored")
    end

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

    # ---- report ----------------------------------------------------------
    for w in wrn
        println("   \u26A0  ", w)
    end
    if !isempty(err)
        error("❗  input validation failed:\n" *
              join(["    • " * e for e in err], "\n") * "\n")
    end
    println("   \u2139  inputs validated (", length(wrn), " warning(s))")

    return nothing
end