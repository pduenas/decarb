using Test
using DECARB
using CSV, DataFrames, Dates

const CASE_SRC = joinpath(@__DIR__, "..", "examples", "minimal")
const HIGHS    = 2          # i_solver
const MIP_GAP  = 1e-2
const TLIMIT   = 120.0

"""
    stage(; patches) -> path

Copy examples/minimal into a temp dir, apply textual patches to files under
`in/`, run the model, return the case path. `patches` maps a filename to either
one `old => new` pair or a collection of pairs.
"""
function stage(; patches = Dict{String,Any}(), run = true)
    dst = mktempdir()
    cp(CASE_SRC, dst; force = true)
    for (fname, file_patches) in patches
        p = joinpath(dst, "in", fname)
        s = read(p, String)
        replacements = file_patches isa Pair ? (file_patches,) : file_patches
        for (old, new) in replacements
            @assert occursin(old, s) "patch target not found in $fname: $old"
            s = replace(s, old => new)
        end
        write(p, s)
    end
    run && DECARB.run_decarb!(dst, MIP_GAP, TLIMIT, HIGHS, false)
    return dst
end

readout(path, f) = CSV.read(joinpath(path, "out", f), DataFrame)

# ─────────────────────────────────────────────────────────────────────────────
@testset "DECARB.jl" begin

@testset "smoke" begin
    @test isdefined(DECARB, :run_decarb!)
    @test pkgversion(DECARB) >= v"1.0.0"
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "minimal :: runs and writes outputs" begin
    p = stage()
    for f in ("eq1.csv", "eq2.csv", "ts.csv")
        @test isfile(joinpath(p, "out", f))
    end
    # balance.csv is written ONLY when the electricity balance fails to close
    @test !isfile(joinpath(p, "out", "balance.csv"))
    @test !isfile(joinpath(p, "out", "iis_constraints.txt"))
end

@testset "minimal :: time series integrity" begin
    p  = stage()
    ts = readout(p, "ts.csv")

    @test nrow(ts) == 24
    @test !any(ismissing, Matrix(ts[:, Not(:Date)]))

    # electricity balance, recomputed independently of the model
    bal = ts.dem .+ ts.sell .+ ts.hvac_HT .+ ts.hvac_AC .+ ts.Qwh .+
          ts.bess_UP .+ ts.ev_UP .-
          ts.buy .- ts.pv .- ts.wind .- ts.chp .-
          ts.bess_DN .- ts.ev_DN .- ts.nse
    @test maximum(abs.(bal)) < 0.1          # matches write_outputs tolerance

    # deselected technologies must contribute nothing
    @test all(iszero, ts.chp)
    @test all(iszero, ts.wind)
    @test all(iszero, ts.bess_UP) && all(iszero, ts.bess_DN)
    @test all(iszero, ts.ev_UP)   && all(iszero, ts.ev_DN)
    @test all(iszero, ts.bess_SOC)          # guards §0.4 regression
    @test all(iszero, ts.WHsoc)             # guards §0.3 regression (tankless)

    # selected technologies must do something
    @test sum(ts.hvac_HT) > 0               # winter day -> heating
    @test sum(ts.pv)      > 0               # daylight hours present
    @test sum(ts.buy)     > 0

    # indoor temperature stays inside the comfort band (slack exists but is priced)
    @test all(ts.Temp .>= 19.0)
    @test all(ts.Temp .<= 26.5)

    # duals recovered from the fixed-integer re-solve
    @test any(!iszero, ts.dualQ)
end

@testset "minimal :: gas path and emissions" begin
    p  = stage()
    ts = readout(p, "ts.csv")
    e1 = readout(p, "eq1.csv")

    # gas water heater must consume gas and serve the DHW demand
    @test sum(ts.Gwh)  > 0
    @test sum(ts.WHhw) > 0
    @test sum(ts.HWns) ≈ 0 atol = 1e-6      # demand fully served
    active = ts.WHhw .> 0
    @test ts.Gwh[active] ≈ ts.WHhw[active] ./ (0.95 * 0.82) atol = 0.02
    @test all(iszero, ts.Gchp) && all(iszero, ts.Gabp)

    val(n) = only(e1.eq[e1.name .== n])
    @test val("direct_emissions")   > 0     # gas combustion
    @test val("indirect_emissions") > 0     # grid imports
    @test val("fuel_purchases")     > 0
end

@testset "minimal :: equipment report, all brownfield" begin
    p   = stage()
    eq2 = readout(p, "eq2.csv")

    @test Set(eq2.Eq) == Set(["HP-MINISPLIT", "WH-TANKLESS-GAS", "PV-MONO"])
    # nothing is newly built: pIT = 0 disables investment
    @test all(iszero, eq2.New)
    @test all(iszero, eq2.CAPEX)
    @test only(eq2.Qty[eq2.Eq .== "PV-MONO"]) == 4
end

# ─────────────────────────────────────────────────────────────────────────────
# Regression guard for the read_thermal aggregation defect (§3.1, last review).
# Two existing units of ONE type must yield ONE row with Qty=2, New=0, CAPEX=0.
# Current code emits two duplicate rows with New=[0,2].
@testset "multiunit :: no duplicate equipment rows" begin
    p = stage(patches = Dict(
        "sp.csv"    => "HP-MINISPLIT,1,NO"  => "HP-MINISPLIT,2,NO",
        "bdg_i.csv" => "pBhvac,1"           => "pBhvac,2",
    ))
    eq2 = readout(p, "eq2.csv")

    hp = eq2[eq2.Eq .== "HP-MINISPLIT", :]
    @test nrow(hp) == 1                       # one row per (type, window)
    @test only(hp.Qty) == 2
    @test only(hp.New) == 0                   # both units pre-existing
    @test only(hp.CAPEX) == 0

    # no (Eq, Date) pair may repeat anywhere in the file
    @test nrow(unique(eq2[:, [:Eq, :Date]])) == nrow(eq2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Regression guard for the DST crash (§0.2).  2019-03-10T02:00 in
# America/New_York does not exist; ZonedDateTime throws NonExistentTimeError.
@testset "dst_transition :: spring forward" begin
    patches = Dict(
        "cfg.csv" => "pP0,2019-01-01T00:00:00" => "pP0,2019-03-10T00:00:00",
        "tm.csv"  => ["2019-01-01T"         => "2019-03-10T",
                      "2019-01-02T00:00:00" => "2019-03-11T00:00:00"],
    )
    @test_logs (:warn, r"local time does not exist") match_mode=:any stage(; patches)
end

# ─────────────────────────────────────────────────────────────────────────────
# Input validation: a name in sp.csv with no surviving catalogue row must
# produce a readable message, not DimensionMismatch (§B5 root cause).
@testset "validation :: unknown equipment name" begin
    err = try
        stage(patches = Dict("sp.csv" => "HP-MINISPLIT,1,NO" => "NO-SUCH-UNIT,1,NO"))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("NO-SUCH-UNIT", sprint(showerror, err))
end

# ─────────────────────────────────────────────────────────────────────────────
# Conversion factors are efficiencies: useful output divided by fcf gives
# purchased fuel, and fuel-consuming equipment cannot exceed unity.
@testset "validation :: fuel conversion factors are efficiencies" begin
    err = try
        stage(patches = Dict(
            "wh.csv" => "WH-TANKLESS-GAS,12.0,0.82,G" =>
                        "WH-TANKLESS-GAS,12.0,1.20,G",
        ))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("fcf must be in (0,1]", sprint(showerror, err))
end

# Economic summary includes the solved objective and keeps annuity separate
# from fixed O&M.
@testset "economic summary" begin
    p  = stage()
    e1 = readout(p, "eq1.csv")
    val(n) = only(e1.eq[e1.name .== n])
    @test "total_cost" in e1.name
    @test val("equipment_annuity") ≈ sum(val(n) for n in (
        "chp_annuity", "hvac_annuity", "abp_annuity", "water_heater_annuity",
        "pv_annuity", "wind_annuity", "battery_annuity")) atol = 0.02
    @test val("equipment_annuity") != val("fixed_om_cost")
end

end # DECARB.jl
