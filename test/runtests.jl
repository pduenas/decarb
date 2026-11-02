using Test
using Logging
using DECARB
using Aqua
using CSV, DataFrames, Dates
using JuMP, HiGHS

const CASE_SRC = joinpath(@__DIR__, "..", "examples", "minimal")
const MIP_GAP  = 1e-2
const TLIMIT   = 120.0

# Opt-in slow tier: DECARB_FULL=true julia --project=. -e 'using Pkg; Pkg.test()'
const FULL = get(ENV, "DECARB_FULL", "") == "true"

# Regression against the committed examples/minimal/out/eq1.csv, using the shared baseline solve.
# Fixtures are rounded in the CSV (0.01 for costs, 0.001 for emissions), hence the small absolute floors.
const REGRESSION_GAP = 1e-4
const REGRESSION_KEYS = (
    "total_cost"  => 0.01,
    "energy_bill" => 0.01,
    "emissions"   => 0.002,   # direct_emissions + indirect_emissions
)

# Release CSV memory maps so Windows can delete/overwrite temp files.
release_mmaps() = Sys.iswindows() && GC.gc()

"""
    stage(; patches, run=true, periods=nothing) -> path

Copy examples/minimal into a temp dir, apply textual patches to files under
`in/`, optionally run the model, return the case path. `patches` maps a filename to either
one `old => new` pair or a collection of pairs. `periods` keeps only the first
n rows of tm.csv; validate_inputs still runs, so inconsistent truncations fail
loudly.
"""
function stage(; patches = Dict{String,Any}(), run = true,
               periods::Union{Nothing,Int} = nothing)
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
    if periods !== nothing
        tm_file = joinpath(dst, "in", "tm.csv")
        # Read from memory, as strings, so values are not reformatted or mapped.
        df = CSV.read(read(tm_file), DataFrame; types=String)
        CSV.write(tm_file, first(df, periods))
    end
    run && DECARB.run_decarb!(dst; mip_gap=MIP_GAP, time_limit=TLIMIT, verbose=false)
    return dst
end

readout(path, f) = CSV.read(joinpath(path, "out", f), DataFrame)

const BASE = Ref{Any}(nothing)
# One default-settings solve of examples/minimal, shared by read-only testsets.
function baseline()
    if BASE[] === nothing
        p = stage(run=false)
        mkpath(joinpath(p, "out"))
        write(joinpath(p, "out", "balance.csv"), "stale\n")
        write(joinpath(p, "out", "notes.txt"), "keep\n")
        mktemp() do _, io
            r = redirect_stdout(io) do
                DECARB.run_decarb!(p; return_result=true)  # defaults: verbose, mip_gap, duals
            end
            seekstart(io)
            BASE[] = (; path=p, result=r, stdout=read(io, String))
        end
    end
    return BASE[]
end

const INVEST_PATCHES = Dict(
    "cfg.csv" => ["pIT,0" => "pIT,1",
                  "pNSTcost,1.0" => "pNSTcost,1000.0"],
    "sp.csv"  => "HP-MINISPLIT,1,NO" => "HP-MINISPLIT,0,YES",
)
const INVEST_BASE = Ref{Any}(nothing)
# One duals=true solve of the investment case, shared by read-only testsets.
function invest_baseline()
    INVEST_BASE[] === nothing && (INVEST_BASE[] = stage(patches=INVEST_PATCHES))
    return INVEST_BASE[]
end

# Load fixtures independently of validation, in the same order as the runner.
function load_fixture(path)
    cfg = DECARB.load_cfg(path)
    sp = DECARB.load_sp(path,cfg)
    bdg = DECARB.load_bdgii(path,DECARB.load_bdgi(path))
    tm = DECARB.load_tm(path,cfg,bdg)
    pv = DECARB.load_pv(path,tm,bdg)
    chp,abp,hvac,wh,ev = DECARB.load_extended_catalog(path,tm,sp,bdg)
    wind = DECARB.load_wind(path,tm)
    bess = DECARB.load_bess(path)
    topo = DECARB.load_topo(path,chp,abp)
    return (; cfg, sp, bdg, tm, chp, abp, hvac, wh, pv, wind, bess, ev, topo)
end

# ─────────────────────────────────────────────────────────────────────────────
# Representative required numeric fields: common cases, field-specific rules.
const INPUT_FILE_CASES = [
    (file="cfg.csv",    field="pIR",      vertical=true,  integer=false, zero=true,  negative=false, full=true),
    (file="bdg_i.csv",  field="pBtck",    vertical=true,  integer=true,  zero=true,  negative=false, full=true),
    (file="bdg_ii.csv", field="pBtime",   vertical=false, integer=false, zero=false, negative=false, full=true),
    (file="sp.csv",     field="pHVACz0",  vertical=false, integer=true,  zero=true,  negative=false, full=true),
    (file="tm.csv",     field="pTout",    vertical=false, integer=false, zero=true,  negative=true,  full=true),
    (file="chp.csv",    field="fcf",      vertical=false, integer=false, zero=false, negative=false, full=false),
    (file="abp.csv",    field="fcf",      vertical=false, integer=false, zero=false, negative=false, full=false),
    (file="hvac.csv",   field="life",     vertical=false, integer=false, zero=false, negative=false, full=false),
    (file="wh.csv",     field="fcf",      vertical=false, integer=false, zero=false, negative=false, full=false),
    (file="pv.csv",     field="tech",     vertical=false, integer=true,  zero=false, negative=false, full=false),
    (file="wind.csv",   field="vmx",      vertical=false, integer=false, zero=false, negative=false, full=false),
    (file="bess.csv",   field="effu",     vertical=false, integer=false, zero=false, negative=false, full=false),
    (file="ev.csv",     field="effu",     vertical=false, integer=false, zero=false, negative=false, full=false),
    (file="topo.csv",   field="eff",      vertical=false, integer=false, zero=true,  negative=false, full=false),
]

# One parent directory per testset; each case gets a fresh subdirectory via newdir().
function with_case_dirs(f)
    mktempdir() do parent
        try
            f(() -> mktempdir(parent))
        finally
            release_mmaps() # one collection before the parent is removed
        end
    end
end

function input_case_result(path, spec, variant)
    cp(joinpath(CASE_SRC, "in"), path; force=true)
    if variant != :valid
        file = joinpath(path, spec.file)
        # Read the test fixture into memory; do not map the file we edit.
        df = CSV.read(read(file), DataFrame; types=String, header=spec.vertical ? false : true)
        if spec.vertical
            row = findfirst(==(spec.field), df[!, 1])
            @assert row !== nothing "fixture field not found: $(spec.file):$(spec.field)"
            df[row, variant == :missing_field ? 1 : 2] =
                variant == :missing_field ? "omitted_$(spec.field)" : variant
        elseif variant == :missing_field
            # Keep column positions intact: test absence of the required name.
            rename!(df, spec.field => "omitted_$(spec.field)")
        else
            df[1, spec.field] = variant
        end
        CSV.write(file, df; header=!spec.vertical)
    end
    logger = Test.TestLogger()
    err = with_logger(logger) do
        try
            inputs = load_fixture(path)
            DECARB.validate_inputs(values(inputs)...; io=devnull)
            nothing
        catch e
            e
        end
    end
    return (; error=err, logs=logger.logs)
end

function test_input_files()
  with_case_dirs() do newdir
    # Catch new input files that have not yet been added to this shared coverage.
    @test Set(spec.file for spec in INPUT_FILE_CASES) ==
        Set(filter(name -> endswith(name, ".csv"), readdir(joinpath(CASE_SRC, "in"))))
    @testset "valid fixture" begin
        result = input_case_result(newdir(), first(INPUT_FILE_CASES), :valid)
        @test result.error === nothing
        @test isempty(result.logs)
    end
    for spec in INPUT_FILE_CASES
        @testset "$(spec.file) :: $(spec.field)" begin
            for variant in (spec.full ? (:missing_field, "invalid", "", "0", "-1", "NaN", "Inf") :
                                        (:missing_field, "invalid", ""))
                @testset "$(repr(variant))" begin
                    result = input_case_result(newdir(), spec, variant)
                    warning_expected = variant == "invalid" ||
                        (spec.integer && variant in ("NaN", "Inf"))
                    @test length(result.logs) == Int(warning_expected)
                    @test all(log -> log.level == Logging.Warn && log._module == CSV &&
                        occursin(r"error parsing (Float64|Int64|Int8|UInt8)", string(log.message)), result.logs)
                    accepted = (variant == "0" && spec.zero) ||
                        (variant == "-1" && spec.negative)
                    if accepted
                        @test result.error === nothing
                    else
                        @test result.error isa ArgumentError
                    end
                end
            end
        end
    end
    @testset "specific rejected inputs" begin
        for (spec, variant) in (
            ((file="tm.csv", field="pSdni", vertical=false), ""),
            ((file="pv.csv", field="tech", vertical=false), "4"),
            ((file="pv.csv", field="tech", vertical=false), "0"),
            ((file="bdg_i.csv", field="pBtz", vertical=true), "Not/AZone"),
        )
            @testset "$(spec.file) :: $(spec.field) = $(repr(variant))" begin
                result = input_case_result(newdir(), spec, variant)
                @test result.error isa ArgumentError
            end
        end
    end
  end
end

# Exercise central validation directly, including values that some loaders reject
# earlier, and verify the documented boundaries across every affected catalog.
function test_input_bounds()
    mktempdir() do path
        cp(joinpath(CASE_SRC, "in"), path; force=true)
        try
            fixture = load_fixture(path)
            invalid_number = (missing, "invalid", NaN, Inf, -Inf)
            specs = NamedTuple[
                (catalog=name, field="life",
                 invalid=(invalid_number..., 0.0, -1.0), valid=(0.1, 1.0),
                 rule="life must be a positive, finite number " *
                      "(it is the annuity denominator, even for deselected rows)")
                for name in (:chp, :abp, :hvac, :wh, :pv, :wind, :bess)
            ]
            for name in (:bess, :ev), field in ("effu", "effd")
                push!(specs, (catalog=name, field=field,
                    invalid=(invalid_number..., 0.0, -1.0, nextfloat(1.0)),
                    valid=(eps(Float64), 0.5, 1.0),
                    rule="$field must be a finite number in (0,1]"))
            end
            wind_rule = "vmn and vmx must be finite numbers with 0 <= vmn < vmx"
            push!(specs, (catalog=:wind, field="vmn",
                invalid=(invalid_number..., -1.0, fixture.wind["vmx"][1]),
                valid=(0.0, prevfloat(fixture.wind["vmx"][1])), rule=wind_rule))
            push!(specs, (catalog=:wind, field="vmx",
                invalid=(invalid_number..., 0.0, -1.0, fixture.wind["vmn"][1]),
                valid=(nextfloat(fixture.wind["vmn"][1]), 12.0), rule=wind_rule))
            for name in (:chp, :abp, :wh)
                # fixture rows consume fuel directly, so a zero fcf is rejected;
                # string values are rejected by the loader, not by validate_inputs
                push!(specs, (catalog=name, field="fcf",
                    invalid=(missing, NaN, Inf, -Inf, 0.0, -1.0, nextfloat(1.0)),
                    valid=(eps(Float64), 0.5, 1.0),
                    rule="fcf must be in (0,1] for fuel-consuming equipment, or " *
                         "[0,1] when direct fuel is not used"))
            end
            push!(specs, (catalog=:topo, field="eff",
                invalid=(invalid_number..., -eps(Float64), nextfloat(1.0)),
                valid=(0.0, 0.5, 1.0), rule="eff must be a finite number in [0,1]"))

            for spec in specs
                @testset "$(spec.catalog).$(spec.field)" begin
                    for (accepted, cases) in ((false, spec.invalid), (true, spec.valid))
                        for (idx, value) in enumerate(cases)
                            @testset "$(repr(value))" begin
                                inputs = deepcopy(fixture)
                                catalog = getproperty(inputs, spec.catalog)
                                @test catalog["N"] > 0
                                catalog[spec.field] = Any[catalog[spec.field]...]
                                catalog[spec.field][1] = value
                                # Non-mutation is checked on the first value of each group only.
                                check_unchanged = idx == 1
                                original = check_unchanged ? deepcopy(inputs) : nothing
                                if accepted
                                    @test DECARB.validate_inputs(values(inputs)...; io=devnull) === nothing
                                else
                                    expected = if ismissing(value)
                                        "$(spec.catalog).csv column $(spec.field) has a missing value at row 1"
                                    else
                                        row = spec.catalog == :topo ? "topo.csv row 1" :
                                            "$(spec.catalog).csv row 1 ($(catalog["ty"][1]))"
                                        "$row: $(spec.rule)"
                                    end
                                    @test_throws expected DECARB.validate_inputs(values(inputs)...; io=devnull)
                                end
                                check_unchanged && @test isequal(inputs, original)
                            end
                        end
                    end
                end
            end
        finally
            release_mmaps()
        end
    end
end

# A minimal recording optimizer exercises solver configuration without requiring
# a proprietary solver or accepting any vendor-specific attributes by accident.
struct RecordingOptimizer <: MOI.AbstractOptimizer
    name::String
    attributes::Dict{MOI.AbstractOptimizerAttribute,Any}
    supported_attributes::Set{MOI.AbstractOptimizerAttribute}
end
RecordingOptimizer(name; supported_attributes=[
    MOI.RelativeGapTolerance(), MOI.TimeLimitSec()]) =
        RecordingOptimizer(name, Dict{MOI.AbstractOptimizerAttribute,Any}(),
            Set{MOI.AbstractOptimizerAttribute}(supported_attributes))
MOI.is_empty(::RecordingOptimizer) = true
MOI.get(opt::RecordingOptimizer, ::MOI.SolverName) = opt.name
MOI.supports(opt::RecordingOptimizer, attr::MOI.AbstractOptimizerAttribute) =
    attr in opt.supported_attributes
MOI.set(opt::RecordingOptimizer, attr::MOI.AbstractOptimizerAttribute, value) =
    (opt.attributes[attr] = value)
MOI.get(opt::RecordingOptimizer, attr::MOI.AbstractOptimizerAttribute) = opt.attributes[attr]

function temperature_fixture()
    cfg = Dict{Any,Any}("Tin0"=>20.0, "NSTcost"=>0.0)
    tm = Dict{Any,Any}("P"=>3, "TM"=>ones(3), "Tout"=>fill(30.0,3),
        "Tmn"=>fill(18.0,3), "Tmx"=>fill(22.0,3), "Ton"=>ones(Int,3),
        "Q_R"=>zeros(3), "Qihg_P"=>zeros(3), "Qihg_L"=>zeros(3), "Qihg_E"=>zeros(3))
    bdg = Dict{Any,Any}("Bk1"=>0.1, "Bk2"=>0.0, "Bk3"=>1.0, "Btime"=>1.0,
        "Bhvac"=>1, "Bchp"=>1, "Babp"=>1)
    sp = Dict{Any,Any}("HVAC0"=>["HP"], "HVACz0"=>[1], "HVACyn"=>["NO"],
        "CHP0"=>String[], "CHPz0"=>Int[], "CHPyn"=>String[],
        "ABP0"=>String[], "ABPz0"=>Int[], "ABPyn"=>String[])
    hvac = Dict{Any,Any}("ty"=>["HP-1","unselected"], "HVmx"=>[0.0,100.0],
        "ACmx"=>[0.0,100.0], "HVmx_"=>[0.02,0.02], "ACmx_"=>[0.02,0.02])
    chp = Dict{Any,Any}("ty"=>String[], "mx"=>Float64[], "h2p"=>Float64[])
    abp = Dict{Any,Any}("ty"=>String[], "mx"=>Float64[])
    topo = Dict{Any,Any}("chp_bdg"=>zeros(0,2))
    return (; cfg, tm, bdg, sp, chp, abp, hvac, topo)
end



@testset "DECARB.jl" verbose=true begin

@testset "smoke" begin
    @test isdefined(DECARB, :run_decarb!)
    @test pkgversion(DECARB) >= v"1.0.0"
end

# A small loader fixture retains real demand/weather values without a solve.
function with_time_fixture(f; start="2019-01-01T05:00:00",
                           ends=["2019-01-01T06:00:00", "2019-01-01T07:00:00"],
                           zone="America/New_York", zero_row=false, dir=nothing)
    path = something(dir, mktempdir())
    try
        cp(joinpath(CASE_SRC, "in"), path; force=true)
        cfgfile = joinpath(path, "cfg.csv")
        write(cfgfile, replace(read(cfgfile, String),
            r"(?m)^pP0,[^\r\n]+" => "pP0,$start"))
        tmfile = joinpath(path, "tm.csv")
        rows = CSV.read(read(tmfile), DataFrame; types=String)[1:length(ends), :]
        rows.pP = ends
        if zero_row
            for column in names(rows)[2:end]
                rows[1, column] = "0"
            end
        end
        CSV.write(tmfile, rows)
        cfg = DECARB.load_cfg(path)
        bdg = DECARB.load_bdgii(path, DECARB.load_bdgi(path))
        bdg["Btz"] = zone
        f(path, cfg, bdg, rows)
    finally
        if dir === nothing
            release_mmaps() # Release CSV mappings before Windows temp cleanup.
            rm(path; recursive=true, force=true)
        end
    end
end

@testset "UTC timestamps :: input contract" begin
  with_case_dirs() do newdir
    # Clock labels that would be problematic locally are ordinary UTC instants.
    for (start, finish) in (("2019-03-10T02:00:00", "2019-03-10T03:00:00"),
                            ("2019-11-03T01:00:00", "2019-11-03T02:00:00"),
                            ("2019-01-01T05:00", "2019-01-01T06:00"),
                            ("2019-01-01T05:00:00.125", "2019-01-01T06:00:00.125"))
        with_time_fixture(; start, ends=[finish], dir=newdir()) do path, cfg, bdg, rows
            tm = DECARB.load_tm(path, cfg, bdg)
            @test cfg["P0"] == DateTime(start)
            @test tm["Putc"] == [DateTime(finish)]
            @test tm["TM"] == [1.0]
            @test tm["Pmidutc"] == [floor(DateTime(start), Minute) + Minute(30)]
        end
    end
    with_logger(Test.TestLogger()) do
        for start in ("", "invalid", "2019-02-30T05:00:00")
            with_time_fixture(; start, dir=newdir()) do path, cfg, bdg, rows
                @test_throws "cfg.csv pP0 must be a valid UTC date-time" DECARB.load_tm(path, cfg, bdg)
            end
        end
    end
    with_logger(Test.TestLogger()) do
        for value in ("", "invalid", "2019-02-30T05:00:00")
            with_time_fixture(; ends=[value], dir=newdir()) do path, cfg, bdg, rows
                @test_throws "tm.csv row 1 pP must be a valid UTC date-time" DECARB.load_tm(path, cfg, bdg)
            end
        end
    end
    with_time_fixture(; dir=newdir()) do path, cfg, bdg, rows
        write(joinpath(path, "tm.csv"), join(names(rows), ',') * "\n")
        @test_throws "tm.csv contains no periods" DECARB.load_tm(path, cfg, bdg)
    end
  end
end

@testset "UTC timestamps :: elapsed intervals and row preservation" begin
  with_case_dirs() do newdir
    with_time_fixture(; zero_row=true, dir=newdir(),
        ends=["2019-01-01T05:15:00", "2019-01-01T07:00:00"]) do path, cfg, bdg, rows
        tm = DECARB.load_tm(path, cfg, bdg)
        @test tm["P"] == 2
        @test tm["TM"] == [0.25, 1.75]
        @test tm["H"] == sum(tm["TM"]) == 2.0
        @test tm["Pmidutc"] == [DateTime(2019, 1, 1, 5, 7), DateTime(2019, 1, 1, 6, 7)]
        @test tm["Qlight"] == parse.(Float64, rows.pQlight)
        @test tm["Sdni"] == parse.(Float64, rows.pSdni)
        @test tm["Sdhi"] == parse.(Float64, rows.pSdhi)
        @test tm["Tout"] == parse.(Float64, rows.pTout)
        @test tm["Qlight"][1] == tm["Sdni"][1] == 0
    end
    for ends in (["2019-01-01T05:00:00"], ["2019-01-01T04:00:00"],
                 ["2019-01-01T06:00:00", "2019-01-01T06:00:00"],
                 ["2019-01-01T07:00:00", "2019-01-01T06:00:00"])
        with_time_fixture(; ends, dir=newdir()) do path, cfg, bdg, rows
            @test_throws "timestamps must be strictly increasing" DECARB.load_tm(path, cfg, bdg)
        end
    end
  end
end

@testset "UTC timestamps :: DST boundaries including the first interval" begin
  with_case_dirs() do newdir
    for (start, ends, expected_local) in (
        ("2018-03-11T07:00:00", ["2018-03-11T08:00:00", "2018-03-11T09:00:00"],
         [DateTime(2018,3,11,3), DateTime(2018,3,11,4)]),
        ("2018-11-04T06:00:00", ["2018-11-04T07:00:00", "2018-11-04T08:00:00"],
         [DateTime(2018,11,4,1), DateTime(2018,11,4,2)]),
        ("2018-11-04T05:00:00", ["2018-11-04T06:00:00", "2018-11-04T07:00:00"],
         [DateTime(2018,11,4,1), DateTime(2018,11,4,1)]),
    )
        with_time_fixture(; start, ends, zone="America/Chicago", dir=newdir()) do path, cfg, bdg, rows
            tm = DECARB.load_tm(path, cfg, bdg)
            @test tm["TM"] == [1.0, 1.0]
            @test tm["H"] == 2.0
            @test tm["Date"] == expected_local
            @test tm["Qlight"] == parse.(Float64, rows.pQlight)
            @test tm["Sdni"] == parse.(Float64, rows.pSdni)
        end
    end
  end
end

@testset "UTC timestamps :: solar uses the shared interval midpoint" begin
    path = joinpath(CASE_SRC, "in")
    cfg = DECARB.load_cfg(path)
    bdg = DECARB.load_bdgii(path, DECARB.load_bdgi(path))
    tm = DECARB.load_tm(path, cfg, bdg)
    # Explicit hourly midpoints for the Boston fixture, rather than its ends.
    midpoints = collect(DateTime(2019, 1, 1, 5, 30):Hour(1):DateTime(2019, 1, 2, 4, 30))
    elevation, azimuth = DECARB.calculate_sun_position(midpoints, bdg["Balt"], bdg["Blat"], bdg["Blon"])
    @test tm["Pmidutc"] == midpoints
    @test tm["ele"] ≈ elevation
    @test tm["azi"] ≈ azimuth
    endpoint_elevation, _ = DECARB.calculate_sun_position(tm["Putc"], bdg["Balt"], bdg["Blat"], bdg["Blon"])
    @test maximum(abs.(tm["ele"] - endpoint_elevation)) > 0.01
    pv = DECARB.load_pv(path, tm, bdg)
    bdg["Btz"] = "America/Chicago"
    other = DECARB.load_tm(path, cfg, bdg)
    @test other["Date"] != tm["Date"]
    for key in ("Putc", "Pmidutc", "TM", "H", "Qlight", "Qequip", "Sdni", "Sdhi", "ele", "azi", "Q_R")
        @test other[key] == tm[key]
    end
    @test DECARB.load_pv(path, other, bdg)["pv_cf"] == pv["pv_cf"]
end

@testset "solar beam terms are zero below the horizon" begin
    tech = [1]
    elevation = [0.0]
    azimuth = [0.0]
    @test DECARB.maximum_solar_potential(tech, 0.0, 0.0, 2, elevation, azimuth,
        [1.0], [0.0], 0.0, 1, [25.0]) == zeros(1, 1)
    @test DECARB.maximum_solar_potential(tech, 0.0, 0.0, 2, elevation, azimuth,
        [1.0], [1.0], 0.0, 1, [25.0])[1] > 0

    bdg = Dict{Any,Any}("Bkroof" => 1.0, "Bfoot" => 1.0, "Bkwest" => 1.0, "Bwest" => 1.0,
        "Bkeast" => 1.0, "Beast" => 1.0, "Bknorth" => 1.0, "Bnorth" => 1.0,
        "Bksouth" => 1.0, "Bsouth" => 1.0)
    @test DECARB.calculate_solar_radiation(bdg, [1.0], elevation, azimuth) == [0.0]

    inputs = load_fixture(joinpath(CASE_SRC, "in"))
    inputs.tm["ele"][1] = 0.0
    inputs.tm["Sdni"][1] = 1.0
    io = IOBuffer()
    DECARB.validate_inputs(values(inputs)...; io)
    @test occursin("pSdni is positive in periods with non-positive midpoint solar elevation",
        String(take!(io)))
end

FULL && @testset "UTC timestamps :: annual examples have 8760 elapsed hours" begin
    for case in ("annual", "investment")
        path = joinpath(@__DIR__, "..", "examples", case, "in")
        cfg = DECARB.load_cfg(path)
        @test cfg["P0"] == DateTime(2018, 1, 1, 5)
        # annual and investment share a byte-identical tm.csv; check it once
        case == "annual" || continue
        bdg = DECARB.load_bdgii(path, DECARB.load_bdgi(path))
        tm = DECARB.load_tm(path, cfg, bdg)
        @test tm["Putc"][1] == DateTime(2018, 1, 1, 6)
        @test tm["Putc"][end] == DateTime(2019, 1, 1, 5)
        @test tm["P"] == 8760
        @test all(==(1.0), tm["TM"])
        @test tm["H"] == sum(tm["TM"]) == 8760.0
        source = CSV.read(read(joinpath(path, "tm.csv")), DataFrame)
        @test tm["Qlight"] == source.pQlight
        @test tm["Sdni"] == source.pSdni
        @test tm["Sdhi"] == source.pSdhi
        @test tm["Tout"] == source.pTout
    end
end

@testset "input files :: common loading and validation" begin
    test_input_files()
end

@testset "input files :: column and row order independence" begin
    src = joinpath(@__DIR__,"..","examples","minimal","in")
    mktempdir() do tmp
        cp(src,joinpath(tmp,"orig"))
        cp(src,joinpath(tmp,"shuf"))
        orig = joinpath(tmp,"orig")
        shuf = joinpath(tmp,"shuf")

        # shuffle hvac.csv columns (reversed order)
        df = CSV.read(joinpath(orig,"hvac.csv"),DataFrame)
        CSV.write(joinpath(shuf,"hvac.csv"),df[:,reverse(names(df))])
        # shuffle cfg.csv rows (reversed order)
        lines = readlines(joinpath(orig,"cfg.csv"))
        write(joinpath(shuf,"cfg.csv"),join(reverse(lines),"\n")*"\n")

        function load_all(p)
            cfg = DECARB.load_cfg(p)
            sp = DECARB.load_sp(p,cfg)
            bdg = DECARB.load_bdgi(p)
            bdg = DECARB.load_bdgii(p,bdg)
            tm = DECARB.load_tm(p,cfg,bdg)
            hvac = DECARB.load_hvac(p,tm,sp,bdg)
            return cfg,hvac
        end

        cfg0,hvac0 = load_all(orig)
        cfg1,hvac1 = load_all(shuf)
        @test cfg1 == cfg0
        @test hvac1 == hvac0
    end
end

@testset "validation :: missing numeric input values" begin
    for (file, patch, expected) in (
        ("tm.csv", "0.18,0.05,0,0.0" => ",0.05,0,0.0",
            "tm.csv column pQcostBuy has a missing value at row 1"),
        ("hvac.csv", ",15.0" => ",",
            "hvac.csv column life has a missing value at row 1"),
    )
        path = stage(run=false, patches=Dict(file => patch))
        @test_throws expected begin
            inputs = load_fixture(joinpath(path, "in"))
            DECARB.validate_inputs(values(inputs)...; io=devnull)
        end
    end
end

@testset "validation :: negative counts and EV series" begin
    for (field, msg) in (("pIT", "cfg.csv pIT must be a non-negative integer"),
                         ("pQmxTM", "cfg.csv pQmxTM must be a non-negative integer"))
        path = stage(run=false, patches=Dict("cfg.csv" => "$field,0" => "$field,-1"))
        inputs = load_fixture(joinpath(path, "in"))
        @test_throws ArgumentError DECARB.validate_inputs(values(inputs)...; io=devnull)
        @test_throws msg DECARB.validate_inputs(values(inputs)...; io=devnull)
    end
    path = stage(run=false)
    for key in ("EVdem", "EVtm")
        inputs = load_fixture(joinpath(path, "in"))
        inputs.tm[key][1] = -1
        @test_throws ArgumentError DECARB.validate_inputs(values(inputs)...; io=devnull)
        @test_throws "tm.csv p$key must be non-negative" DECARB.validate_inputs(values(inputs)...; io=devnull)
    end
end

@testset "validation :: finite values and numeric bounds" begin
    test_input_bounds()
end

@testset "solver API :: generic MOI defaults and Gurobi tuning" begin
    generic = RecordingOptimizer("Industrial MIP";
        supported_attributes=[MOI.RelativeGapTolerance()])
    @test_logs (:warn, r"MOI.TimeLimitSec") begin
        DECARB.configure_solver!(direct_model(generic), 1//1000, 60)
        DECARB.configure_solver!(direct_model(generic), 1//1000, 60)
    end
    @test generic.attributes == Dict(
        MOI.RelativeGapTolerance() => 0.001,
    )
    @test !haskey(generic.attributes, MOI.TimeLimitSec())

    gurobi = RecordingOptimizer("Gurobi Optimizer")
    DECARB.configure_solver!(direct_model(gurobi), 0.002, 90)
    @test MOI.get(gurobi, MOI.RawOptimizerAttribute("MIPGap")) == 0.002
    @test MOI.get(gurobi, MOI.RawOptimizerAttribute("TimeLimit")) == 90
    @test MOI.get(gurobi, MOI.RawOptimizerAttribute("MIPFocus")) == 1
    @test !haskey(gurobi.attributes, MOI.RelativeGapTolerance())
end

@testset "solver API :: factory, overrides, quiet MIP without duals" begin
    p = stage(run=false)
    calls = Ref(0)
    factory = () -> begin
        calls[] += 1
        HiGHS.Optimizer()
    end
    mktemp() do _, io
        result = redirect_stdout(io) do
            @test_logs DECARB.run_decarb!(p; optimizer=factory,
                attributes=Dict{Any,Any}("mip_rel_gap"=>0.002,
                    MOI.TimeLimitSec()=>90.0),
                mip_gap=0.1, time_limit=120, duals=false, verbose=false,
                return_result=true)
        end
        seekstart(io)
        @test isempty(read(io, String))
        @test calls[] == 1
        @test get_attribute(result.model, "mip_rel_gap") == 0.002
        @test get_attribute(result.model, MOI.TimeLimitSec()) == 90.0
        @test get_attribute(result.model, MOI.Silent())
        @test primal_status(result.model) == MOI.FEASIBLE_POINT
        @test any(is_binary, all_variables(result.model))
        @test all(iszero, Matrix(readout(p, "ts.csv")[:, [:dualQ, :dualT, :dualHW]]))
    end
end

@testset "solver API :: optimizer_with_attributes and relaxed solve" begin
    p = stage(run=false)
    optimizer = optimizer_with_attributes(HiGHS.Optimizer, "presolve"=>"off")
    result = DECARB.run_decarb!(p; optimizer,
        attributes=Dict("presolve"=>"on"),
        relax_integrality=true, verbose=false, return_result=true)
    @test get_attribute(result.model, "presolve") == "on"
    @test get_attribute(result.model, "mip_rel_gap") == 1e-4
    @test !any(v -> is_binary(v) || is_integer(v), all_variables(result.model))
    @test has_duals(result.model)
    @test any(!iszero, readout(p, "ts.csv").dualQ)
    disabled_duals, _ = DECARB.read_econ(result.model, result.tm; duals=false)
    @test all(iszero, Matrix(disabled_duals))
    eq2 = readout(p, "eq2.csv")
    @test nrow(eq2) > 0
    @test eltype(eq2.Qty) <: AbstractFloat
    @test !any(<(1e-6), eq2.Qty)
end

@testset "solver API :: optional dual re-solve" begin
    for (relax, duals, expected_solves) in ((false,false,1), (false,true,2),
                                          (true,false,1), (true,true,1))
        model = Model(HiGHS.Optimizer)
        DECARB.configure_solver!(model, 1e-4, 30.0)
        set_silent(model)
        @variable(model, x, Bin)
        @constraint(model, x >= 0.5)
        @objective(model, Min, x)
        calls = Ref(0)
        set_optimize_hook(model, m -> begin
            calls[] += 1
            optimize!(m; ignore_optimize_hook=true)
        end)
        info = DECARB.solve_model!(mktempdir(), model, relax; duals, verbose=false)
        @test calls[] == expected_solves
        @test value(x) ≈ (relax ? 0.5 : 1.0)
        @test is_binary(x) == (!relax && !duals)
        @test has_duals(model) == (relax || duals)
        @test info.duals_available == duals
        @test info.termination_status == MOI.OPTIMAL
        @test info.primal_status == MOI.FEASIBLE_POINT
        @test info.objective_value ≈ (relax ? 0.5 : 1.0)
        @test isfinite(info.relative_gap) && info.solve_time_s >= 0
    end

    # failed LP re-solve: restore the MIP, keep its solution, report no duals
    model = Model(HiGHS.Optimizer)
    DECARB.configure_solver!(model, 1e-4, 30.0)
    set_silent(model)
    @variable(model, x, Bin)
    @constraint(model, x >= 0.5)
    @objective(model, Min, x)
    calls = Ref(0)
    set_optimize_hook(model, m -> begin
        calls[] += 1
        # make the second (fixed-integer) solve infeasible
        calls[] == 2 && fix(x, 0; force=true)
        optimize!(m; ignore_optimize_hook=true)
    end)
    info = @test_logs (:warn, r"duals will be zero") DECARB.solve_model!(
        mktempdir(), model, false; duals=true, verbose=false)
    @test calls[] == 3
    @test !info.duals_available
    @test info.objective_value ≈ 1.0       # captured before the re-solve
    @test info.primal_status == MOI.FEASIBLE_POINT
    @test info.dual_resolve_status == MOI.INFEASIBLE
    @test info.mip_resolved
    @test info.resolved_objective_value ≈ 1.0
    @test isfinite(info.resolved_relative_gap) && info.resolved_relative_gap >= 0
    @test info.resolved_solve_time_s >= 0
    @test !has_duals(model)
    @test is_binary(x) && !is_fixed(x)
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test value(x) ≈ 1.0
end
@testset "temperature validation :: comfort, capacity, and selection" begin
    f = temperature_fixture()
    before = deepcopy(f)
    report = DECARB.check_temperature(values(f)...)
    @test f == before
    @test report.free_temperature ≈ [21.0,21.9,22.71]
    @test report.conflict.period == 3
    @test report.conflict.direction == "cooling"
    @test report.conflict.lower ≈ 22.71
    @test report.conflict.required_upper == 22.0

    f.cfg["NSTcost"] = 1.0
    @test DECARB.check_temperature(values(f)...).conflict === nothing
    f.cfg["NSTcost"] = 0.0
    f.tm["Ton"] .= 0
    @test DECARB.check_temperature(values(f)...).conflict === nothing
    f.tm["Ton"] .= 1
    f.hvac["ACmx"][1] = 2.0
    f.sp["HVACz0"][1] = 0
    @test DECARB.check_temperature(values(f)...).conflict !== nothing
    f.sp["HVACyn"][1] = "YES"
    @test DECARB.check_temperature(values(f)...).conflict === nothing
    f.bdg["Bhvac"] = 0
    @test DECARB.check_temperature(values(f)...).conflict !== nothing

    f = temperature_fixture()
    f.tm["Tout"] .= 0
    @test DECARB.check_temperature(values(f)...).conflict.direction == "heating"
    f.hvac["HVmx"][1] = 4.0
    @test DECARB.check_temperature(values(f)...).conflict === nothing
end

@testset "temperature validation :: gains and calibration" begin
    f = temperature_fixture()
    f.bdg["Bk1"] = 0.0
    f.bdg["Bk2"] = 0.5
    f.bdg["Bk3"] = 2.0
    f.bdg["Btime"] = 0.25
    f.tm["TM"] = [0.25,0.75,0.25]
    f.tm["Q_R"] .= 2
    f.tm["Qihg_P"] .= 1
    f.tm["Ton"] .= 0
    report = DECARB.check_temperature(values(f)...)
    @test report.free_temperature ≈ [23.0,32.0,35.0]
    @test report.conflict.period == 2
    @test report.conflict.required_upper == 30.0 # Outdoor-derived bound still applies.

    f.bdg["Bk1"] = 0.5
    f.bdg["Bk2"] = f.bdg["Bk3"] = 0.0
    f.tm["TM"] = [1.0,0.25,0.5]
    f.tm["Tout"] .= 0
    @test DECARB.check_temperature(values(f)...).free_temperature ≈ [1.25,0.625,0.15625]

    f.bdg["Bk1"] = 1.0
    f.cfg["Tin0"] = 40.0
    @test DECARB.check_temperature(values(f)...).conflict === nothing
end

@testset "temperature validation :: derating uses resulting temperature" begin
    cold = DECARB.temperature_at_capacity(20.0,0.5,35.0,[2.0],[0.02],0.0; cooling=true)
    heat = DECARB.temperature_at_capacity(20.0,0.5,0.0,[2.0],[0.02],0.0; cooling=false)
    @test cold ≈ (20.0-0.5*2.0*(1-0.02*35.0))/(1+0.5*2.0*0.02) atol=1e-10
    @test heat ≈ (20.0+0.5*2.0)/(1+0.5*2.0*0.02) atol=1e-10
end

@testset "temperature validation :: alternative thermal supplies" begin
    f = temperature_fixture()
    f.abp["ty"] = ["Chiller-1"]
    f.abp["mx"] = [2.0]
    f.sp["ABP0"], f.sp["ABPz0"], f.sp["ABPyn"] = ["Chiller"], [0], ["YES"]
    @test DECARB.check_temperature(values(f)...).conflict === nothing

    f = temperature_fixture()
    f.tm["Tout"] .= 0
    f.chp["ty"], f.chp["mx"], f.chp["h2p"] = ["Boiler-1"], [3.0], [0.0]
    f.sp["CHP0"], f.sp["CHPz0"], f.sp["CHPyn"] = ["Boiler"], [1], ["NO"]
    f.topo["chp_bdg"] = [0.8 0.0]
    @test DECARB.check_temperature(values(f)...).conflict === nothing
    f.topo["chp_bdg"][1,1] = 0
    @test DECARB.check_temperature(values(f)...).conflict !== nothing
end

@testset "temperature validation :: integrated warnings and guards" begin
    p = stage(run=false, patches=Dict(
        "cfg.csv" => ["pNSTcost,1.0" => "pNSTcost,0.0", "pTin0,20.0" => "pTin0,19.0"],
        "sp.csv" => "HP-MINISPLIT,1,NO" => "HP-MINISPLIT,0,NO",
    ))
    path = joinpath(p, "in")
    io = IOBuffer()
    inputs = load_fixture(path)
    DECARB.validate_inputs(values(inputs)...; io)
    message = String(take!(io))
    @test occursin("temperature precheck: insufficient heating", message)
    @test occursin(string(inputs.tm["Date"][1]), message)
    input = joinpath(path, "bdg_ii.csv")
    header = "pBk1,pBk2,pBk3,pBtime\n"
    write(input, header * "-0.1,0.05,0.15,1\n")
    @test_throws "pBk1 must be in [0,1]" DECARB.validate_inputs(values(load_fixture(path))...; io=devnull)
    write(input, "pBk1,pBk2,pBk3\n0.1,0.05,0.15\n")
    try
        @test_throws Exception DECARB.load_bdgii(path, Dict())
    finally
        # Release CSV memory mappings before rewriting the file on Windows.
        release_mmaps()
    end
    write(input, header * "1,0.05,0.15,1\n")
    config = joinpath(path, "cfg.csv")
    write(config, replace(read(config, String), "pTin0,19.0" => "pTin0,100.0"))
    # Initial temperature is not a modeled period.
    @test DECARB.validate_inputs(values(load_fixture(path))...; io=devnull) === nothing
end

FULL && @testset "temperature validation :: annual investment diagnostic" begin
    path = joinpath(@__DIR__, "..", "examples", "investment", "in")
    cfg = DECARB.load_cfg(path)
    sp = DECARB.load_sp(path,cfg)
    bdg = DECARB.load_bdgii(path,DECARB.load_bdgi(path))
    # Reproduce the 15-minute calibration used by the original diagnostic,
    # without modifying the bundled case's calibration or catalog files.
    bdg["Btime"] = 0.25
    tm = DECARB.load_tm(path,cfg,bdg)
    chp,abp,hvac,_,_ = DECARB.load_extended_catalog(path,tm,sp,bdg)
    topo = DECARB.load_topo(path,chp,abp)
    report = DECARB.check_temperature(cfg,tm,bdg,sp,chp,abp,hvac,topo)
    @test report.conflict !== nothing
    @test report.conflict.direction == "cooling"
    @test maximum(report.free_temperature) > maximum(tm["Tout"])
    hvac["ACmx"] .*= 2
    @test DECARB.check_temperature(cfg,tm,bdg,sp,chp,abp,hvac,topo).conflict === nothing
end
@testset "thermal inputs :: bdg_ii schema and values" begin
    mktempdir() do case_path
        cp(CASE_SRC, case_path; force=true)
        path = joinpath(case_path, "in")
        input = joinpath(path, "bdg_ii.csv")
        header = "pBk1,pBk2,pBk3,pBtime\n"
        expected = Dict("Bk1" => 0.03, "Bk2" => 0.05, "Bk3" => 0.15, "Btime" => 1.0)

        function check_error(contents, message)
            write(input, contents)
            err = nothing
            with_logger(Test.TestLogger()) do
                try
                    inputs = load_fixture(path)
                    original = deepcopy(inputs)
                    err = try
                        DECARB.validate_inputs(values(inputs)...; io=devnull)
                    catch e
                        e
                    end
                    @test isequal(inputs, original)
                catch e
                    err = e
                end
            end
            @test err isa ArgumentError
            if err isa ArgumentError
                @test err.msg == message
            end
        end

        @testset "named columns" begin
            for contents in (
                header * "0.03,0.05,0.15,1\n",
                "pBtime,pBk3,pBk1,pBk2\n1,0.15,0.03,0.05\n",
            )
                write(input, contents)
                @test DECARB.validate_inputs(values(load_fixture(path))...; io=devnull) === nothing
                bdg = Dict{String,Any}("existing" => 42)
                @test DECARB.load_bdgii(path, bdg) === bdg
                @test bdg["existing"] == 42
                @test bdg == merge(expected, Dict("existing" => 42))
                @test all(bdg[key] isa Float64 for key in keys(expected))
            end
        end

        @testset "malformed CSV fails during loading" begin
            columns = ["pBk1", "pBk2", "pBk3", "pBtime"]
            vals = ["0.03", "0.05", "0.15", "1"]
            for omitted in eachindex(columns)
                keep = setdiff(eachindex(columns), [omitted])
                write(input, join(columns[keep], ',') * "\n" * join(vals[keep], ',') * "\n")
                try
                    @test_throws Exception DECARB.load_bdgii(path, Dict())
                finally
                    release_mmaps()
                end
            end
            # Native CSV/indexing errors are allowed; the loader is unchanged.
            for contents in (header, "")
                write(input, contents)
                try
                    @test_throws Exception DECARB.load_bdgii(path, Dict())
                finally
                    # Parser exceptions can bypass CSV's memory-map cleanup.
                    release_mmaps()
                end
            end
            # Types are keyed by column name, so extra columns are ignored.
            write(input, "note,pBk2,pBtime,pBk1,pBk3\ncalibrated,0.05,1,0.03,0.15\n")
            @test DECARB.load_bdgii(path, Dict())["Bk1"] == 0.03
            release_mmaps()
        end

        @testset "calibration interval" begin
            for hours in ("", "0", "-1", "NaN", "Inf", "-Inf", "invalid")
                message = isempty(hours) || hours == "invalid" ?
                    "bdg_ii.csv column pBtime has a missing value at row 1" :
                    occursin("Inf", hours) || hours == "NaN" ?
                    "bdg_ii.csv column pBtime has a non-finite value at row 1" :
                    "bdg_ii.pBtime must be a positive, finite number"
                check_error(header * "0.03,0.05,0.15,$hours\n",
                    message)
            end
            write(input, header * "0.03,0.05,0.15,0.25\n")
            @test DECARB.validate_inputs(values(load_fixture(path))...; io=devnull) === nothing
            @test DECARB.load_bdgii(path, Dict())["Btime"] == 0.25
        end

        @testset "coefficients" begin
            for value in ("", "-0.1", "1.1", "NaN", "Inf", "-Inf", "invalid")
                message = isempty(value) || value == "invalid" ?
                    "bdg_ii.csv column pBk1 has a missing value at row 1" :
                    occursin("Inf", value) || value == "NaN" ?
                    "bdg_ii.csv column pBk1 has a non-finite value at row 1" :
                    "bdg_ii.pBk1 must be in [0,1] for timestep resampling"
                check_error(header * "$value,0.05,0.15,1\n",
                    message)
            end
            for (column, row) in (("pBk2", v -> "0.03,$v,0.15,1\n"),
                                  ("pBk3", v -> "0.03,0.05,$v,1\n"))
                for value in ("", "NaN", "Inf", "-Inf", "invalid")
                    message = isempty(value) || value == "invalid" ?
                        "bdg_ii.csv column $column has a missing value at row 1" :
                        "bdg_ii.csv column $column has a non-finite value at row 1"
                    check_error(header * row(value), message)
                end
            end
            # A malformed first row must not be silently discarded.
            check_error(header * ",0.05,0.15,1\n0.03,0.05,0.15,1\n",
                "bdg_ii.csv column pBk1 has a missing value at row 1")
            for k1 in (0.0, 1.0)
                write(input, header * "$k1,0,-0.15,1\n")
                @test DECARB.validate_inputs(values(load_fixture(path))...; io=devnull) === nothing
                loaded = DECARB.load_bdgii(path, Dict())
                @test loaded ==
                    Dict("Bk1" => k1, "Bk2" => 0.0, "Bk3" => -0.15, "Btime" => 1.0)
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "validation :: sp.csv selections and investment flags" begin
    path = mktempdir()
    cp(CASE_SRC, path; force=true)
    inputs = load_fixture(joinpath(path, "in"))

    bad = deepcopy(inputs)
    bad.sp["PV0"][1] = "PV-UNKNOWN"
    @test_throws ArgumentError DECARB.validate_inputs(values(bad)...; io=devnull)

    bad = deepcopy(inputs)
    @test inputs.cfg["IT"] == 1
    bad.sp["HVACyn_raw"][1] = "yes"
    err = try DECARB.validate_inputs(values(bad)...; io=devnull) catch e e end
    @test err isa ArgumentError
    @test occursin("pHVACyn row 1", err.msg) && occursin("\"yes\"", err.msg)

    warn = deepcopy(inputs)
    warn.sp["PVz0"][1] = 0
    out = IOBuffer()
    DECARB.validate_inputs(values(warn)...; io=out)
    @test occursin("has pPVz0 = 0 and pPVyn = NO", String(take!(out)))
end

@testset "validation :: loaded inputs need no source files" begin
    path, inputs = mktempdir() do path
        cp(CASE_SRC, path; force=true)
        loaded = load_fixture(joinpath(path, "in"))
        release_mmaps()
        return path, loaded
    end
    @test !ispath(path)
    original = deepcopy(inputs)
    @test DECARB.validate_inputs(values(inputs)...; io=devnull) === nothing
    @test isequal(inputs, original)
end

@testset "thermal inputs :: case validates after loading" begin
    p = stage(run=false)
    input = joinpath(p, "in", "bdg_ii.csv")
    header = "pBk1,pBk2,pBk3,pBtime\n"
    # "" is rejected by the loader, "0" by validation; both surface through run_decarb!.
    for (hours, message) in (
        ("", "bdg_ii.csv column pBtime has a missing value at row 1"),
        ("0", "bdg_ii.pBtime must be a positive, finite number"),
    )
        write(input, header * "0.03,0.05,0.15,$hours\n")
        err = try
            DECARB.run_decarb!(p; verbose=false)
        catch e
            e
        end
        @test err isa ArgumentError
        if err isa ArgumentError
            @test err.msg == message
        end
    end
end

@testset "minimal :: stale outputs removed before input validation" begin
    p = stage(run=false)
    mkpath(joinpath(p, "out"))
    write(joinpath(p, "out", "eq1.csv"), "stale\n")
    write(joinpath(p, "in", "bdg_ii.csv"),
        "pBk1,pBk2,pBk3,pBtime\n0.03,0.05,0.15,0\n")

    @test_throws ArgumentError DECARB.run_decarb!(p; verbose=false)
    @test !isfile(joinpath(p, "out", "eq1.csv"))
end

@testset "minimal :: runs and writes outputs" begin
    b = baseline()
    p = b.path
    # run_decarb! returning `nothing` by default is asserted in the merged thermal recurrence testset.
    @test occursin("loading inputs", b.stdout)
    for f in ("eq1.csv", "eq2.csv", "ts.csv")
        @test isfile(joinpath(p, "out", f))
    end
    si   = readout(p, "solve_info.csv")
    info = Dict(zip(string.(si.name), string.(si.value)))
    @test info["termination_status"] == "OPTIMAL"
    @test info["primal_status"] == "FEASIBLE_POINT"
    for k in ("decarb_version", "git_commit", "julia_version", "solver_name",
              "solver_version", "objective_value", "solve_time_s", "mip_gap",
              "time_limit", "relax_integrality", "duals", "threads",
              "dual_resolve_status", "mip_resolved", "resolved_objective_value",
              "resolved_relative_gap", "resolved_solve_time_s", "utc_timestamp")
        @test haskey(info, k)
    end
    @test info["dual_resolve_status"] == "n/a"
    @test info["mip_resolved"] == "false"
    @test info["resolved_objective_value"] == "n/a"
    @test info["resolved_relative_gap"] == "n/a"
    @test info["resolved_solve_time_s"] == "n/a"
    # balance.csv is written ONLY when the electricity balance fails to close
    @test !isfile(joinpath(p, "out", "balance.csv"))
    @test !isfile(joinpath(p, "out", "iis_constraints.txt"))
end

@testset "minimal :: stale outputs removed, user files kept" begin
    b = baseline()
    p = b.path
    # baseline() pre-writes a stale balance.csv and a user notes.txt before solving
    @test !isfile(joinpath(p, "out", "balance.csv"))
    @test isfile(joinpath(p, "out", "notes.txt"))
    @test b.result.solve_info.termination_status == MOI.OPTIMAL
    @test haskey(b.result.solve_info, :git_commit)
end

@testset "minimal :: time series integrity" begin
    p  = baseline().path
    ts = readout(p, "ts.csv")

    @test nrow(ts) == 24
    @test !any(ismissing, Matrix(ts))
    @test ts.Date[1] == "01/01/2019 01:00"
    @test names(ts)[1] == "Date"
    @test !any(n -> n in ("DateUTC", "DateLocal", "DurationHours"), names(ts))
    eq2 = readout(p, "eq2.csv")
    @test all(==(ts.Date[1]), eq2.Date)
    @test names(eq2) == ["Eq", "Inv", "Qty", "New", "CAPEX", "Date"]

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
    @test all(iszero, ts.bess_SOC)          # guards a BESS state-of-charge regression
    @test all(iszero, ts.WHsoc)             # tankless water heater has no storage

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

# Outdoor temperature contributes in the current period; this case disables
# solar and internal-gain terms. Each period compounds four 15-minute calibration steps.
@testset "thermal recurrence :: current weather, compounded calibration interval" begin
    p = stage(run=false, periods=3, patches = Dict(
        "cfg.csv"    => "pTmode,true" => "pTmode,false",
        "bdg_ii.csv" => "0.03,0.05,0.15,1" => "0.5,0.0,0.0,0.25",
    ))
    @test DECARB.run_decarb!(p; mip_gap=MIP_GAP, time_limit=TLIMIT, verbose=false) === nothing
    ts = readout(p, "ts.csv")
    inputs = load_fixture(joinpath(p, "in"))
    Tout, Tin0, Bk1 = inputs.tm["Tout"], inputs.cfg["Tin0"], inputs.bdg["Bk1"]

    # 15-minute coefficients are compounded four times per one-hour period.
    # Temp[2] depends on Tout[2], so a lagged-weather defect would fail.
    T1 = Tout[1] + (Tin0 - Tout[1]) * (1 - Bk1)^4
    T2 = Tout[2] + (T1 - Tout[2]) * (1 - Bk1)^4
    @test ts.Temp[1] ≈ T1 atol = 0.005 + 1e-6   # half of the 2-digit ts.csv rounding step
    @test ts.Temp[2] ≈ T2 atol = 0.005 + 1e-6
end

# thermal_response_coefficients(bdg, durations) resamples the calibrated step:
#   ratio = durations / Btime;  k1 = 1 - (1 - Bk1)^ratio;  k2,k3 = Bk2,Bk3 * k1 / Bk1
# (or Bk2,Bk3 * ratio when Bk1 == 0, which "temperature validation :: gains and
# calibration" already reaches through check_temperature).
@testset "thermal recurrence :: response coefficients" begin
    coeffs(Bk1, Bk2, Bk3, Btime, TM) = DECARB.thermal_response_coefficients(
        Dict{Any,Any}("Bk1"=>Bk1, "Bk2"=>Bk2, "Bk3"=>Bk3, "Btime"=>Btime), TM)

    # period equals calibration time: ratio 1, the step applies exactly once
    k1, _, _ = coeffs(0.5, 0.0, 0.0, 1.0, [1.0, 1.0])
    @test k1 == [0.5, 0.5]

    # four calibration steps per period: 1 - 0.5^4
    k1, _, _ = coeffs(0.5, 0.0, 0.0, 0.25, [1.0, 1.0])
    @test k1 ≈ [0.9375, 0.9375]

    # per-period ratios 0.25 and 1.75: 1 - 0.5^0.25 and 1 - 0.5^1.75
    k1, _, _ = coeffs(0.5, 0.0, 0.0, 1.0, [0.25, 1.75])
    @test k1 ≈ [0.15910358474628547, 0.7026982211749329]

    # Bk2/Bk3 scale by k1/Bk1; ratios 4 and 2 give k1 = [0.9375, 0.75], scale = [1.875, 1.5]
    k1, k2, k3 = coeffs(0.5, 0.1, 0.2, 0.25, [1.0, 0.5])
    @test k1 ≈ [0.9375, 0.75]
    @test k2 ≈ [0.1875, 0.15]
    @test k3 ≈ [0.375, 0.3]
end

# Active-equipment outputs are modeled as period energy. The thermal
# recurrence must divide that energy by the period duration to recover kW.
@testset "thermal recurrence :: equipment output enters as power" begin
    p = stage(periods=2, patches = Dict(
        "tm.csv" => "2019-01-01T06:00:00" => "2019-01-01T05:15:00",
        "bdg_ii.csv" => "0.03,0.05,0.15,1" => "0.03,0.0,0.15,0.25",
    ))
    ts = readout(p, "ts.csv")

    inputs = load_fixture(joinpath(p, "in"))
    Tin0, Tout1 = inputs.cfg["Tin0"], inputs.tm["Tout"][1]
    Bk1, Bk3, dt = inputs.bdg["Bk1"], inputs.bdg["Bk3"], inputs.tm["TM"][1]
    ihg = ts.IHGp[1] + ts.IHGl[1] + ts.IHGe[1]
    predicted = Tin0 + Bk1*(Tout1 - Tin0) + Bk3*(ts.HVACht[1]/dt + ihg)
    @test ts.HVACht[1] > 0
    @test ts.Temp[1] ≈ predicted atol = 0.02
end

@testset "minimal :: gas path and emissions" begin
    p  = baseline().path
    ts = readout(p, "ts.csv")
    e1 = readout(p, "eq1.csv")

    # gas water heater must consume gas and serve the DHW demand
    @test sum(ts.Gwh)  > 0
    @test sum(ts.WHhw) > 0
    @test sum(ts.HWns) ≈ 0 atol = 1e-6      # demand fully served
    active = ts.WHhw .> 0
    wh = load_fixture(joinpath(p, "in")).wh
    row = findfirst(startswith("WH-TANKLESS-GAS"), wh["ty"])
    @test ts.Gwh[active] ≈ ts.WHhw[active] ./ (wh["eff"][row] * wh["fcf"][row]) atol = 0.02
    @test all(iszero, ts.Gchp) && all(iszero, ts.Gabp)

    val(n) = only(e1.eq[e1.name .== n])
    @test val("direct_emissions")   > 0     # gas combustion
    @test val("indirect_emissions") > 0     # grid imports
    @test val("fuel_purchases")     > 0
end

@testset "water heater :: normalized tank state" begin
    p = stage(patches = Dict(
        "cfg.csv" => ["pWHsto0,0.5" => "pWHsto0,0.8",
                      "pWHstof,0.5" => "pWHstof,0.8"],
        "wh.csv" => "WH-TANKLESS-GAS,12.0,0.82,G,0.95,0.0" =>
                    "WH-TANKLESS-GAS,12.0,0.82,G,0.95,0.5",
    ))
    ts = readout(p, "ts.csv")

    @test all(0 .<= ts.WHsoc .<= 0.5)
    @test ts.WHsoc[end] ≈ 0.4 atol = 0.01
end

@testset "minimal :: equipment report, all brownfield" begin
    p   = baseline().path
    eq2 = readout(p, "eq2.csv")

    @test Set(eq2.Eq) == Set(["HP-MINISPLIT", "WH-TANKLESS-GAS", "PV-MONO"])
    # nothing is newly built: pIT = 0 disables investment
    @test all(iszero, eq2.New)
    @test all(iszero, eq2.CAPEX)
    @test only(eq2.Qty[eq2.Eq .== "PV-MONO"]) == 4
end

@testset "investment :: eq2 identical with duals=false and duals=true" begin
    # duals=true is the shared investment-case solve; solve it again with duals=false
    pf = stage(run=false, patches=INVEST_PATCHES)
    DECARB.run_decarb!(pf; mip_gap=MIP_GAP, time_limit=TLIMIT, duals=false, verbose=false)
    @test isequal(readout(pf, "eq2.csv"), readout(invest_baseline(), "eq2.csv"))
    @test any(>(0), readout(pf, "eq2.csv").New)
end

@testset "minimal :: deselected BESS and wind types are fixed to zero" begin
    r = baseline().result
    m = r.model

    @test all(iszero, r.bess["zmx0"]) && all(iszero, r.wind["zmx0"])
    for s in 1:r.bess["N"], t in 1:r.tm["P"]
        @test is_fixed(m[:vBESSup][t,s]) && value(m[:vBESSup][t,s]) == 0
        @test is_fixed(m[:vBESSdn][t,s]) && value(m[:vBESSdn][t,s]) == 0
    end
    for s in 1:r.bess["N"], t in 0:r.tm["P"]
        @test is_fixed(m[:vBESSsoc][t,s]) && value(m[:vBESSsoc][t,s]) == 0
    end
    for d in 1:r.wind["N"], t in 1:r.tm["P"]
        @test is_fixed(m[:vWINDq][t,d]) && value(m[:vWINDq][t,d]) == 0
    end
end

@testset "chp :: boiler fuel includes heat to absorption chiller" begin
    cfg = Dict{Any,Any}("IT" => 1)
    tm = Dict{Any,Any}("P" => 2, "TM" => [1.0, 1.0], "IW" => Int32[1])
    bdg = Dict{Any,Any}("Bchp" => 1)
    sp = Dict{Any,Any}("CHP0" => ["BOILER"], "CHPz0" => [1], "CHPyn" => ["NO"])
    chp = Dict{Any,Any}("N" => 1, "ty" => ["BOILER-1"], "mx" => [100.0], "mn" => [0.0],
               "fuel" => ["G"], "h2p" => [0.0], "fcf" => [0.9])
    abp = Dict{Any,Any}("N" => 1)
    topo = Dict{Any,Any}("chp_bdg" => zeros(1, 2), "chp_abp" => reshape([0.8], 1, 1))

    model = Model()
    DECARB.heat_connections!(model, topo, tm, chp, abp)
    DECARB.chp_units!(model, cfg, tm, bdg, topo, sp, chp, abp)

    for t in 1:tm["P"]
        @test JuMP.coefficient(model[:vCHP_G][t], model[:vCHPabp][t, 1, 1]) > 0
    end
    @test JuMP.coefficient(model[:vCHP_G][1], model[:vCHPabp][2, 1, 1]) == 0
end

@testset "chp :: heat connection bounds reflect heat-to-power ratio" begin
    tm = Dict{Any,Any}("P" => 1)
    abp = Dict{Any,Any}("N" => 0)
    topo = Dict{Any,Any}("chp_bdg" => [0.0 0.8], "chp_abp" => zeros(1, 0))

    model = Model()
    chp = Dict{Any,Any}("N" => 1, "h2p" => [1.5])
    DECARB.heat_connections!(model, topo, tm, chp, abp)
    @test upper_bound(model[:vCHPbdg][1, 1, 2]) == 1.5

    model = Model()
    chp = Dict{Any,Any}("N" => 1, "h2p" => [0.0])
    DECARB.heat_connections!(model, topo, tm, chp, abp)
    @test upper_bound(model[:vCHPbdg][1, 1, 2]) == 1.0
end

@testset "abp :: CHP heat path uses the same cooling efficiency" begin
    cfg = Dict{Any,Any}("IT" => 1)
    tm = Dict{Any,Any}("P" => 1, "TM" => [1.0], "IW" => Int32[1])
    bdg = Dict{Any,Any}("Babp" => 1)
    sp = Dict{Any,Any}("ABP0" => ["CHILLER"], "ABPz0" => [1], "ABPyn" => ["NO"])
    chp = Dict{Any,Any}("N" => 1, "mx_eff" => [100.0], "h2p" => [1.0])
    abp = Dict{Any,Any}("N" => 1, "ty" => ["CHILLER-1"], "mx" => [50.0], "fuel" => ["G"],
               "ac" => [0.7], "fcf" => [0.9])
    topo = Dict{Any,Any}("chp_bdg" => zeros(1, 2), "chp_abp" => reshape([0.8], 1, 1))

    model = Model()
    DECARB.heat_connections!(model, topo, tm, chp, abp)
    DECARB.abp_chillers!(model, cfg, tm, bdg, topo, sp, chp, abp)

    func = JuMP.constraint_object(model[:eABPq][1, 1]).func
    # fuel path: unit coefficient; CHP path: ac * link efficiency * mx_chp / mx_abp
    @test JuMP.coefficient(func, model[:vABPq][1, 1]) ≈ -1.0
    @test JuMP.coefficient(func, model[:vCHPabp][1, 1, 1]) ≈ -0.7 * 0.8 * 100.0 / 50.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Regression guard: two existing units of one type must yield one row with
# Qty=2, New=0, CAPEX=0.
@testset "multiunit :: no duplicate equipment rows" begin
    p = stage(periods=3, patches = Dict(
        "sp.csv"    => "HP-MINISPLIT,1,NO"  => "HP-MINISPLIT,2,NO",
        "bdg_i.csv" => "pBhvac,1"           => "pBhvac,2",
    ))
    eq2 = readout(p, "eq2.csv")

    hp = eq2[eq2.Eq .== "HP-MINISPLIT", :]
    @test nrow(hp) == 1                       # one row per (type, window)
    @test only(hp.Qty) == 2
    @test only(hp.New) == 0                   # both units pre-existing
    @test only(hp.CAPEX) == 0

    # no (Eq, Date) pair may repeat anywhere in this fixture
    @test nrow(unique(eq2[:, [:Eq, :Date]])) == nrow(eq2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Solve across spring DST on an uninterrupted UTC chronology.
@testset "dst_transition :: spring forward" begin
    patches = Dict(
        "cfg.csv" => "pP0,2019-01-01T05:00:00" => "pP0,2019-03-10T05:00:00",
        "tm.csv"  => ["2019-01-01T" => "2019-03-10T",
                      "2019-01-02T" => "2019-03-11T"],
    )
    ts = readout(stage(; patches, periods=4), "ts.csv")
    @test nrow(ts) == 4
    @test ts.Date[1:3] == ["03/10/2019 01:00", "03/10/2019 03:00", "03/10/2019 04:00"]
end

@testset "dst_transition :: fall back output" begin
    p = stage(run=false, periods=4, patches=Dict(
        "cfg.csv" => "pP0,2019-01-01T05:00:00" => "pP0,2019-11-03T04:00:00",
    ))
    file = joinpath(p, "in", "tm.csv")
    rows = CSV.read(read(file), DataFrame; types=String)
    rows.pP = string.(
        collect(DateTime(2019,11,3,5):Hour(1):DateTime(2019,11,3,8)))
    CSV.write(file, rows)
    result = DECARB.run_decarb!(p; mip_gap=MIP_GAP, time_limit=TLIMIT, verbose=false,
        return_result=true)
    ts = readout(p, "ts.csv")
    @test all(==(1.0), result.tm["TM"])
    @test result.tm["H"] == 4.0
    @test ts.Date[1:2] == ["11/03/2019 01:00", "11/03/2019 01:00"]
    @test length(unique(result.tm["Putc"])) == 4
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "validation :: extended input rules" begin
    base = load_fixture(joinpath(CASE_SRC, "in"))
    function validate_with(mutate)
        inputs = deepcopy(base)
        mutate(inputs)
        io = IOBuffer()
        err = try
            DECARB.validate_inputs(values(inputs)...; io)
            nothing
        catch e
            e
        end
        return err, String(take!(io))
    end
    function rejects(mutate, msg)
        err, _ = validate_with(mutate)
        @test err isa ArgumentError
        @test err isa ArgumentError && occursin(msg, err.msg)
    end
    function warns(mutate, msg)
        err, out = validate_with(mutate)
        @test err === nothing
        @test occursin(msg, out)
    end
    @test validate_with(identity)[1] === nothing

    rejects(i -> i.tm["Qmx"][1] = 1, "pQmx values must lie in 0:pQmxTM")
    warns(i -> (i.cfg["QmxTM"] = 1; i.tm["Qmx"][1:2] .= 1; i.tm["QmxCost"][1:2] = [1.0, 2.0]),
        "pQmxCost is inconsistent within peak group 1")
    rejects(i -> i.tm["EVtype"][1] = 1, "pEVtype contains values above")
    rejects(i -> (i.sp["EVz0"] = [1]; i.tm["EVtype"][1] = 1), "odd number of events for EV 1")
    rejects(i -> i.tm["Ton"][1] = 2, "pTon must be 0 or 1")
    for key in ("Qlight","Qequip","Bppl","Sdni","Sdhi")
        rejects(i -> i.tm[key][1] = -1, "p$(key) must be non-negative")
    end
    rejects(i -> i.sp["PVz0"][1] = 10^6, "existing PV stock")
    rejects(i -> i.sp["BESSz0"] = [1], "existing BESS modules")
    rejects(i -> i.sp["WINDz0"] = [1], "existing wind turbines")
    rejects(i -> (push!(i.sp["PV0"], "PV-MONO"); push!(i.sp["PVz0"], 0)),
        "duplicate equipment: PV-MONO")
    warns(i -> i.topo["up"][1] = "UNKNOWN", "matches no selected catalog equipment")
    rejects(i -> (i.wind["N"] = 1; i.wind["ty"] = ["W"]; i.wind["mn"] = [2.0];
                  i.wind["mx"] = [1.0]; i.wind["vmn"] = [1.0]; i.wind["vmx"] = [2.0];
                  i.wind["life"] = [1.0]; i.wind["inv"] = [0.0]; i.wind["fom"] = [0.0];
                  i.wind["fail"] = [0.0]; i.wind["lc"] = [0.0]),
        "mn must not exceed mx")
    rejects(i -> i.chp["mn"][1] = 1.5, "mn must be in [0,1]")
    rejects(i -> i.chp["h2p"][1] = -1.0, "h2p must be non-negative")
    rejects(i -> i.pv["eff"][1] = 1.5, "eff must be in (0,1]")
    rejects(i -> i.cfg["IT"] = i.tm["P"] + 1, "exceeds the number of periods")
    rejects(i -> i.tm["IW"] = Int32[1, 1], "duplicate start periods")
end

# A name in sp.csv with no matching catalog row must produce a readable
# ArgumentError, not a DimensionMismatch.
@testset "validation :: unknown equipment name" begin
    err = try
        stage(patches = Dict("sp.csv" => "HP-MINISPLIT,1,NO" => "NO-SUCH-UNIT,1,NO"))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("NO-SUCH-UNIT", sprint(showerror, err))
end

# ─────────────────────────────────────────────────────────────────────────────
@testset "validation :: selected equipment with zero capacity" begin
    for (prefix, file) in (("CHP", "chp"), ("ABP", "abp"), ("HVAC", "hvac"),
                           ("WH", "wh"), ("EV", "ev"))
        p = stage(run=false)
        path = joinpath(p, "in")
        catalog_path = joinpath(path, "$file.csv")
        catalog = CSV.read(read(catalog_path), DataFrame)
        selection = CSV.read(read(joinpath(path, "sp.csv")), DataFrame; types=String)
        selection[1, Symbol("p$(prefix)0")] = catalog.ty[1]
        CSV.write(joinpath(path, "sp.csv"), selection)
        for column in (prefix == "HVAC" ? (:pHVmx, :pACmx) : (:mx,))
            catalog[!, column] .= 0
        end
        CSV.write(catalog_path, catalog)
        @test_throws "equipment \"$(catalog.ty[1])\" is selected in sp.csv but absent" DECARB.validate_inputs(values(load_fixture(path))...; io=devnull)
    end
end

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
    @test err isa ArgumentError
    @test occursin("fcf must be in (0,1]", sprint(showerror, err))
end

@testset "validation :: fuel codes are case-sensitive" begin
    fixture = load_fixture(joinpath(CASE_SRC, "in"))
    for name in (:chp, :abp, :wh)
        inputs = deepcopy(fixture)
        catalog = getproperty(inputs, name)
        catalog["fuel"][1] = "g"
        err = try
            DECARB.validate_inputs(values(inputs)...; io=devnull)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test err isa ArgumentError && occursin("$(name).csv", err.msg) &&
            occursin("fuel", err.msg)
    end
end

function ev_unit_model(; kwh, time, soc0=0.5, mnsoc=0.5, v2g=1, up=0.0, dn=10.0)
    P = size(kwh, 1)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    cfg = Dict{Any,Any}("EVsoc0"=>soc0, "EVmnsoc"=>mnsoc, "EVv2g"=>v2g,
        "NSEVcost"=>1.0)
    tm = Dict{Any,Any}("P"=>P, "TM"=>ones(P))
    ev = Dict{Any,Any}("N"=>1, "mx"=>[50.0], "up"=>[up], "dn"=>[dn],
        "effu"=>[1.0], "effd"=>[1.0], "kwh"=>kwh, "time"=>time)
    DECARB.electric_vehicles!(model, cfg, tm, ev)
    return model, ev
end

@testset "EV :: trip shortfall slack makes infeasible trip feasible" begin
    # 40 kWh trip with only 25 kWh stored and no charging available
    kwh = reshape([0.0, 40.0, 0.0], 3, 1)
    time = reshape([1.0, 0.0, 1.0], 3, 1)
    model, _ = ev_unit_model(; kwh, time, mnsoc=0.0)
    @objective(model, Min, sum(model[:vEVtrip][2, 1]))
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    # away-from-home charging: 15 kWh, represented as 15/50 p.u.
    @test value(model[:vEVtrip][2, 1]) ≈ 0.3 atol = 1e-6
    # no slack is needed in periods without a trip
    @test value(model[:vEVtrip][1, 1]) ≈ 0.0 atol = 1e-6
    @test !is_fixed(model[:vEVtrip][2, 1])
    @test is_fixed(model[:vEVtrip][1, 1])
    @test is_fixed(model[:vEVtrip][3, 1])
end

@testset "EV :: zero shortfall cost hardens trip and minimum SOC" begin
    kwh = reshape([0.0, 40.0, 0.0], 3, 1)
    time = reshape([1.0, 0.0, 1.0], 3, 1)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    cfg = Dict{Any,Any}("EVsoc0"=>0.5, "EVmnsoc"=>0.5, "EVv2g"=>1,
        "NSEVcost"=>0.0)
    tm = Dict{Any,Any}("P"=>3, "TM"=>ones(3))
    ev = Dict{Any,Any}("N"=>1, "mx"=>[50.0], "up"=>[0.0], "dn"=>[10.0],
        "effu"=>[1.0], "effd"=>[1.0], "kwh"=>kwh, "time"=>time)
    DECARB.electric_vehicles!(model, cfg, tm, ev)
    chp = Dict{Any,Any}("N"=>0)
    hvac = Dict{Any,Any}("N"=>0)
    wh = Dict{Any,Any}("N"=>0)
    pv = Dict{Any,Any}("N"=>0)
    wind = Dict{Any,Any}("N"=>0)
    bess = Dict{Any,Any}("N"=>0)
    load_tm = merge(tm, Dict{Any,Any}("Qlight"=>zeros(3), "Qequip"=>zeros(3),
        "Qmx"=>zeros(3), "QmxCost"=>zeros(3)))
    load_cfg = merge(cfg, Dict{Any,Any}("NSEcost"=>0.0, "QmxTM"=>0, "QmxBuy"=>100.0,
        "QmxSell"=>100.0))
    DECARB.electric_load!(model, load_cfg, load_tm, chp, hvac, wh, pv, wind, bess, ev;
        io=devnull)
    @test all(is_fixed, model[:vEVtrip])
    @test all(is_fixed, model[:vEVlo])
    @test all(iszero ∘ fix_value, model[:vEVtrip])
    @test all(iszero ∘ fix_value, model[:vEVlo])
end

@testset "EV :: km_to_kwh is clamped at zero" begin
    @test DECARB.km_to_kwh(0.0, 10.0, 40.0) == 0.0
    @test DECARB.km_to_kwh(0.2, 10.0, 0.0) ≈ 2.0
end

# Compare identical installations in two windows for every equipment category.
@testset "fixed O&M :: active investment duration" begin
    for (durations, first_hours, second_hours) in (
        ([1.0, 1.0, 1.0, 1.0], 4.0, 2.0),
        ([0.25, 1.75, 0.5, 1.0], 3.5, 1.5),
    )
        model = Model()
        cfg = Dict{Any,Any}("IT"=>2, "IR"=>0.05, "QmxTM"=>0, "EVdriver"=>0,
            "EVpen"=>0.0, "NSEcost"=>0.0, "NSTcost"=>0.0,
            "NSHWcost"=>0.0, "NSEVcost"=>0.0, "Gco2"=>0.0, "Lco2"=>0.0)
        tm = Dict{Any,Any}("P"=>4, "TM"=>durations, "IW"=>Int32[1, 3],
            "H"=>first_hours, "QmxCostN"=>Float64[])
        for key in ("HWdem", "QcostBuy", "QcostSell", "Gcost", "Lcost", "Qco2")
            tm[key] = zeros(4)
        end
        for key in (:vNSE_Q, :vTup, :vTlo, :vNShw, :vQbuy, :vQsell,
                    :vCHP_G, :vABP_G, :vWH_G, :vCHP_L, :vABP_L, :vWH_L)
            model[key] = @variable(model, [1:4])
        end
        for key in (:vCHP_Q, :vABP_AC, :vHVAC_HT, :vHVAC_AC, :vWH_HW)
            model[key] = @variable(model, [1:4, 1:1])
        end
        model[:vQmx] = @variable(model, [1:0])
        model[:vEVlo] = @variable(model, [1:4, 1:0])
        model[:vEVsoc] = @variable(model, [1:4, 1:0])
        ev = Dict{Any,Any}("N"=>0, "mx_eff"=>Float64[], "kwh"=>zeros(4, 0))
        vars = (:bCHPty, :bABPty, :bHVACty, :bWHty, :zPV, :zWIND, :zBESS)
        catalogs = [Dict{Any,Any}("N"=>1, "fom"=>[100.0*k], "inv"=>[0.0],
            "life"=>[20.0], "vom"=>[0.0]) for k in eachindex(vars)]
        for (k, key) in enumerate(vars)
            catalogs[k]["z0"] = zeros(2, 1)
            if k <= 4
                model[key] = @variable(model, [1:2, 1:1], Bin)
                set_upper_bound.(model[key], 1)
            else
                model[key] = @variable(model, [1:2, 1:1], lower_bound=0, upper_bound=3, integer=true)
            end
        end
        chp, abp, hvac, wh, pv, wind, bess = catalogs
        DECARB.objective_function!(model,cfg,tm,chp,abp,hvac,wh,pv,bess,ev,wind)

        for (key, catalog) in zip(vars, catalogs)
            @testset "$key :: $durations" begin
                w1, w2 = model[key][:, 1]
                qty = is_binary(w1) ? 1 : 2
                first_cost = value(v -> v == w1 ? qty : 0.0, model[:COST_FOM])
                second_cost = value(v -> v == w2 ? qty : 0.0, model[:COST_FOM])
                @test first_cost ≈ only(catalog["fom"])*qty*first_hours/8760
                @test second_cost ≈ only(catalog["fom"])*qty*second_hours/8760
                @test second_cost < first_cost
            end
        end
    end
end

# Existing equipment incurs fixed O&M, but its capital cost is sunk.
@testset "economic summary :: existing capital is sunk" begin
    p  = baseline().path
    e1 = readout(p, "eq1.csv")
    val(n) = only(e1.eq[e1.name .== n])
    @test "total_cost" in e1.name
    @test val("equipment_annuity") ≈ sum(val(n) for n in (
        "chp_annuity", "hvac_annuity", "abp_annuity", "water_heater_annuity",
        "pv_annuity", "wind_annuity", "battery_annuity")) atol = 0.02
    @test val("equipment_annuity") == 0
    @test val("fixed_om_cost") > 0
    inputs = load_fixture(joinpath(p, "in"))
    expected_fom = (sum(inputs.hvac["fom"]) + sum(inputs.wh["fom"]) +
        4*only(inputs.pv["fom"]))*inputs.tm["H"]/8760
    @test val("fixed_om_cost") ≈ expected_fom atol = 0.005
    @test val("equipment_annuity") != val("fixed_om_cost")
end

# A unit selected by the model is new equipment and must incur both reported
# CAPEX and a time-prorated annualized investment cost.
@testset "economic summary :: new equipment pays annuity" begin
    p = invest_baseline()
    e1 = readout(p, "eq1.csv")
    eq2 = readout(p, "eq2.csv")
    val(n) = only(e1.eq[e1.name .== n])
    hp = eq2[eq2.Eq .== "HP-MINISPLIT", :]

    @test nrow(hp) == 1
    @test only(hp.New) == 1
    @test only(hp.CAPEX) > 0
    @test val("hvac_annuity") > 0
    @test val("water_heater_annuity") == 0
    @test val("pv_annuity") == 0
end

# Existing integer-count DER stock belongs only to the first investment
# window; enabling later investment windows must not duplicate it.
@testset "investment windows :: existing DER stock is not repeated" begin
    p = stage(patches = Dict(
        "cfg.csv"   => "pIT,0" => "pIT,2",
        "sp.csv"    => "PV-MONO,4,NO" => "PV-MONO,4,YES",
        "bdg_i.csv" => "pBpv,20.0" => "pBpv,7.2",
    ))
    e1 = readout(p, "eq1.csv")
    eq2 = readout(p, "eq2.csv")
    pv = eq2[eq2.Eq .== "PV-MONO", :]

    @test nrow(pv) == 1
    @test only(pv.Qty) == 4
    @test only(pv.New) == 0
    @test only(pv.CAPEX) == 0
    @test only(e1.eq[e1.name .== "pv_annuity"]) == 0
    inputs = load_fixture(joinpath(p, "in"))
    expected_fom = (sum(inputs.hvac["fom"]) + sum(inputs.wh["fom"]) +
        4*only(inputs.pv["fom"]))*inputs.tm["H"]/8760
    @test only(e1.eq[e1.name .== "fixed_om_cost"]) ≈ expected_fom atol = 0.005
end

@testset "existing stock :: explicit z0 arrays on minimal" begin
    r = baseline().result
    for (d, name) in ((r.chp,"chp"), (r.hvac,"hvac"), (r.abp,"abp"), (r.wh,"wh"),
                      (r.pv,"pv"), (r.wind,"wind"), (r.bess,"bess"))
        @test size(d["z0"]) == (r.cfg["IT"], d["N"])
    end
    @test all(iszero, r.chp["z0"]) && all(iszero, r.abp["z0"])
    @test all(iszero, r.wind["z0"]) && all(iszero, r.bess["z0"])
    @test r.hvac["z0"][1, r.hvac["ty"] .== "HP-MINISPLIT-1"] == [1.0]
    @test sum(r.hvac["z0"]) == 1
    @test r.wh["z0"][1, :] == Float64.(startswith.(r.wh["ty"], "WH-TANKLESS-GAS"))
    @test r.pv["z0"][1, r.pv["ty"] .== "PV-MONO"] == [4.0]
end

@testset "interest rate :: zero rate uses 1/life annuity" begin
    p = stage(run=false, patches=Dict(
        "cfg.csv" => ["pIR,0.06" => "pIR,0.0", INVEST_PATCHES["cfg.csv"]...],
        "sp.csv" => INVEST_PATCHES["sp.csv"]))
    DECARB.validate_inputs(values(load_fixture(joinpath(p, "in")))...; io=devnull)
    r = DECARB.run_decarb!(p; mip_gap=MIP_GAP, time_limit=TLIMIT, verbose=false,
        return_result=true)
    h = findfirst(==("HP-MINISPLIT-1"), r.hvac["ty"])
    @test r.cfg["IR"] == 0
    @test value(r.model[:vHVACinv]) ≈
        r.hvac["inv"][h] / r.hvac["life"][h] * value(r.model[:bHVACty][1,h]) *
        sum(r.tm["TM"]) / 8760
    @test value(r.model[:vHVACinv]) > 0

    inputs = load_fixture(joinpath(p, "in"))
    inputs.cfg["IR"] = -0.01
    @test_throws "[0,1)" DECARB.validate_inputs(values(inputs)...; io=devnull)
    inputs.cfg["IR"] = 1.0
    @test_throws "[0,1)" DECARB.validate_inputs(values(inputs)...; io=devnull)
end

@testset "inputs :: counts above the 8-bit range load and validate" begin
    p = stage(run=false, patches=Dict(
        "bdg_i.csv" => ["pBbess,0" => "pBbess,500", "pBpv,20.0" => "pBpv,2000.0"],
        "sp.csv"    => "PV-MONO,4,NO" => "PV-MONO,300,NO",
    ))
    inputs = load_fixture(joinpath(p, "in"))
    @test inputs.sp["PVz0"] == [300]
    @test inputs.bdg["Bbess"] == 500
    @test DECARB.validate_inputs(values(inputs)...; io=devnull) === nothing
end

@testset "validate_inputs :: EV feasibility precheck" begin
    tm = Dict("P" => 3, "TM" => [1.0, 1.0, 1.0],
        "Date" => [DateTime(2024, 1, 1, h) for h in 0:2])
    ev = Dict("N" => 1, "ty" => ["EV-1"], "mx" => [50.0], "up" => [7.0],
        "effu" => [0.9], "time" => zeros(3, 1), "kwh" => [0.0, 20.0, 0.0])
    cfg = Dict("EVsoc0" => 0.2, "EVmnsoc" => 0.5, "NSEVcost" => 0.0)

    err = String[]
    DECARB.check_ev_feasibility!(err, cfg, tm, ev)   # never plugged in: skipped
    @test isempty(err)

    ev["time"][1, 1] = 1                             # one plugged period
    ev["up"] = [0.5]
    err = String[]
    DECARB.check_ev_feasibility!(err, cfg, tm, ev)
    @test length(err) == 1
    @test occursin("set pNSEVcost > 0", err[1])
    @test_throws ArgumentError throw(ArgumentError(join(err, "\n")))

    cfg["NSEVcost"] = 1.0
    err = String[]
    DECARB.check_ev_feasibility!(err, cfg, tm, ev)
    @test isempty(err)

    cfg["NSEVcost"] = 0.0
    ev["up"] = [50.0]                                # fast enough: feasible
    err = String[]
    DECARB.check_ev_feasibility!(err, cfg, tm, ev)
    @test isempty(err)
end

function eq1_metrics(path)
    df = CSV.read(joinpath(path, "out", "eq1.csv"), DataFrame)
    d  = Dict(string(n) => Float64(v) for (n, v) in zip(df.name, df.eq))
    d["emissions"] = d["direct_emissions"] + d["indirect_emissions"]
    return d
end

@testset "regression :: minimal eq1.csv matches committed fixture" begin
    got, ref = eq1_metrics(baseline().path), eq1_metrics(CASE_SRC)
    rtol = max(1e-3, 2 * REGRESSION_GAP)   # never tighter than the solver gap allows
    for (key, atol) in REGRESSION_KEYS
        @test isapprox(got[key], ref[key]; rtol, atol)
    end
end

FULL && @testset "Aqua.jl" begin
    Aqua.test_all(DECARB;
        ambiguities = true,
        unbound_args = true,
        stale_deps = true,
        deps_compat = true,
        undefined_exports = false,
        project_extras = false,
        piracies = false,
        persistent_tasks = false,
    )
end

end # DECARB.jl

# Opt-in local JET checks: DECARB_JET=true julia --project=. -e 'using Pkg; Pkg.test()'
get(ENV, "DECARB_JET", "") == "true" && include("jet.jl")
