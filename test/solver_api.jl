# A minimal recording optimizer exercises solver configuration without requiring
# a proprietary solver or accepting any vendor-specific attributes by accident.
struct RecordingOptimizer <: MOI.AbstractOptimizer
    name::String
    attributes::Dict{MOI.AbstractOptimizerAttribute,Any}
end
RecordingOptimizer(name) = RecordingOptimizer(name, Dict{MOI.AbstractOptimizerAttribute,Any}())
MOI.is_empty(::RecordingOptimizer) = true
MOI.get(opt::RecordingOptimizer, ::MOI.SolverName) = opt.name
MOI.supports(::RecordingOptimizer, ::MOI.AbstractOptimizerAttribute) = true
MOI.set(opt::RecordingOptimizer, attr::MOI.AbstractOptimizerAttribute, value) =
    (opt.attributes[attr] = value)
MOI.get(opt::RecordingOptimizer, attr::MOI.AbstractOptimizerAttribute) = opt.attributes[attr]

@testset "solver API :: generic MOI defaults and Gurobi tuning" begin
    generic = RecordingOptimizer("Industrial MIP")
    DECARB.configure_solver!(direct_model(generic), 1//1000, 60)
    @test generic.attributes == Dict(
        MOI.RelativeGapTolerance() => 0.001,
        MOI.TimeLimitSec() => 60.0,
    )

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
                mip_gap=0.1, time_limit=120, duals=false, verbose=false)
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
        relax_integrality=true, verbose=false)
    @test get_attribute(result.model, "presolve") == "on"
    @test get_attribute(result.model, "mip_rel_gap") == 1e-4
    @test !any(v -> is_binary(v) || is_integer(v), all_variables(result.model))
    @test has_duals(result.model)
    @test any(!iszero, readout(p, "ts.csv").dualQ)
    disabled_duals, _ = DECARB.read_econ(result.model, result.tm; duals=false)
    @test all(iszero, Matrix(disabled_duals))
end

@testset "solver API :: optional dual re-solve" begin
    for (relax, duals, expected_solves) in ((false,false,1), (false,true,2),
                                          (true,false,1), (true,true,1))
        model = Model(HiGHS.Optimizer)
        # Keep the thread count consistent with the process-wide HiGHS scheduler
        # initialized by the preceding case runs.
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
        DECARB.solve_model!(tempdir(), model, relax; duals, verbose=false)
        @test calls[] == expected_solves
        @test value(x) ≈ (relax ? 0.5 : 1.0)
        @test is_binary(x) == (!relax && !duals)
        @test has_duals(model) == (relax || duals)
    end
end
