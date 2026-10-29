using Test
using JET
using Dates
using DECARB

# JET added JET_AVAILABLE after earlier releases compatible with Julia 1.10.
jet_available = if isdefined(JET, :JET_AVAILABLE)
    JET.JET_AVAILABLE
elseif isdefined(JET, :JET_LOADABLE)
    JET.JET_LOADABLE
else
    false
end

if jet_available
    @testset "JET targeted helper calls" begin
        @test_call DECARB.investment_windows(2, 4.0, [1.0, 1.0, 1.0, 1.0])

        bdg = Dict(
            "Btime" => 1.0,
            "Bk1" => 0.2,
            "Bk2" => 0.05,
            "Bk3" => 0.15,
        )
        @test_call DECARB.thermal_response_coefficients(bdg, [0.5, 1.0, 2.0])

        start = DateTime(2024, 1, 1)
        periods = [start + Hour(1), start + Hour(2)]
        @test_call DECARB.validate_tm_timestamps(start, periods)
    end
else
    @info "Skipping targeted JET checks: JET is unavailable for Julia $(VERSION)"
end
