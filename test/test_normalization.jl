using Test
using HydroGNN
using Statistics

const NORM_TEST_DATA = joinpath(@__DIR__, "..", "data", "lake1d_surge", "train.jld2")

@testset "DynamicNormStats construction" begin
    stats = DynamicNormStats(Float32[1.0, 2.0], Float32[0.5, 1.0])
    @test stats.mu    == Float32[1.0, 2.0]
    @test stats.sigma == Float32[0.5, 1.0]
end

@testset "strategy_to_dict" begin
    gn = GlobalNorm(DynamicNormStats(Float32[0f0], Float32[1f0]))
    @test strategy_to_dict(gn)["name"] == "GlobalNorm"

    ptn = PerTrajectoryNorm()
    @test strategy_to_dict(ptn)["name"] == "PerTrajectoryNorm"
end

@testset "_norm scalar broadcast" begin
    x = Float32[2.0, 4.0, 6.0]
    @test HydroGNN._norm(x, 4.0f0, 2.0f0) ≈ Float32[-1.0, 0.0, 1.0]
end

@testset "_norm matrix broadcast" begin
    X = Float32[0.0 2.0; 4.0 6.0]
    @test HydroGNN._norm(X, 3.0f0, 2.0f0) ≈ Float32[-1.5 -0.5; 0.5 1.5]
end

@testset "_compute_traj_norm_stats without tau" begin
    waterlevel = Float32[0.0 1.0; 2.0 3.0]   # values: 0,1,2,3
    velocity   = Float32[1.0 2.0; 3.0 4.0]   # values: 1,2,3,4

    s = HydroGNN._compute_traj_norm_stats(velocity, waterlevel)

    @test length(s.mu)    == 2
    @test length(s.sigma) == 2
    @test s.mu[1]    ≈ mean(Float32[0, 1, 2, 3])
    @test s.sigma[1] ≈ std(Float32[0, 1, 2, 3])
    @test s.mu[2]    ≈ mean(Float32[1, 2, 3, 4])
    @test s.sigma[2] ≈ std(Float32[1, 2, 3, 4])
end

@testset "_compute_traj_norm_stats with tau" begin
    waterlevel = Float32[0.0 1.0; 2.0 3.0]
    velocity   = Float32[1.0 2.0; 3.0 4.0]
    tau        = Float32[5.0 6.0; 7.0 8.0]   # values: 5,6,7,8

    s = HydroGNN._compute_traj_norm_stats(velocity, waterlevel, tau)

    @test length(s.mu)    == 3
    @test length(s.sigma) == 3
    @test s.mu[3]    ≈ mean(Float32[5, 6, 7, 8])
    @test s.sigma[3] ≈ std(Float32[5, 6, 7, 8])
end

@testset "sigma floor: constant signal clamped to 1e-8" begin
    waterlevel = fill(5.0f0, 4, 3)
    velocity   = fill(2.0f0, 4, 3)

    s = HydroGNN._compute_traj_norm_stats(velocity, waterlevel)

    @test all(s.sigma .>= 1f-8)
end

@testset "resolve_stats: GlobalNorm ignores input data" begin
    fixed = DynamicNormStats(Float32[10.0, 20.0], Float32[2.0, 3.0])
    gn    = GlobalNorm(fixed)

    # Pass data with entirely different statistics — result must still be `fixed`
    velocity   = rand(Float32, 5, 4) .* 1000f0
    waterlevel = rand(Float32, 5, 4) .* 1000f0

    s = HydroGNN.resolve_stats(gn, velocity, waterlevel)
    @test s.mu    == fixed.mu
    @test s.sigma == fixed.sigma
end

@testset "resolve_stats: PerTrajectoryNorm computes from data" begin
    waterlevel = Float32[0.0 1.0; 2.0 3.0]
    velocity   = Float32[10.0 20.0; 30.0 40.0]

    s = HydroGNN.resolve_stats(PerTrajectoryNorm(), velocity, waterlevel)

    @test s.mu[1] ≈ mean(Float32[0, 1, 2, 3])
    @test s.mu[2] ≈ mean(Float32[10, 20, 30, 40])
end

@testset "resolve_stats: PerTrajectoryNorm with tau" begin
    waterlevel = Float32[0.0 1.0; 2.0 3.0]
    velocity   = Float32[1.0 2.0; 3.0 4.0]
    tau        = Float32[5.0 6.0; 7.0 8.0]

    s = HydroGNN.resolve_stats(PerTrajectoryNorm(), velocity, waterlevel, tau)

    @test length(s.mu) == 3
    @test s.mu[3] ≈ mean(Float32[5, 6, 7, 8])
end

@testset "GlobalNorm vs PerTrajectoryNorm diverge on shifted trajectory" begin
    wl_ref  = rand(Float32, 20, 10)
    vel_ref = rand(Float32, 20, 10)
    global_stats = HydroGNN._compute_traj_norm_stats(vel_ref, wl_ref)
    gn = GlobalNorm(global_stats)

    # Shift trajectory far from the global mean
    wl_shifted  = wl_ref  .+ 100f0
    vel_shifted = vel_ref .+ 100f0

    s_global = HydroGNN.resolve_stats(gn,                  vel_shifted, wl_shifted)
    s_per    = HydroGNN.resolve_stats(PerTrajectoryNorm(), vel_shifted, wl_shifted)

    # Per-trajectory norm tracks the shift; global norm does not
    @test abs(s_per.mu[1]    - mean(vec(wl_shifted))) < 1f-2
    @test abs(s_global.mu[1] - mean(vec(wl_shifted))) > 50f0
end

@testset "compute_norm_stats on real data" begin
    stats = compute_norm_stats(NORM_TEST_DATA)
    @test length(stats.mu)    >= 2
    @test length(stats.sigma) >= 2
    @test all(stats.sigma .> 0)
    @test all(isfinite.(stats.mu))
    @test all(isfinite.(stats.sigma))
end
