using Test
using HydroGNN
using Flux
using GraphNeuralNetworks

# ─── Helpers ──────────────────────────────────────────────────────────────────

const T_NNODES  = 4
const T_NDYN    = 2
const T_NSTATIC = 3

function _ring_edges(n=T_NNODES)
    src = Int64.(1:n)
    dst = Int64.(mod.(1:n, n) .+ 1)
    return vcat(src, dst), vcat(dst, src)   # bidirectional
end

# Single-step batch: (x_graph, y_graph). y has plain matrix ndata (accessed as y.x).
function _single_step_batch(; nnodes=T_NNODES, ndyn=T_NDYN, nstatic=T_NSTATIC)
    src, dst = _ring_edges(nnodes)
    x = GNNGraph(src, dst,
            ndata=(; static  = randn(Float32, nstatic, nnodes),
                     dynamic = randn(Float32, ndyn,    nnodes),
                     forcing = zeros(Float32, 0,       nnodes)))
    y = GNNGraph(src, dst, ndata=randn(Float32, ndyn, nnodes))
    return x, y
end

# Multi-step sequence of length nsteps+1, all with NamedTuple ndata.
function _multistep_batch(nsteps; nnodes=T_NNODES, ndyn=T_NDYN, nstatic=T_NSTATIC)
    src, dst = _ring_edges(nnodes)
    static  = randn(Float32, nstatic, nnodes)
    forcing = zeros(Float32, 0, nnodes)
    [GNNGraph(src, dst,
              ndata=(; static=static,
                       dynamic=randn(Float32, ndyn, nnodes),
                       forcing=forcing))
     for _ in 1:(nsteps + 1)]
end

# Tiny GNN model matching the helper graph dimensions (no forcing).
_small_model(; ndyn=T_NDYN, nstatic=T_NSTATIC) = GNN(ndyn + nstatic, 8, ndyn, 2; skip=false)

# ─── TrainSettings ────────────────────────────────────────────────────────────

@testset "TrainSettings defaults" begin
    s = TrainSettings()
    @test s.nepochs  == 250
    @test s.lr       ≈  3e-3
    @test s.lr_final ≈  1e-5
    @test s.lr_step  == 10
    @test s.nbatch   == 32
end

@testset "train_settings_from_toml full dict" begin
    d = Dict{String,Any}(
        "nepochs"         => 100,
        "lr"              => 1e-3,
        "lr_final"        => 1e-6,
        "lr_step"         => 5,
        "nbatch"          => 16,
        "train_data_path" => "data/train.jld2",
        "valid_data_path" => "data/valid.jld2",
        "model_name"      => "myrun",
        "save_dir"        => "models",
    )
    s = train_settings_from_toml(d)
    @test s.nepochs         == 100
    @test s.lr              ≈  1e-3
    @test s.lr_final        ≈  1e-6
    @test s.lr_step         == 5
    @test s.nbatch          == 16
    @test s.train_data_path == "data/train.jld2"
    @test s.model_name      == "myrun"
end

@testset "train_settings_from_toml partial dict preserves defaults" begin
    s   = train_settings_from_toml(Dict{String,Any}("nepochs" => 50))
    ref = TrainSettings()
    @test s.nepochs == 50
    @test s.lr      ≈ ref.lr
    @test s.nbatch  == ref.nbatch
end

# ─── strategy_to_dict ─────────────────────────────────────────────────────────

@testset "strategy_to_dict for all TrainStrategy types" begin
    @test strategy_to_dict(NoNoise())["name"] == "NoNoise"

    d_ssn = strategy_to_dict(SingleStepNoise(0.05))
    @test d_ssn["name"]  == "SingleStepNoise"
    @test d_ssn["scale"] ≈  0.05

    d_msr = strategy_to_dict(MultiStepRollout(3, 0.02))
    @test d_msr["name"]        == "MultiStepRollout"
    @test d_msr["nsteps"]      == 3
    @test d_msr["noise_scale"] ≈  0.02

    d_pfr = strategy_to_dict(PushforwardRollout(4, 0.0))
    @test d_pfr["name"]   == "PushforwardRollout"
    @test d_pfr["nsteps"] == 4

    d_sr = strategy_to_dict(ScheduledRollout(identity, 5, 0.01))
    @test d_sr["name"]        == "ScheduledRollout"
    @test d_sr["nsteps"]      == 5
    @test d_sr["noise_scale"] ≈  0.01

    d_sp = strategy_to_dict(ScheduledPushforward(identity, 6, 0.0))
    @test d_sp["name"]   == "ScheduledPushforward"
    @test d_sp["nsteps"] == 6
end

# ─── step_schedule! ───────────────────────────────────────────────────────────

@testset "step_schedule! is a no-op for NoNoise and SingleStepNoise" begin
    HydroGNN.step_schedule!(NoNoise(),            1)   # must not error
    HydroGNN.step_schedule!(SingleStepNoise(0.1), 5)   # must not error
    @test true
end

@testset "step_schedule! advances ScheduledRollout current_nsteps" begin
    sr = ScheduledRollout(identity, 10, 0.0)
    @test sr.current_nsteps == 1
    HydroGNN.step_schedule!(sr, 4)
    @test sr.current_nsteps == 4
    HydroGNN.step_schedule!(sr, 20)     # capped at nsteps=10
    @test sr.current_nsteps == 10
end

@testset "step_schedule! advances ScheduledPushforward current_nsteps" begin
    sp = ScheduledPushforward(identity, 8, 0.0)
    @test sp.current_nsteps == 1
    HydroGNN.step_schedule!(sp, 3)
    @test sp.current_nsteps == 3
    HydroGNN.step_schedule!(sp, 100)    # capped at nsteps=8
    @test sp.current_nsteps == 8
end

# ─── compute_loss: single-step strategies ─────────────────────────────────────

@testset "compute_loss NoNoise: matches direct MSE, noiseless==loss" begin
    model    = _small_model()
    x, y     = _single_step_batch()
    expected = Flux.mse(model(x), y.x)

    loss, grad, noiseless = HydroGNN.compute_loss(NoNoise(), model, (x, y), Flux.cpu)

    @test loss      isa AbstractFloat
    @test loss      >= 0
    @test loss      ≈  expected
    @test noiseless ≈  loss        # NoNoise: noiseless_loss == batch_loss
    @test grad      !== nothing
end

@testset "compute_loss SingleStepNoise scale=0: noiseless≈batch loss" begin
    model                 = _small_model()
    x, y                  = _single_step_batch()
    loss, grad, noiseless = HydroGNN.compute_loss(SingleStepNoise(0.0), model, (x, y), Flux.cpu)

    @test loss      >= 0
    @test noiseless >= 0
    @test grad      !== nothing
    @test noiseless ≈ loss   # zero noise → dynamic unchanged
end

@testset "compute_loss SingleStepNoise scale>0: valid loss and gradients" begin
    model                 = _small_model()
    x, y                  = _single_step_batch()
    loss, grad, noiseless = HydroGNN.compute_loss(SingleStepNoise(1.0), model, (x, y), Flux.cpu)

    @test loss      >= 0
    @test noiseless >= 0
    @test grad      !== nothing
end

# ─── compute_loss: multi-step rollout strategies ──────────────────────────────

@testset "compute_loss MultiStepRollout nsteps=1 equals single-step MSE" begin
    model = _small_model()
    x_seq = _multistep_batch(1)
    strat = MultiStepRollout(1, 0.0)

    loss, grad, noiseless = HydroGNN.compute_loss(strat, model, x_seq, Flux.cpu)

    expected = Flux.mse(model(x_seq[1]), x_seq[2].ndata.dynamic)
    @test loss      ≈ expected
    @test noiseless ≈ expected
    @test grad !== nothing
end

@testset "compute_loss MultiStepRollout nsteps=2: valid loss and gradients" begin
    model = _small_model()
    x_seq = _multistep_batch(2)
    strat = MultiStepRollout(2, 0.0)

    loss, grad, noiseless = HydroGNN.compute_loss(strat, model, x_seq, Flux.cpu)

    @test loss      >= 0
    @test noiseless >= 0
    @test grad !== nothing
end

@testset "compute_loss PushforwardRollout nsteps=1 equals single-step MSE" begin
    model = _small_model()
    x_seq = _multistep_batch(1)
    strat = PushforwardRollout(1, 0.0)

    loss, grad, noiseless = HydroGNN.compute_loss(strat, model, x_seq, Flux.cpu)

    # nsteps=1: warm-up loop is skipped; gradient step uses x_seq[end-1]→x_seq[end]
    expected = Flux.mse(model(x_seq[1]), x_seq[2].ndata.dynamic)
    @test loss      ≈ expected
    @test noiseless ≈ expected
    @test grad !== nothing
end

@testset "compute_loss PushforwardRollout nsteps=3: valid loss and gradients" begin
    model = _small_model()
    x_seq = _multistep_batch(3)
    strat = PushforwardRollout(3, 0.0)

    loss, grad, noiseless = HydroGNN.compute_loss(strat, model, x_seq, Flux.cpu)

    @test loss      >= 0
    @test noiseless >= 0
    @test grad !== nothing
end

@testset "compute_loss ScheduledRollout respects current_nsteps" begin
    model = _small_model()
    strat = ScheduledRollout(identity, 3, 0.0)   # starts at current_nsteps=1
    x_seq = _multistep_batch(3)                  # length 4

    loss1, grad1, _ = HydroGNN.compute_loss(strat, model, x_seq, Flux.cpu)
    @test loss1 >= 0
    @test grad1 !== nothing

    HydroGNN.step_schedule!(strat, 2)
    @test strat.current_nsteps == 2

    loss2, grad2, _ = HydroGNN.compute_loss(strat, model, x_seq, Flux.cpu)
    @test loss2 >= 0
    @test grad2 !== nothing
end

@testset "compute_loss ScheduledPushforward respects current_nsteps" begin
    model = _small_model()
    strat = ScheduledPushforward(identity, 3, 0.0)   # starts at current_nsteps=1
    x_seq = _multistep_batch(3)                       # length 4

    loss1, grad1, _ = HydroGNN.compute_loss(strat, model, x_seq, Flux.cpu)
    @test loss1 >= 0
    @test grad1 !== nothing

    HydroGNN.step_schedule!(strat, 2)
    @test strat.current_nsteps == 2

    loss2, grad2, _ = HydroGNN.compute_loss(strat, model, x_seq, Flux.cpu)
    @test loss2 >= 0
    @test grad2 !== nothing
end
