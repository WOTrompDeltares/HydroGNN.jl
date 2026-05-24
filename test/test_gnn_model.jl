using Test
using HydroGNN
using TOML, JLD2, Flux
using GraphNeuralNetworks

const TEST_DATA = joinpath(@__DIR__, "..", "test_data", "lake1d_surge", "train.jld2")
const TMPDIR    = mktempdir()

@testset "ModelSettings defaults and field assignment" begin
    ms = ModelSettings()
    @test ms.dhidden          == 32
    @test ms.nhidden          == 5
    @test ms.skip_connections == false
    @test ms.din              == 0
    @test ms.dout             == 0

    ms.dhidden          = 16
    ms.nhidden          = 3
    ms.skip_connections = true
    ms.din              = 10
    ms.dout             = 2
    @test ms.dhidden          == 16
    @test ms.nhidden          == 3
    @test ms.skip_connections == true
    @test ms.din              == 10
    @test ms.dout             == 2
end

@testset "ModelSettings TOML round-trip" begin
    ms = ModelSettings()
    ms.dhidden = 16; ms.nhidden = 3; ms.skip_connections = true; ms.din = 10; ms.dout = 2

    toml_path = joinpath(TMPDIR, "model_settings.toml")
    save_model_settings(ms, toml_path)
    @test isfile(toml_path)

    ms2 = load_model_settings(toml_path)
    @test ms2.dhidden          == ms.dhidden
    @test ms2.nhidden          == ms.nhidden
    @test ms2.skip_connections == ms.skip_connections
    @test ms2.din              == ms.din
    @test ms2.dout             == ms.dout
end

@testset "GNN forward pass" begin
    norm_strategy = GlobalNorm(compute_norm_stats(TEST_DATA))
    graphs, _     = load_data(TEST_DATA, norm_strategy)
    g             = graphs[1]

    din  = size(g.ndata.dynamic, 1) + size(g.ndata.forcing, 1) + size(g.ndata.static, 1)
    dout = size(g.ndata.dynamic, 1)

    # Without skip connections
    model = GNN(din, 16, dout, 2; skip=false)
    out   = model(g)
    @test size(out) == size(g.ndata.dynamic)

    # With skip connections
    model_skip = GNN(din, 16, dout, 2; skip=true)
    out_skip   = model_skip(g)
    @test size(out_skip) == size(g.ndata.dynamic)
end

@testset "GNN JLD2 save and reload" begin
    norm_strategy = GlobalNorm(compute_norm_stats(TEST_DATA))
    graphs, _     = load_data(TEST_DATA, norm_strategy)
    g             = graphs[1]

    din  = size(g.ndata.dynamic, 1) + size(g.ndata.forcing, 1) + size(g.ndata.static, 1)
    dout = size(g.ndata.dynamic, 1)

    model      = GNN(din, 16, dout, 2; skip=false)
    out_before = model(g)

    model_path = joinpath(TMPDIR, "model.jld2")
    jldsave(model_path; model=model, norm_strategy=norm_strategy, train_strategy=NoNoise())
    @test isfile(model_path)

    d            = JLD2.load(model_path)
    model_loaded = d["model"]
    out_after    = model_loaded(g)

    @test out_before ≈ out_after
end
