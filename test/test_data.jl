using Test
using HydroGNN
using GraphNeuralNetworks
using Statistics
using Flux: DataLoader

const DATA_TEST_FILE = joinpath(@__DIR__, "..", "test_data", "lake1d_surge", "train.jld2")

# ─── read_trajectory ──────────────────────────────────────────────────────────

@testset "read_trajectory returns correct types and shapes" begin
    velocity, waterlevel, mesh_pos, node_type, bathymetry, edges, tau =
        HydroGNN.read_trajectory(DATA_TEST_FILE, 1)

    @test velocity   isa Matrix
    @test waterlevel isa Matrix
    @test mesh_pos   isa Vector
    @test node_type  isa Vector
    @test bathymetry isa Vector
    @test edges      isa Matrix
    @test size(edges, 1) == 2                          # rows = [src; dst]
    @test size(velocity, 1) == size(waterlevel, 1) == length(mesh_pos)
end

@testset "read_trajectory edges are bidirected" begin
    _, _, _, _, _, edges, _ = HydroGNN.read_trajectory(DATA_TEST_FILE, 1)
    n = size(edges, 2)
    @test iseven(n)                                    # forward + reverse pairs
    src, dst = edges[1, :], edges[2, :]
    # Every edge (s,t) must have a corresponding reverse edge (t,s)
    all_edges = Set(zip(src, dst))
    @test all(((t, s),) -> (t, s) in all_edges, zip(src, dst))
end

# ─── load_data ────────────────────────────────────────────────────────────────

@testset "load_data with GlobalNorm: structure" begin
    norm = GlobalNorm(compute_norm_stats(DATA_TEST_FILE))
    xs, ys = load_data(DATA_TEST_FILE, norm)

    @test length(xs) == length(ys)
    @test length(xs) > 0
    @test xs[1] isa GNNGraph
    @test ys[1] isa GNNGraph
end

@testset "load_data with GlobalNorm: feature shapes" begin
    norm = GlobalNorm(compute_norm_stats(DATA_TEST_FILE))
    xs, _ = load_data(DATA_TEST_FILE, norm)
    g = xs[1]

    @test size(g.ndata.dynamic, 1)  == 2    # waterlevel + velocity
    @test size(g.ndata.static, 1)   == 8    # bath + pos + 6 one-hot
    @test size(g.ndata.forcing, 1)  == 1    # tau present in lake1d_surge
    @test eltype(g.ndata.dynamic)   == Float32
    @test eltype(g.ndata.static)    == Float32
end

@testset "load_data with PerTrajectoryNorm: same shapes" begin
    xs, _ = load_data(DATA_TEST_FILE, PerTrajectoryNorm())
    g = xs[1]

    @test size(g.ndata.dynamic, 1) == 2
    @test size(g.ndata.static, 1)  == 8
    @test size(g.ndata.forcing, 1) == 1
end

# ─── load_data_multistep ──────────────────────────────────────────────────────

@testset "load_data_multistep: window length equals nsteps+1" begin
    norm    = GlobalNorm(compute_norm_stats(DATA_TEST_FILE))
    windows = load_data_multistep(DATA_TEST_FILE, norm, 3)

    @test length(windows) > 0
    @test all(length(w) == 4 for w in windows)         # 3 + 1
end

@testset "load_data_multistep: nsteps=1 window count matches single-step count" begin
    norm      = GlobalNorm(compute_norm_stats(DATA_TEST_FILE))
    xs_ss, _  = load_data(DATA_TEST_FILE, norm)
    windows1  = load_data_multistep(DATA_TEST_FILE, norm, 1)

    @test length(windows1) == length(xs_ss)
end

@testset "load_data_multistep: graphs within a window share topology" begin
    norm    = GlobalNorm(compute_norm_stats(DATA_TEST_FILE))
    windows = load_data_multistep(DATA_TEST_FILE, norm, 3)
    w = windows[1]

    @test all(g.num_nodes == w[1].num_nodes for g in w)
    @test all(g.num_edges == w[1].num_edges for g in w)
end

@testset "load_data_multistep: static features identical across steps" begin
    norm    = GlobalNorm(compute_norm_stats(DATA_TEST_FILE))
    windows = load_data_multistep(DATA_TEST_FILE, norm, 3)
    w = windows[1]

    for k in 2:length(w)
        @test w[k].ndata.static == w[1].ndata.static
    end
end

@testset "load_data_multistep: dynamic features differ across steps" begin
    norm    = GlobalNorm(compute_norm_stats(DATA_TEST_FILE))
    windows = load_data_multistep(DATA_TEST_FILE, norm, 3)
    w = windows[1]

    @test w[2].ndata.dynamic != w[1].ndata.dynamic
end

# ─── collate_multistep_batch ──────────────────────────────────────────────────

function _make_window(nsteps, nnodes, nedges, tag)
    s = collect(1:nedges)
    t = [mod1(i + 1, nnodes) for i in 1:nedges]
    [GNNGraph(s, t; num_nodes=nnodes,
              ndata=(; dynamic=fill(Float32(tag * 10 + k), 2, nnodes),
                       static=ones(Float32, 8, nnodes)))
     for k in 1:(nsteps + 1)]
end

@testset "collate_multistep_batch: output length equals nsteps+1" begin
    B, nsteps, N, E = 3, 4, 10, 10
    windows = [_make_window(nsteps, N, E, b) for b in 1:B]
    result  = collate_multistep_batch(windows)

    @test length(result) == nsteps + 1
end

@testset "collate_multistep_batch: batched graph has B×N nodes and B graphs" begin
    B, nsteps, N, E = 3, 4, 10, 10
    windows = [_make_window(nsteps, N, E, b) for b in 1:B]
    result  = collate_multistep_batch(windows)

    @test result[1].num_nodes  == B * N
    @test result[1].num_graphs == B
end

@testset "collate_multistep_batch: feature arrays concatenated over node dim" begin
    B, nsteps, N, E = 3, 4, 10, 10
    windows = [_make_window(nsteps, N, E, b) for b in 1:B]
    result  = collate_multistep_batch(windows)

    @test size(result[1].ndata.dynamic, 2) == B * N
    @test size(result[1].ndata.static,  2) == B * N
end

@testset "collate_multistep_batch: each step carries correct per-window data" begin
    B, nsteps, N, E = 2, 2, 5, 5
    windows = [_make_window(nsteps, N, E, b) for b in 1:B]
    result  = collate_multistep_batch(windows)

    # Window b at step k was filled with Float32(b*10 + k).
    # After batching, window b occupies columns ((b-1)*N+1):(b*N).
    for k in 1:(nsteps + 1)
        for b in 1:B
            expected   = Float32(b * 10 + k)
            node_slice = ((b - 1) * N + 1):(b * N)
            @test all(result[k].ndata.dynamic[:, node_slice] .== expected)
        end
    end
end

@testset "collate_multistep_batch: B=1 passes through unchanged" begin
    nsteps, N, E = 3, 8, 8
    window = _make_window(nsteps, N, E, 1)
    result = collate_multistep_batch([window])

    @test length(result) == nsteps + 1
    @test result[1].num_nodes == N
    @test result[1].ndata.dynamic == window[1].ndata.dynamic
end

@testset "collate_multistep_batch: works via DataLoader with real data" begin
    norm    = GlobalNorm(compute_norm_stats(DATA_TEST_FILE))
    windows = load_data_multistep(DATA_TEST_FILE, norm, 3)
    dl      = DataLoader(windows, batchsize=4, shuffle=false,
                         collate=collate_multistep_batch)

    batch = first(dl)
    @test batch isa Vector{<:GNNGraph}
    @test length(batch) == 4                            # nsteps + 1
    @test batch[1].num_graphs == min(4, length(windows))
end
