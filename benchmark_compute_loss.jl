cd(@__DIR__)
using Pkg
Pkg.activate(".")

using CUDA
using Flux
using Flux: DataLoader
using HydroGNN
using Statistics
using Printf

# ─── Configuration ────────────────────────────────────────────────────────────

const DATA_FILE  = "test_data/lake1d_surge/train.jld2"
const NWARMUP    = 3       # full-epoch passes to discard (JIT warm-up)
const NMEASURE   = 10      # full-epoch passes to time
const NSTEPS_PFR = 3       # pushforward rollout depth

const DEVICE = Flux.gpu
CUDA.functional() || @warn "CUDA not available — falling back to CPU"

# ─── Data & model ─────────────────────────────────────────────────────────────

println("Loading data...")
norm_strategy = GlobalNorm(compute_norm_stats(DATA_FILE))

# Single-step data — used by NoNoise
train_x_ss, train_y_ss = load_data(DATA_FILE, norm_strategy)
dl_ss = DataLoader((train_x_ss, train_y_ss), batchsize=32, shuffle=false, collate=true)

# Multi-step data — used by PushforwardRollout
train_x_ms = load_data_multistep(DATA_FILE, norm_strategy, NSTEPS_PFR)
dl_ms = DataLoader(train_x_ms, batchsize=32, shuffle=false, collate=collate_multistep_batch, parallel=false)

# Build a shared model from the first single-step graph
g0   = train_x_ss[1]
din  = size(g0.ndata.dynamic, 1) + size(g0.ndata.forcing, 1) + size(g0.ndata.static, 1)
dout = size(g0.ndata.dynamic, 1)
model = GNN(din, 32, dout, 4; skip=false) |> DEVICE

println("  single-step samples : $(length(train_x_ss))")
println("  multi-step windows  : $(length(train_x_ms))  (nsteps=$NSTEPS_PFR)")
println("  din=$din  dout=$dout  params: $(sum(length, Flux.trainables(model)))")
println()

# ─── Benchmark helper ─────────────────────────────────────────────────────────

"""
    benchmark(strategy, dl, model) -> (median_s, std_s, losses)

Runs `compute_loss` over every batch in `dl` for `NWARMUP` throwaway passes
(JIT compilation), then times `NMEASURE` full-epoch passes.
Returns median and std wall-clock time per epoch (seconds), plus the vector
of per-batch noiseless losses collected in the final timed pass.
"""
function benchmark(strategy, dl, model)
    for _ in 1:NWARMUP
        for batch in dl
            batch = batch |> DEVICE
            HydroGNN.compute_loss(strategy, model, batch, DEVICE)
        end
        CUDA.synchronize()
    end

    times  = Vector{Float64}(undef, NMEASURE)
    losses = Float64[]

    for i in 1:NMEASURE
        empty!(losses)
        t = CUDA.@elapsed begin
            for batch in dl
                batch = batch |> DEVICE
                _, _, nl = HydroGNN.compute_loss(strategy, model, batch, DEVICE)
                push!(losses, Float64(nl))
            end
            CUDA.synchronize()
        end
        times[i] = t
    end

    return median(times), std(times), losses
end

# ─── Run ──────────────────────────────────────────────────────────────────────

strategies = [
    ("NoNoise",                           NoNoise(),                            dl_ss),
    ("PushforwardRollout(n=$NSTEPS_PFR)", PushforwardRollout(NSTEPS_PFR, 0.0), dl_ms),
]

results = NamedTuple{(:label,:median_s,:std_s,:mean_loss)}[]

println("Benchmarking  (warmup=$NWARMUP, measure=$NMEASURE passes each)...")
println()

for (label, strategy, dl) in strategies
    print("  $label ... ")
    flush(stdout)
    med, sd, losses = benchmark(strategy, dl, model)
    push!(results, (label=label, median_s=med, std_s=sd, mean_loss=mean(losses)))
    @printf("done   %.3f ± %.3f s   mean noiseless loss = %.6f\n", med, sd, mean(losses))
end

# ─── Summary table ────────────────────────────────────────────────────────────

println()
println("=" ^ 72)
@printf("  %-36s  %9s  %7s  %6s  %12s\n",
        "Strategy", "median(s)", "std(s)", "ratio", "mean loss")
println("=" ^ 72)

baseline = results[1].median_s
for r in results
    @printf("  %-36s  %9.3f  %7.3f  %5.2fx  %12.6f\n",
            r.label, r.median_s, r.std_s, r.median_s / baseline, r.mean_loss)
end
println("=" ^ 72)
println()
println("ratio  = time relative to NoNoise (1.00x)")
println("Median over $NMEASURE full-epoch passes, device=$(CUDA.functional() ? CUDA.name(CUDA.device()) : "CPU").")
println("Increase NSTEPS_PFR at the top of this file to explore deeper rollouts.")
