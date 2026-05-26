cd(@__DIR__)
using Pkg
Pkg.activate(".")

using GraphNeuralNetworks, JLD2, Statistics, Flux, Dates, TOML
using CUDA, cuDNN
using Flux: DataLoader
using HydroGNN

if CUDA.has_cuda() && CUDA.functional()
    println("CUDA is available. Using GPU.")
    device = gpu
else
    println("CUDA is not available. Using CPU.")
    device = cpu
end

# ─── Fixed configuration ──────────────────────────────────────────────────────
train_path = "data/lake1d_surge/train.jld2"
valid_path  = "data/lake1d_surge/valid.jld2"
save_dir    = "models/rollout_strategy_search"
log_file    = joinpath(save_dir, "results.toml")

mkpath(save_dir)

norm_strategy = GlobalNorm(compute_norm_stats(train_path))

val_x, val_y = load_data(valid_path, norm_strategy)
dl_valid     = DataLoader((val_x, val_y), batchsize=32, shuffle=false, collate=true, parallel=true) |> device

din  = size(val_x[1].ndata.dynamic, 1) + size(val_x[1].ndata.forcing, 1) + size(val_x[1].ndata.static, 1)
dout = size(val_y[1].ndata.x, 1)

# ─── Fixed model settings ─────────────────────────────────────────────────────
const DHIDDEN = 128
const NHIDDEN = 3
const SKIP    = false
const NBATCH  = 32
const NEPOCHS = 50

# ─── Search space ─────────────────────────────────────────────────────────────
strategy_candidates = [
    ("MultiStepRollout",   nsteps -> MultiStepRollout(nsteps, 0.0)),
    ("PushforwardRollout", nsteps -> PushforwardRollout(nsteps, 0.0)),
]
nsteps_candidates = [2, 3, 5, 8]

# ─── Data cache (load once per nsteps, reused across strategy types) ──────────
data_cache = Dict{Int, Any}()

function get_train_dl(nsteps)
    if !haskey(data_cache, nsteps)
        data_cache[nsteps] = load_data_multistep(train_path, norm_strategy, nsteps)
    end
    return DataLoader(data_cache[nsteps], batchsize=NBATCH, shuffle=true,
                      collate=collate_multistep_batch, parallel=true) |> device
end

# ─── Load existing log to skip completed trials ───────────────────────────────
global completed = isfile(log_file) ? TOML.parsefile(log_file) : Dict{String, Any}()

global best_model          = nothing
global best_model_settings = nothing
global best_train_strategy = nothing
global best_val_global     = Inf

# ─── Trial loop ───────────────────────────────────────────────────────────────
for (strategy_name, make_strategy) in strategy_candidates, nsteps in nsteps_candidates
    trial_name = "$(strategy_name)_ns$(nsteps)"

    if haskey(completed, trial_name)
        @info "Skipping $trial_name (already logged)"
        continue
    end

    @info "\n=== Trial: $trial_name ==="

    settings = TrainSettings()
    settings.nepochs         = NEPOCHS
    settings.nbatch          = NBATCH
    settings.train_data_path = train_path
    settings.valid_data_path = valid_path
    settings.model_name      = trial_name
    settings.save_dir        = save_dir

    model_settings = ModelSettings()
    model_settings.dhidden          = DHIDDEN
    model_settings.nhidden          = NHIDDEN
    model_settings.skip_connections = SKIP
    model_settings.din              = din
    model_settings.dout             = dout

    train_strategy = make_strategy(nsteps)
    dl_train       = get_train_dl(nsteps)
    model          = GNN(din, DHIDDEN, dout, NHIDDEN; skip=SKIP) |> device
    nparams        = sum(length, Flux.trainables(model))
    @info "  strategy: $strategy_name  nsteps: $nsteps  parameters: $nparams"

    t_start = time()
    train_loss, loss_noiseless, valid_loss = train_model!(model, dl_train, dl_valid, device, settings,
                                                          norm_strategy, train_strategy)
    elapsed = time() - t_start

    trial_dir = joinpath(save_dir, trial_name)
    mkpath(trial_dir)
    plot_loss(train_loss, loss_noiseless, valid_loss, joinpath(trial_dir, "loss_plot.png"))
    open(joinpath(trial_dir, "train_settings.toml"), "w") do io
        d = Dict{String,Any}(string(f) => getfield(settings, f) for f in fieldnames(TrainSettings))
        d["norm_strategy"]  = strategy_to_dict(norm_strategy)
        d["train_strategy"] = strategy_to_dict(train_strategy)
        TOML.print(io, d)
    end
    save_model_settings(model_settings, joinpath(trial_dir, "model_settings.toml"))

    # ── Per-trial validation rollout ──────────────────────────────────────────
    println("  Running validation rollout for $trial_name...")
    evaluate_all_trajectories(valid_path, model |> cpu, joinpath(trial_dir, "validation"),
                               norm_strategy; device=cpu)

    trial_best_val = minimum(valid_loss)
    final_val      = last(valid_loss)
    @info "  best valid loss: $trial_best_val  |  time: $(round(elapsed, digits=1))s"

    if trial_best_val < best_val_global
        global best_val_global     = trial_best_val
        global best_model          = model |> cpu
        global best_model_settings = model_settings
        global best_train_strategy = train_strategy
    end

    completed[trial_name] = Dict(
        "strategy_name"  => strategy_name,
        "nsteps"         => nsteps,
        "nparams"        => nparams,
        "best_val_loss"  => trial_best_val,
        "final_val_loss" => final_val,
        "train_seconds"  => round(elapsed, digits=2),
        "timestamp"      => string(now()),
    )
    open(log_file, "w") do io
        TOML.print(io, completed)
    end
end

# ─── Report ───────────────────────────────────────────────────────────────────
entries = sort(collect(completed), by = kv -> kv[2]["best_val_loss"])

@info "\n=== Results (sorted by best validation loss) ==="
@info rpad("trial", 36) * rpad("strategy", 22) * rpad("nsteps", 8) *
      rpad("nparams", 10) * rpad("best_val", 14) * "time (s)"
for (name, r) in entries
    @info rpad(name, 36) * rpad(r["strategy_name"], 22) * rpad(r["nsteps"], 8) *
          rpad(r["nparams"], 10) * rpad(round(r["best_val_loss"], sigdigits=5), 14) *
          string(r["train_seconds"])
end

best_name, best = first(entries)
@info "Best: $best_name  strategy=$(best["strategy_name"])  nsteps=$(best["nsteps"])  " *
      "params=$(best["nparams"])  val=$(best["best_val_loss"])  time=$(best["train_seconds"])s"

# ─── Save best model and evaluate all validation trajectories ─────────────────
if best_model !== nothing
    best_dir = joinpath(save_dir, "best_model")
    mkpath(best_dir)

    jldsave(joinpath(best_dir, "model.jld2");
            model=best_model, norm_strategy=norm_strategy,
            train_strategy=best_train_strategy)
    save_model_settings(best_model_settings, joinpath(best_dir, "model_settings.toml"))

    winning_dir = joinpath(save_dir, best_name)
    cp(joinpath(winning_dir, "train_settings.toml"), joinpath(best_dir, "train_settings.toml"); force=true)

    @info "\nBest model saved to: $best_dir"

    println("\nRunning full validation rollout for best model ($best_name)...")
    evaluate_all_trajectories(valid_path, best_model, joinpath(best_dir, "validation"),
                               norm_strategy; device=cpu)
    println("Validation rollout complete.")
else
    @info "\nAll trials were skipped (already completed). Re-run without the log file to retrain."
end
