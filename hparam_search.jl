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
save_dir    = "models/hparam_search_test_run"
log_file    = joinpath(save_dir, "results.toml")

mkpath(save_dir)

norm_strategy  = GlobalNorm(compute_norm_stats(train_path))
train_strategy = SingleStepNoise(3e-2)   # or: NoNoise(), MultiStepRollout(5, 0.0), PushforwardRollout(5, 0.0)

# Pre-load data once — shared across all trials
if train_strategy isa RolloutStrategy
    train_x       = load_data_multistep(train_path, norm_strategy, train_strategy.nsteps)
    make_dl_train = nbatch -> DataLoader(train_x, batchsize=1, shuffle=true, collate=false, parallel=true)
else
    train_x, train_y = load_data(train_path, norm_strategy)
    make_dl_train    = nbatch -> DataLoader((train_x, train_y), batchsize=nbatch, shuffle=true, collate=true, parallel=true)
end

val_x, val_y = load_data(valid_path, norm_strategy)
dl_valid     = DataLoader((val_x, val_y), batchsize=32, shuffle=false, collate=true, parallel=true) |> device

din  = size(val_x[1].ndata.dynamic, 1) + size(val_x[1].ndata.forcing, 1) + size(val_x[1].ndata.static, 1)
dout = size(val_y[1].ndata.x, 1)

# ─── Search space ─────────────────────────────────────────────────────────────
dhidden_candidates = [16]
nhidden_candidates = [2]
skip_candidates    = [false]
noise_candidates   = [1e-2]

# ─── Load existing log to skip completed trials ───────────────────────────────
global completed = isfile(log_file) ? TOML.parsefile(log_file) : Dict{String, Any}()

global best_model          = nothing
global best_model_settings = nothing
global best_val_global     = Inf

# ─── Trial loop ───────────────────────────────────────────────────────────────
for dhidden in dhidden_candidates, nhidden in nhidden_candidates, skip in skip_candidates, noise in noise_candidates
    trial_name = "dh$(dhidden)_nh$(nhidden)_skip$(skip)_ns$(noise)"

    if haskey(completed, trial_name)
        @info "Skipping $trial_name (already logged)"
        continue
    end

    @info "\n=== Trial: $trial_name ==="

    settings = TrainSettings()
    settings.nepochs          = 5
    settings.nbatch           = 32
    settings.train_data_path  = train_path
    settings.valid_data_path  = valid_path
    settings.model_name       = trial_name
    settings.save_dir         = save_dir

    model_settings = ModelSettings()
    model_settings.dhidden          = dhidden
    model_settings.nhidden          = nhidden
    model_settings.skip_connections = skip
    model_settings.din              = din
    model_settings.dout             = dout

    dl_train = make_dl_train(settings.nbatch) |> device
    model    = GNN(model_settings.din, model_settings.dhidden, model_settings.dout, model_settings.nhidden; skip=model_settings.skip_connections) |> device
    trial_strategy = iszero(noise) ? NoNoise() : SingleStepNoise(noise)
    nparams  = sum(length, Flux.trainables(model))
    @info "  parameters: $nparams"

    t_start = time()
    train_loss, loss_noiseless, valid_loss = train_model!(model, dl_train, dl_valid, device, settings,
                                                          norm_strategy, trial_strategy)
    elapsed = time() - t_start

    trial_dir = joinpath(save_dir, trial_name)
    mkpath(trial_dir)
    plot_loss(train_loss, loss_noiseless, valid_loss, joinpath(trial_dir, "loss_plot.png"))
    open(joinpath(trial_dir, "train_settings.toml"), "w") do io
        d = Dict{String,Any}(string(f) => getfield(settings, f) for f in fieldnames(TrainSettings))
        d["norm_strategy"]  = strategy_to_dict(norm_strategy)
        d["train_strategy"] = strategy_to_dict(trial_strategy)
        TOML.print(io, d)
    end
    save_model_settings(model_settings, joinpath(trial_dir, "model_settings.toml"))

    trial_best_val = minimum(valid_loss)
    final_val      = last(valid_loss)
    @info "  best valid loss: $trial_best_val  |  time: $(round(elapsed, digits=1))s"

    # Keep best model in memory
    if trial_best_val < best_val_global
        global best_val_global     = trial_best_val
        global best_model          = model |> cpu
        global best_model_settings = model_settings
    end

    # Incrementally write this trial to the TOML log
    completed[trial_name] = Dict(
        "dhidden"          => dhidden,
        "nhidden"          => nhidden,
        "skip_connections" => skip,
        "noise_scale"      => noise,
        "nparams"          => nparams,
        "best_val_loss"    => trial_best_val,
        "final_val_loss"   => final_val,
        "train_seconds"    => round(elapsed, digits=2),
        "timestamp"        => string(now()),
    )
    open(log_file, "w") do io
        TOML.print(io, completed)
    end
end

# ─── Report ───────────────────────────────────────────────────────────────────
entries = sort(collect(completed), by = kv -> kv[2]["best_val_loss"])

@info "\n=== Results (sorted by best validation loss) ==="
@info rpad("trial", 38) * rpad("dh", 6) * rpad("nh", 6) * rpad("skip", 8) *
      rpad("noise", 8) * rpad("nparams", 10) * rpad("best_val", 14) * "time (s)"
for (name, r) in entries
    @info rpad(name, 38) * rpad(r["dhidden"], 6) * rpad(r["nhidden"], 6) *
          rpad(r["skip_connections"], 8) * rpad(r["noise_scale"], 8) *
          rpad(r["nparams"], 10) * rpad(round(r["best_val_loss"], sigdigits=5), 14) *
          string(r["train_seconds"])
end

best_name, best = first(entries)
@info "Best: $best_name  dh=$(best["dhidden"])  nh=$(best["nhidden"])  " *
      "skip=$(best["skip_connections"])  noise=$(best["noise_scale"])  " *
      "params=$(best["nparams"])  val=$(best["best_val_loss"])  time=$(best["train_seconds"])s"

# ─── Save best model and evaluate all validation trajectories ─────────────────
if best_model !== nothing
    best_dir = joinpath(save_dir, "best_model")
    mkpath(best_dir)

    jldsave(joinpath(best_dir, "model.jld2");
        model          = best_model,
        norm_strategy  = norm_strategy,
        train_strategy = train_strategy)

    winning_dir = joinpath(save_dir, best_name)
    cp(joinpath(winning_dir, "train_settings.toml"), joinpath(best_dir, "train_settings.toml"); force=true)
    cp(joinpath(winning_dir, "model_settings.toml"), joinpath(best_dir, "model_settings.toml"); force=true)

    @info "\nBest model saved to: $best_dir"

    @info "Evaluating all validation trajectories..."
    eval_dir = joinpath(best_dir, "validation")
    evaluate_all_trajectories(valid_path, best_model, eval_dir, norm_strategy; device=device)
else
    @info "\nAll trials were skipped (already completed). Re-run without the log file to retrain."
end
