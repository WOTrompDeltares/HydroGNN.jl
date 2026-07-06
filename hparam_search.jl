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
save_dir    = "models/scheduled_rollout_search"
log_file    = joinpath(save_dir, "results.toml")

mkpath(save_dir)

norm_strategy = GlobalNorm(compute_norm_stats(train_path))

val_x, val_y = load_data(valid_path, norm_strategy)

din  = size(val_x[1].ndata.dynamic, 1) + size(val_x[1].ndata.forcing, 1) + size(val_x[1].ndata.static, 1)
dout = size(val_y[1].ndata.x, 1)

# ─── Fixed model settings ─────────────────────────────────────────────────────
const DHIDDEN = 128
const NHIDDEN = 3
const SKIP    = false
const NBATCH  = 32
const NEPOCHS       = 250
const NSTEPS        = 5    # schedule ramps from 1 up to this, then plateaus
const STEP_INTERVAL = 50   # epochs per step increase

# epoch 1-50 → 1 step, 51-100 → 2, 101-150 → 3, 151-200 → 4, 201-250 → 5
step_schedule(epoch) = div(epoch - 1, STEP_INTERVAL) + 1

# ─── Search space ─────────────────────────────────────────────────────────────
strategy_candidates = [
    ("ScheduledRollout",     () -> ScheduledRollout(step_schedule,     NSTEPS, 0.0)),
    ("ScheduledPushforward", () -> ScheduledPushforward(step_schedule, NSTEPS, 0.0)),
]

# ─── Data cache (load once per nsteps, reused across strategy types) ──────────
data_cache       = Dict{Int, Any}()
valid_data_cache = Dict{Int, Any}()

function get_train_dl(nsteps)
    if !haskey(data_cache, nsteps)
        data_cache[nsteps] = load_data_multistep(train_path, norm_strategy, nsteps)
    end
    return DataLoader(data_cache[nsteps], batchsize=NBATCH, shuffle=true,
                      collate=collate_multistep_batch, parallel=true) |> device
end

function get_valid_dl(nsteps)
    if !haskey(valid_data_cache, nsteps)
        valid_data_cache[nsteps] = load_data_multistep(valid_path, norm_strategy, nsteps)
    end
    return DataLoader(valid_data_cache[nsteps], batchsize=NBATCH, shuffle=false,
                      collate=collate_multistep_batch, parallel=true) |> device
end

# ─── Load existing log to skip completed trials ───────────────────────────────
global completed = isfile(log_file) ? TOML.parsefile(log_file) : Dict{String, Any}()

global best_model          = nothing
global best_model_settings = nothing
global best_train_strategy = nothing
global best_val_global     = Inf

# ─── Trial loop ───────────────────────────────────────────────────────────────
for (strategy_name, make_strategy) in strategy_candidates
    trial_name = strategy_name

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

    train_strategy = make_strategy()
    dl_train       = get_train_dl(NSTEPS)
    model          = GNN(din, DHIDDEN, dout, NHIDDEN; skip=SKIP) |> device
    nparams        = sum(length, Flux.trainables(model))
    @info "  strategy: $strategy_name  nsteps_max: $NSTEPS  parameters: $nparams"

    t_start = time()
    train_loss, loss_noiseless, valid_loss_strategy, valid_loss_1step, val_bc_adj_loss, val_bc_grad_loss = train_model!(
        model, dl_train, get_valid_dl(NSTEPS), device, settings, norm_strategy, train_strategy)
    elapsed = time() - t_start

    trial_dir = joinpath(save_dir, trial_name)
    mkpath(trial_dir)
    plot_loss(train_loss, loss_noiseless, valid_loss_strategy, joinpath(trial_dir, "loss_plot.png");
              val_loss_1step=valid_loss_1step,
              val_bc_adj_loss=val_bc_adj_loss,
              val_bc_grad_loss=val_bc_grad_loss)
    open(joinpath(trial_dir, "train_settings.toml"), "w") do io
        d = Dict{String,Any}(string(f) => getfield(settings, f) for f in fieldnames(TrainSettings))
        d["norm_strategy"]  = strategy_to_dict(norm_strategy)
        d["train_strategy"] = strategy_to_dict(train_strategy)
        TOML.print(io, d)
    end
    save_model_settings(model_settings, joinpath(trial_dir, "model_settings.toml"))

    # ── Per-trial validation rollout ──────────────────────────────────────────
    println("  Running validation rollout for $trial_name...")
    model_cpu = model |> cpu
    mean_final_rmse, _ = evaluate_all_trajectories(valid_path, model_cpu,
                             joinpath(trial_dir, "validation"), norm_strategy; device=cpu)

    trial_best_val = minimum(valid_loss_strategy)
    final_val      = last(valid_loss_strategy)
    @info "  best valid loss: $trial_best_val  |  rollout RMSE: $mean_final_rmse  |  time: $(round(elapsed, digits=1))s"

    if mean_final_rmse < best_val_global
        global best_val_global     = mean_final_rmse
        global best_model          = model_cpu
        global best_model_settings = model_settings
        global best_train_strategy = train_strategy
    end

    completed[trial_name] = Dict(
        "strategy_name"   => strategy_name,
        "nsteps_max"      => NSTEPS,
        "step_interval"   => STEP_INTERVAL,
        "nparams"         => nparams,
        "mean_final_rmse" => mean_final_rmse,
        "best_val_loss"   => trial_best_val,
        "final_val_loss"  => final_val,
        "train_seconds"   => round(elapsed, digits=2),
        "timestamp"       => string(now()),
    )
    open(log_file, "w") do io
        TOML.print(io, completed)
    end
end

# ─── Report ───────────────────────────────────────────────────────────────────
entries = sort(collect(completed), by = kv -> get(kv[2], "mean_final_rmse", Inf))

@info "\n=== Results (sorted by rollout RMSE) ==="
@info rpad("trial", 26) * rpad("strategy", 22) * rpad("nsteps_max", 12) * rpad("step_iv", 10) *
      rpad("nparams", 10) * rpad("rollout_rmse", 16) * rpad("best_val", 14) * "time (s)"
for (name, r) in entries
    rmse_str = isinf(get(r, "mean_final_rmse", Inf)) ? "Inf (diverged)" :
               string(round(r["mean_final_rmse"], sigdigits=5))
    @info rpad(name, 26) * rpad(r["strategy_name"], 22) * rpad(r["nsteps_max"], 12) *
          rpad(r["step_interval"], 10) * rpad(r["nparams"], 10) * rpad(rmse_str, 16) *
          rpad(round(r["best_val_loss"], sigdigits=5), 14) * string(r["train_seconds"])
end

best_name, best = first(entries)
@info "Best: $best_name  strategy=$(best["strategy_name"])  nsteps_max=$(best["nsteps_max"])  " *
      "params=$(best["nparams"])  rollout_rmse=$(get(best, "mean_final_rmse", Inf))  time=$(best["train_seconds"])s"

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
