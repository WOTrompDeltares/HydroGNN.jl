import CUDA
import cuDNN
using Dates
using Flux
using Flux: DataLoader
using GraphNeuralNetworks
using JLD2
using TOML

# ─── Private helpers ──────────────────────────────────────────────────────────

function _strategy_label(d::Dict)
    parts = [d["name"]]
    haskey(d, "nsteps")        && push!(parts, "ns$(d["nsteps"])")
    haskey(d, "step_interval") && push!(parts, "si$(d["step_interval"])")
    get(d, "noise_scale", 0.0) != 0.0 && push!(parts, "noise$(d["noise_scale"])")
    return join(parts, "_")
end

function _split_swept(d::Dict)
    fixed = Dict{String,Any}()
    swept = Dict{String,Vector}()
    for (k, v) in d
        v isa Vector ? (swept[k] = v) : (fixed[k] = v)
    end
    return fixed, swept
end

# ─── Main entry point ─────────────────────────────────────────────────────────

"""
    run_hparam_search(config_path::String)

Run a hyperparameter search experiment defined by a TOML config file.

The config must have top-level sections `[experiment]`, `[model]`, `[train]`,
and `[[strategies]]`. Any field that is a TOML array is swept over; scalar
fields are fixed. Returns a sorted vector of `(trial_name, result_dict)` pairs.
"""
function run_hparam_search(config_path::String)
    cfg = TOML.parsefile(config_path)
    exp_cfg    = cfg["experiment"]
    model_cfg  = cfg["model"]
    train_cfg  = cfg["train"]
    strat_cfgs = cfg["strategies"]

    # ── Device ────────────────────────────────────────────────────────────────
    if CUDA.has_cuda() && CUDA.functional()
        println("CUDA is available. Using GPU.")
        device = gpu
    else
        println("CUDA is not available. Using CPU.")
        device = cpu
    end

    # ── Paths & norm ──────────────────────────────────────────────────────────
    train_path = exp_cfg["train_data"]
    valid_path = exp_cfg["valid_data"]
    save_dir   = exp_cfg["save_dir"]
    log_file   = joinpath(save_dir, "results.toml")

    mkpath(save_dir)

    norm_strategy = get(exp_cfg, "norm", "GlobalNorm") == "PerTrajectoryNorm" ?
                    PerTrajectoryNorm() : GlobalNorm(compute_norm_stats(train_path))

    val_x, val_y = load_data(valid_path, norm_strategy)
    din  = size(val_x[1].ndata.dynamic, 1) + size(val_x[1].ndata.forcing, 1) + size(val_x[1].ndata.static, 1)
    dout = size(val_y[1].ndata.x, 1)

    # ── Swept vs fixed dims ───────────────────────────────────────────────────
    model_fixed, model_swept = _split_swept(model_cfg)
    train_fixed, train_swept = _split_swept(train_cfg)

    all_swept    = merge(model_swept, train_swept)
    swept_keys   = sort(collect(keys(all_swept)))
    swept_values = [all_swept[k] for k in swept_keys]

    combos = if isempty(swept_keys)
        [Dict{String,Any}()]
    else
        vec([Dict{String,Any}(zip(swept_keys, vals))
             for vals in Iterators.product(swept_values...)])
    end

    # ── Data caches (lazy-loaded, keyed by nsteps) ────────────────────────────
    ms_train_cache = Dict{Int,Any}()
    ms_valid_cache = Dict{Int,Any}()
    ss_train_data  = Ref{Any}(nothing)

    function get_ms_train_dl(nsteps, nbatch)
        haskey(ms_train_cache, nsteps) ||
            (ms_train_cache[nsteps] = load_data_multistep(train_path, norm_strategy, nsteps))
        return DataLoader(ms_train_cache[nsteps], batchsize=nbatch, shuffle=true,
                          collate=collate_multistep_batch, parallel=true) |> device
    end

    function get_ms_valid_dl(nsteps, nbatch)
        haskey(ms_valid_cache, nsteps) ||
            (ms_valid_cache[nsteps] = load_data_multistep(valid_path, norm_strategy, nsteps))
        return DataLoader(ms_valid_cache[nsteps], batchsize=nbatch, shuffle=false,
                          collate=collate_multistep_batch, parallel=true) |> device
    end

    function get_ss_train_dl(nbatch)
        if ss_train_data[] === nothing
            ss_train_data[] = load_data(train_path, norm_strategy)
        end
        tx, ty = ss_train_data[]
        return DataLoader((tx, ty), batchsize=nbatch, shuffle=true,
                          collate=true, parallel=true) |> device
    end

    # ── Resume state ──────────────────────────────────────────────────────────
    completed           = isfile(log_file) ? TOML.parsefile(log_file) : Dict{String,Any}()
    best_model          = nothing
    best_model_settings = nothing
    best_train_strategy = nothing
    best_val_global     = Inf

    # ── Trial loop ────────────────────────────────────────────────────────────
    for strat_d in strat_cfgs, combo in combos
        slabel     = _strategy_label(strat_d)
        clabel     = join(["$(k)$(combo[k])" for k in swept_keys], "_")
        trial_name = isempty(clabel) ? slabel : "$(slabel)_$(clabel)"

        if haskey(completed, trial_name)
            @info "Skipping $trial_name (already completed)"
            continue
        end

        @info "\n=== Trial: $trial_name ==="

        train_strategy = build_strategy_from_config(strat_d)
        nsteps = get(strat_d, "nsteps", 1)

        settings                 = train_settings_from_toml(merge(train_fixed, combo))
        settings.train_data_path = train_path
        settings.valid_data_path = valid_path
        settings.model_name      = trial_name
        settings.save_dir        = save_dir

        if train_strategy isa RolloutStrategy
            dl_train          = get_ms_train_dl(nsteps, settings.nbatch)
            dl_valid_strategy = get_ms_valid_dl(nsteps, settings.nbatch)
        else
            dl_train          = get_ss_train_dl(settings.nbatch)
            dl_valid_strategy = DataLoader((val_x, val_y), batchsize=settings.nbatch, shuffle=false,
                                           collate=true, parallel=true) |> device
        end

        model_settings      = model_settings_from_dict(merge(model_fixed, combo))
        model_settings.din  = din
        model_settings.dout = dout

        model   = GNN(din, model_settings.dhidden, dout, model_settings.nhidden;
                      skip=model_settings.skip_connections) |> device
        nparams = sum(length, Flux.trainables(model))
        @info "  dhidden=$(model_settings.dhidden)  nhidden=$(model_settings.nhidden)  skip=$(model_settings.skip_connections)  strategy=$train_strategy  nparams=$nparams"

        t_start = time()
        train_loss, loss_noiseless, valid_loss_strategy, valid_loss_1step = train_model!(
            model, dl_train, dl_valid_strategy, device, settings, norm_strategy, train_strategy)
        elapsed = time() - t_start

        trial_dir = joinpath(save_dir, trial_name)
        mkpath(trial_dir)
        plot_loss(train_loss, loss_noiseless, valid_loss_strategy, joinpath(trial_dir, "loss_plot.png");
                  val_loss_1step=valid_loss_1step)
        open(joinpath(trial_dir, "train_settings.toml"), "w") do io
            d = Dict{String,Any}(string(f) => getfield(settings, f) for f in fieldnames(TrainSettings))
            d["norm_strategy"]  = strategy_to_dict(norm_strategy)
            d["train_strategy"] = strategy_to_dict(train_strategy)
            TOML.print(io, d)
        end
        save_model_settings(model_settings, joinpath(trial_dir, "model_settings.toml"))

        println("  Running validation rollout for $trial_name...")
        model_cpu = model |> cpu
        mean_final_rmse, _ = evaluate_all_trajectories(valid_path, model_cpu,
                                 joinpath(trial_dir, "validation"), norm_strategy; device=cpu)

        trial_best_val = minimum(valid_loss_strategy)
        final_val      = last(valid_loss_strategy)
        @info "  best valid loss: $trial_best_val  |  rollout RMSE: $mean_final_rmse  |  time: $(round(elapsed, digits=1))s"

        if mean_final_rmse < best_val_global
            best_val_global     = mean_final_rmse
            best_model          = model_cpu
            best_model_settings = model_settings
            best_train_strategy = train_strategy
        end

        completed[trial_name] = Dict(
            "strategy"         => slabel,
            "dhidden"          => model_settings.dhidden,
            "nhidden"          => model_settings.nhidden,
            "skip_connections" => model_settings.skip_connections,
            "nepochs"          => settings.nepochs,
            "nbatch"           => settings.nbatch,
            "nparams"          => nparams,
            "mean_final_rmse"  => mean_final_rmse,
            "best_val_loss"    => trial_best_val,
            "final_val_loss"   => final_val,
            "train_seconds"    => round(elapsed, digits=2),
            "timestamp"        => string(now()),
        )
        open(log_file, "w") do io
            TOML.print(io, completed)
        end
    end

    # ── Report ────────────────────────────────────────────────────────────────
    entries = sort(collect(completed), by = kv -> get(kv[2], "mean_final_rmse", Inf))

    @info "\n=== Results (sorted by rollout RMSE) ==="
    @info rpad("trial", 44) * rpad("nparams", 10) * rpad("rollout_rmse", 16) * rpad("best_val", 14) * "time (s)"
    for (name, r) in entries
        rmse_str = isinf(get(r, "mean_final_rmse", Inf)) ? "Inf (diverged)" :
                   string(round(r["mean_final_rmse"], sigdigits=5))
        @info rpad(name, 44) * rpad(r["nparams"], 10) * rpad(rmse_str, 16) *
              rpad(round(r["best_val_loss"], sigdigits=5), 14) * string(r["train_seconds"])
    end

    best_name, best = first(entries)
    @info "Best: $best_name  rollout_rmse=$(get(best, "mean_final_rmse", Inf))  time=$(best["train_seconds"])s"

    # ── Save best model ───────────────────────────────────────────────────────
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
        @info "\nAll trials were skipped (already completed)."
    end

    return entries
end
