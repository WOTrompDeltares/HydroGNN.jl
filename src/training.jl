using Flux
using Zygote
using JLD2
using ProgressMeter
using ParameterSchedulers
using TOML
using Random

abstract type TrainStrategy end
abstract type RolloutStrategy <: TrainStrategy end

struct SingleStepNoise <: TrainStrategy
    scale::Float64
end

struct NoNoise <: TrainStrategy end

struct MultiStepRollout <: RolloutStrategy
    nsteps::Int
    noise_scale::Float64
end

struct PushforwardRollout <: RolloutStrategy
    nsteps::Int
    noise_scale::Float64
end

mutable struct ScheduledRollout <: RolloutStrategy
    steps::Vector{Int}      # nsteps value at each stage
    durations::Vector{Int}  # number of epochs to spend at each stage
    noise_scale::Float64
    current_nsteps::Int
end

"""Fully explicit constructor."""
function ScheduledRollout(steps::Vector{Int}, durations::Vector{Int}, noise_scale::Float64)
    length(steps) == length(durations) || error("steps and durations must have the same length")
    ScheduledRollout(steps, durations, noise_scale, steps[1])
end

"""Uniform schedule: ramps 1:nsteps, each stage lasting step_interval epochs."""
ScheduledRollout(nsteps::Int, step_interval::Int, noise_scale::Float64) =
    ScheduledRollout(collect(1:nsteps), fill(step_interval, nsteps), noise_scale)

"""Threshold schedule: step increases by 1 at each epoch in epoch_thresholds (length = nsteps-1)."""
function ScheduledRollout(epoch_thresholds::Vector{Int}, noise_scale::Float64)
    n = length(epoch_thresholds) + 1
    ScheduledRollout(collect(1:n), [epoch_thresholds[1] - 1; diff(epoch_thresholds); 1], noise_scale)
end

step_schedule!(::TrainStrategy, epoch) = nothing
function step_schedule!(s::ScheduledRollout, epoch)
    stage = clamp(searchsortedfirst(cumsum(s.durations), epoch), 1, length(s.steps))
    s.current_nsteps = s.steps[stage]
end

mutable struct ScheduledPushforward <: RolloutStrategy
    steps::Vector{Int}
    durations::Vector{Int}
    noise_scale::Float64
    current_nsteps::Int
end

"""Fully explicit constructor."""
function ScheduledPushforward(steps::Vector{Int}, durations::Vector{Int}, noise_scale::Float64)
    length(steps) == length(durations) || error("steps and durations must have the same length")
    ScheduledPushforward(steps, durations, noise_scale, steps[1])
end

"""Uniform schedule: ramps 1:nsteps, each stage lasting step_interval epochs."""
ScheduledPushforward(nsteps::Int, step_interval::Int, noise_scale::Float64) =
    ScheduledPushforward(collect(1:nsteps), fill(step_interval, nsteps), noise_scale)

"""Threshold schedule: step increases by 1 at each epoch in epoch_thresholds (length = nsteps-1)."""
function ScheduledPushforward(epoch_thresholds::Vector{Int}, noise_scale::Float64)
    n = length(epoch_thresholds) + 1
    ScheduledPushforward(collect(1:n), [epoch_thresholds[1] - 1; diff(epoch_thresholds); 1], noise_scale)
end

function step_schedule!(s::ScheduledPushforward, epoch)
    stage = clamp(searchsortedfirst(cumsum(s.durations), epoch), 1, length(s.steps))
    s.current_nsteps = s.steps[stage]
end

strategy_to_dict(s::SingleStepNoise)      = Dict{String,Any}("name" => "SingleStepNoise",      "scale"       => s.scale)
strategy_to_dict(::NoNoise)               = Dict{String,Any}("name" => "NoNoise")
strategy_to_dict(s::MultiStepRollout)     = Dict{String,Any}("name" => "MultiStepRollout",     "nsteps"      => s.nsteps,      "noise_scale" => s.noise_scale)
strategy_to_dict(s::PushforwardRollout)   = Dict{String,Any}("name" => "PushforwardRollout",   "nsteps"      => s.nsteps,      "noise_scale" => s.noise_scale)
strategy_to_dict(s::ScheduledRollout)     = Dict{String,Any}("name" => "ScheduledRollout",     "steps" => s.steps, "durations" => s.durations, "noise_scale" => s.noise_scale)
strategy_to_dict(s::ScheduledPushforward) = Dict{String,Any}("name" => "ScheduledPushforward", "steps" => s.steps, "durations" => s.durations, "noise_scale" => s.noise_scale)

from_dict(::Type{NoNoise},              d) = NoNoise()
from_dict(::Type{SingleStepNoise},      d) = SingleStepNoise(d["scale"])
from_dict(::Type{MultiStepRollout},     d) = MultiStepRollout(d["nsteps"], d["noise_scale"])
from_dict(::Type{PushforwardRollout},   d) = PushforwardRollout(d["nsteps"], d["noise_scale"])
from_dict(::Type{ScheduledRollout},     d) = ScheduledRollout(Int.(d["steps"]), Int.(d["durations"]), d["noise_scale"])
from_dict(::Type{ScheduledPushforward}, d) = ScheduledPushforward(Int.(d["steps"]), Int.(d["durations"]), d["noise_scale"])

const _STRATEGY_TYPES = Dict{String,Type{<:TrainStrategy}}(
    "NoNoise"              => NoNoise,
    "SingleStepNoise"      => SingleStepNoise,
    "MultiStepRollout"     => MultiStepRollout,
    "PushforwardRollout"   => PushforwardRollout,
    "ScheduledRollout"     => ScheduledRollout,
    "ScheduledPushforward" => ScheduledPushforward,
)

function save_train_strategy(s::TrainStrategy, path::String)
    open(path, "w") do io
        TOML.print(io, strategy_to_dict(s))
    end
end

"""
    load_train_strategy(path) -> TrainStrategy

Read a TOML file written by `save_train_strategy` and reconstruct the
corresponding `TrainStrategy`.  For `ScheduledRollout` / `ScheduledPushforward`
the schedule callable is not serialised; it is restored as `identity` and
should be reassigned after loading if scheduling is required.
"""
function load_train_strategy(path::String)
    d    = TOML.parsefile(path)
    name = d["name"]
    T    = get(_STRATEGY_TYPES, name, nothing)
    T === nothing && error("Unknown TrainStrategy name: \"$name\"")
    return from_dict(T, d)
end

# Compact single-line display:  MultiStepRollout(nsteps=3, noise_scale=0.02)
function Base.show(io::IO, s::TrainStrategy)
    d    = strategy_to_dict(s)
    name = pop!(d, "name")
    if isempty(d)
        print(io, name, "()")
    else
        print(io, name, "(")
        pairs = collect(d)
        for (i, (k, v)) in enumerate(pairs)
            print(io, k, "=", v)
            i < length(pairs) && print(io, ", ")
        end
        print(io, ")")
    end
end

# Multiline REPL display
function Base.show(io::IO, ::MIME"text/plain", s::TrainStrategy)
    d    = strategy_to_dict(s)
    name = pop!(d, "name")
    if isempty(d)
        print(io, name, "()")
    else
        println(io, name, ":")
        for (k, v) in d
            println(io, "  ", rpad(k, 14), " = ", v)
        end
        if hasproperty(s, :current_nsteps)
            print(io, "  ", rpad("current_nsteps", 14), " = ", s.current_nsteps)
        end
    end
end

# Returns (strategy_loss, loss_1step, bc_adj_loss, bc_grad_loss).
# Gradients flow only through strategy_loss; all diagnostics are wrapped in ignore_derivatives.
function compute_loss(strategy::SingleStepNoise, model, batch, device)
    x, y    = batch
    bc_mask = x.ndata.bc_mask
    loss_1step, bc_adj, bc_grad = Flux.ignore_derivatives() do
        pred0 = model(x)
        Flux.mse(pred0, y.x), _bc_adjacent_mse(pred0, y.x, bc_mask), _bc_gradient_mse(pred0, y.x, bc_mask)
    end
    if strategy.scale > 0
        x = Flux.ignore_derivatives() do
            GNNGraph(x, ndata=(; x.ndata...,
                                dynamic=x.ndata.dynamic .+ Float32(strategy.scale) .* randn!(similar(x.ndata.dynamic))))
        end
    end
    return Flux.mse(model(x), y.x), loss_1step, bc_adj, bc_grad
end

function compute_loss(::NoNoise, model, batch, device)
    x, y    = batch
    bc_mask = x.ndata.bc_mask
    loss    = Flux.mse(model(x), y.x)
    bc_adj, bc_grad = Flux.ignore_derivatives() do
        pred = model(x)
        _bc_adjacent_mse(pred, y.x, bc_mask), _bc_gradient_mse(pred, y.x, bc_mask)
    end
    return loss, Flux.ignore_derivatives(() -> loss), bc_adj, bc_grad
end

# MSE at cells immediately adjacent to each BC node (first interior neighbours).
# Returns 0 when bc_mask is all-false (no-op for non-BC datasets).
function _bc_adjacent_mse(yhat, target, bc_mask)
    any(bc_mask) || return 0.0f0
    nnodes   = size(bc_mask, 2)
    bc_cols  = unique([ci[2] for ci in findall(bc_mask)])
    adj_cols = Int[]
    for c in bc_cols
        c > 1      && push!(adj_cols, c - 1)
        c < nnodes && push!(adj_cols, c + 1)
    end
    adj_cols = setdiff(unique(adj_cols), bc_cols)
    isempty(adj_cols) && return 0.0f0
    diff = yhat[:, adj_cols] .- target[:, adj_cols]
    return mean(diff .^ 2)
end

# MSE of the spatial gradient (one-sided finite difference) at each BC node vs ground truth.
# Returns 0 when bc_mask is all-false.
function _bc_gradient_mse(yhat, target, bc_mask)
    any(bc_mask) || return 0.0f0
    nnodes  = size(bc_mask, 2)
    bc_cols = unique([ci[2] for ci in findall(bc_mask)])
    total   = 0.0f0
    n       = 0
    for c in bc_cols
        if c < nnodes
            grad_pred = yhat[:, c+1]   .- yhat[:, c]
            grad_tgt  = target[:, c+1] .- target[:, c]
        else
            grad_pred = yhat[:, c]   .- yhat[:, c-1]
            grad_tgt  = target[:, c] .- target[:, c-1]
        end
        total += mean((grad_pred .- grad_tgt) .^ 2)
        n     += 1
    end
    return total / n
end

# Override predicted dynamic state at BC-prescribed positions with ground-truth values.
# bc_mask is (ndyn × nnodes) Bool — all-false for datasets without BCs (no-op).
function _bc_override(dyn_pred, dyn_target, bc_mask)
    any(bc_mask) || return dyn_pred
    out = copy(dyn_pred)
    out[bc_mask] .= dyn_target[bc_mask]
    return out
end

# MSE excluding BC-prescribed positions so the model is not penalized for them.
function _masked_mse(yhat, target, bc_mask)
    any(bc_mask) || return Flux.mse(yhat, target)
    interior = .!bc_mask
    diff = yhat[interior] .- target[interior]
    return mean(diff .^ 2)
end

function compute_loss(strategy::MultiStepRollout, model, batch, device)
    x_seq    = batch
    bc_mask1 = x_seq[2].ndata.bc_mask
    loss_1step, bc_adj, bc_grad = Flux.ignore_derivatives() do
        pred0 = model(x_seq[1])
        tgt0  = x_seq[2].ndata.dynamic
        Flux.mse(pred0, tgt0), _bc_adjacent_mse(pred0, tgt0, bc_mask1), _bc_gradient_mse(pred0, tgt0, bc_mask1)
    end
    dyn_cur    = x_seq[1].ndata.dynamic
    total_loss = 0.0f0
    for k in 1:strategy.nsteps
        if strategy.noise_scale > 0
            dyn_cur = dyn_cur .+ (Float32(strategy.noise_scale) .* randn!(similar(dyn_cur)))
        end
        g_k = Flux.ignore_derivatives() do
            GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
        end
        yhat_k  = model(g_k)
        bc_mask = x_seq[k+1].ndata.bc_mask
        total_loss += _masked_mse(yhat_k, x_seq[k+1].ndata.dynamic, bc_mask)
        dyn_cur = Flux.ignore_derivatives(() -> _bc_override(yhat_k, x_seq[k+1].ndata.dynamic, bc_mask))
    end
    return total_loss / strategy.nsteps, loss_1step, bc_adj, bc_grad
end

function compute_loss(strategy::PushforwardRollout, model, batch, device)
    x_seq    = batch
    bc_mask1 = x_seq[2].ndata.bc_mask
    loss_1step, bc_adj, bc_grad = Flux.ignore_derivatives() do
        pred0 = model(x_seq[1])
        tgt0  = x_seq[2].ndata.dynamic
        Flux.mse(pred0, tgt0), _bc_adjacent_mse(pred0, tgt0, bc_mask1), _bc_gradient_mse(pred0, tgt0, bc_mask1)
    end
    dyn_cur    = x_seq[1].ndata.dynamic
    for k in 1:(strategy.nsteps - 1)
        if strategy.noise_scale > 0
            dyn_cur = dyn_cur .+ (Float32(strategy.noise_scale) .* randn!(similar(dyn_cur)))
        end
        g_k     = Flux.ignore_derivatives() do
            GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
        end
        yhat_k  = Flux.ignore_derivatives(() -> model(g_k))
        bc_mask = x_seq[k+1].ndata.bc_mask
        dyn_cur = Flux.ignore_derivatives(() -> _bc_override(yhat_k, x_seq[k+1].ndata.dynamic, bc_mask))
    end
    g_last  = Flux.ignore_derivatives() do
        GNNGraph(x_seq[end-1], ndata=(; x_seq[end-1].ndata..., dynamic=dyn_cur))
    end
    bc_mask = x_seq[end].ndata.bc_mask
    return _masked_mse(model(g_last), x_seq[end].ndata.dynamic, bc_mask), loss_1step, bc_adj, bc_grad
end

function compute_loss(strategy::ScheduledPushforward, model, batch, device)
    x_seq    = batch
    bc_mask1 = x_seq[2].ndata.bc_mask
    loss_1step, bc_adj, bc_grad = Flux.ignore_derivatives() do
        pred0 = model(x_seq[1])
        tgt0  = x_seq[2].ndata.dynamic
        Flux.mse(pred0, tgt0), _bc_adjacent_mse(pred0, tgt0, bc_mask1), _bc_gradient_mse(pred0, tgt0, bc_mask1)
    end
    nsteps     = strategy.current_nsteps
    dyn_cur    = x_seq[1].ndata.dynamic
    for k in 1:(nsteps - 1)
        if strategy.noise_scale > 0
            dyn_cur = dyn_cur .+ (Float32(strategy.noise_scale) .* randn!(similar(dyn_cur)))
        end
        g_k     = Flux.ignore_derivatives() do
            GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
        end
        yhat_k  = Flux.ignore_derivatives(() -> model(g_k))
        bc_mask = x_seq[k+1].ndata.bc_mask
        dyn_cur = Flux.ignore_derivatives(() -> _bc_override(yhat_k, x_seq[k+1].ndata.dynamic, bc_mask))
    end
    g_last  = Flux.ignore_derivatives() do
        GNNGraph(x_seq[nsteps], ndata=(; x_seq[nsteps].ndata..., dynamic=dyn_cur))
    end
    bc_mask = x_seq[nsteps+1].ndata.bc_mask
    return _masked_mse(model(g_last), x_seq[nsteps+1].ndata.dynamic, bc_mask), loss_1step, bc_adj, bc_grad
end

function compute_loss(strategy::ScheduledRollout, model, batch, device)
    x_seq    = batch
    bc_mask1 = x_seq[2].ndata.bc_mask
    loss_1step, bc_adj, bc_grad = Flux.ignore_derivatives() do
        pred0 = model(x_seq[1])
        tgt0  = x_seq[2].ndata.dynamic
        Flux.mse(pred0, tgt0), _bc_adjacent_mse(pred0, tgt0, bc_mask1), _bc_gradient_mse(pred0, tgt0, bc_mask1)
    end
    nsteps     = strategy.current_nsteps
    dyn_cur    = x_seq[1].ndata.dynamic
    total_loss = 0.0f0
    for k in 1:nsteps
        if strategy.noise_scale > 0
            dyn_cur = dyn_cur .+ (Float32(strategy.noise_scale) .* randn!(similar(dyn_cur)))
        end
        g_k = Flux.ignore_derivatives() do
            GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
        end
        yhat_k  = model(g_k)
        bc_mask = x_seq[k+1].ndata.bc_mask
        total_loss += _masked_mse(yhat_k, x_seq[k+1].ndata.dynamic, bc_mask)
        dyn_cur = Flux.ignore_derivatives(() -> _bc_override(yhat_k, x_seq[k+1].ndata.dynamic, bc_mask))
    end
    return total_loss / nsteps, loss_1step, bc_adj, bc_grad
end

Base.@kwdef mutable struct TrainSettings
    nepochs::Int = 250
    lr::Float64 = 3e-3
    lr_final::Float64 = 1e-5
    lr_step::Int = 10
    nbatch::Int = 32
    train_data_path::String = "data/wave1d/train.jld2"
    valid_data_path::String = "data/wave1d/valid.jld2"
    model_name::String = "test_run3d"
    save_dir::String = "models"
end

function train_model!(model, dl_train, dl_valid_strategy, device, settings::TrainSettings, norm_strategy::NormStrategy, train_strategy::TrainStrategy)
    nepochs  = settings.nepochs
    lr       = settings.lr
    lr_final = settings.lr_final
    lr_step  = settings.lr_step

    lr_decay = (lr_final / lr)^(lr_step / nepochs)
    opt      = Flux.setup(Adam(lr), model)
    schedule = Step(start=lr, decay=lr_decay, step_sizes=lr_step)
    pr       = Progress(nepochs, desc="Training Progress", showspeed=true)

    train_loss          = zeros(nepochs)
    valid_loss_strategy = zeros(nepochs)
    valid_loss_1step    = zeros(nepochs)
    loss_noiseless      = zeros(nepochs)
    val_bc_adj_loss     = zeros(nepochs)
    val_bc_grad_loss    = zeros(nepochs)

    for epoch in 1:nepochs
        Flux.adjust!(opt, eta=schedule(epoch))
        step_schedule!(train_strategy, epoch)
        loss           = 0.0f0
        noiseless_loss = 0.0f0
        n_valid        = 0

        for batch in dl_train
            (batch_loss, nl_loss, _, _), back = Zygote.pullback(m -> compute_loss(train_strategy, m, batch, device), model)
            if !isnan(batch_loss)
                grads = back((one(batch_loss), nothing, nothing, nothing))[1]
                Flux.Optimise.update!(opt, model, grads)
                loss           += batch_loss
                noiseless_loss += nl_loss
                n_valid        += 1
            end
        end

        val_strat   = 0.0f0
        val_1step   = 0.0f0
        bc_adj      = 0.0f0
        bc_grad     = 0.0f0
        n_valid_val = 0
        for batch in dl_valid_strategy
            s_l, o_l, a_l, g_l = compute_loss(train_strategy, model, batch, device)
            if !isnan(s_l)
                val_strat   += s_l
                val_1step   += o_l
                bc_adj      += a_l
                bc_grad     += g_l
                n_valid_val += 1
            end
        end
        val_strat /= max(1, n_valid_val)
        val_1step /= max(1, n_valid_val)
        bc_adj    /= max(1, n_valid_val)
        bc_grad   /= max(1, n_valid_val)

        loss           /= max(1, n_valid)
        noiseless_loss /= max(1, n_valid)
        next!(pr;
            showvalues=[(:epoch, epoch), (:train_loss, loss), (:val_strategy, val_strat),
                        (:val_1step, val_1step), (:noiseless, noiseless_loss),
                        (:val_bc_adj, bc_adj), (:val_bc_grad, bc_grad)]
        )
        train_loss[epoch]          = loss
        valid_loss_strategy[epoch] = val_strat
        valid_loss_1step[epoch]    = val_1step
        loss_noiseless[epoch]      = noiseless_loss
        val_bc_adj_loss[epoch]     = bc_adj
        val_bc_grad_loss[epoch]    = bc_grad
    end

    return train_loss, loss_noiseless, valid_loss_strategy, valid_loss_1step, val_bc_adj_loss, val_bc_grad_loss
end

function train_settings_from_toml(d::Dict)
    s = TrainSettings()
    for f in fieldnames(TrainSettings)
        key = string(f)
        haskey(d, key) && setfield!(s, f, convert(fieldtype(TrainSettings, f), d[key]))
    end
    return s
end

"""
    build_strategy_from_config(d) -> TrainStrategy

Construct a `TrainStrategy` from a config dict (e.g. from a `[[strategies]]` TOML
section).
"""
function build_strategy_from_config(d::Dict)
    name = d["name"]
    T    = get(_STRATEGY_TYPES, name, nothing)
    T === nothing && error("Unknown strategy name: \"$name\"")
    return from_dict(T, d)
end