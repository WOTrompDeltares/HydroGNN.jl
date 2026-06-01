using Flux
using JLD2
using ProgressMeter
using ParameterSchedulers
using TOML

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
    schedule          # callable: epoch -> nsteps (e.g. a ParameterSchedulers schedule)
    nsteps::Int       # maximum nsteps — used by load_data_multistep at data-load time
    noise_scale::Float64
    current_nsteps::Int  # updated each epoch by step_schedule!
end

function ScheduledRollout(schedule, nsteps::Int, noise_scale::Float64)
    ScheduledRollout(schedule, nsteps, noise_scale, 1)
end

step_schedule!(::TrainStrategy, epoch) = nothing
step_schedule!(s::ScheduledRollout, epoch) = (s.current_nsteps = min(round(Int, s.schedule(epoch)), s.nsteps))

mutable struct ScheduledPushforward <: RolloutStrategy
    schedule          # callable: epoch -> nsteps
    nsteps::Int       # maximum nsteps — used by load_data_multistep at data-load time
    noise_scale::Float64
    current_nsteps::Int
end

function ScheduledPushforward(schedule, nsteps::Int, noise_scale::Float64)
    ScheduledPushforward(schedule, nsteps, noise_scale, 1)
end

step_schedule!(s::ScheduledPushforward, epoch) = (s.current_nsteps = min(round(Int, s.schedule(epoch)), s.nsteps))

strategy_to_dict(s::SingleStepNoise)      = Dict{String,Any}("name" => "SingleStepNoise",      "scale"       => s.scale)
strategy_to_dict(::NoNoise)               = Dict{String,Any}("name" => "NoNoise")
strategy_to_dict(s::MultiStepRollout)     = Dict{String,Any}("name" => "MultiStepRollout",     "nsteps"      => s.nsteps,      "noise_scale" => s.noise_scale)
strategy_to_dict(s::PushforwardRollout)   = Dict{String,Any}("name" => "PushforwardRollout",   "nsteps"      => s.nsteps,      "noise_scale" => s.noise_scale)
strategy_to_dict(s::ScheduledRollout)     = Dict{String,Any}("name" => "ScheduledRollout",     "nsteps"      => s.nsteps,      "noise_scale" => s.noise_scale)
strategy_to_dict(s::ScheduledPushforward) = Dict{String,Any}("name" => "ScheduledPushforward", "nsteps"      => s.nsteps,      "noise_scale" => s.noise_scale)

from_dict(::Type{NoNoise},              d) = NoNoise()
from_dict(::Type{SingleStepNoise},      d) = SingleStepNoise(d["scale"])
from_dict(::Type{MultiStepRollout},     d) = MultiStepRollout(d["nsteps"], d["noise_scale"])
from_dict(::Type{PushforwardRollout},   d) = PushforwardRollout(d["nsteps"], d["noise_scale"])
from_dict(::Type{ScheduledRollout},     d) = ScheduledRollout(identity, d["nsteps"], d["noise_scale"])
from_dict(::Type{ScheduledPushforward}, d) = ScheduledPushforward(identity, d["nsteps"], d["noise_scale"])

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

function compute_loss(strategy::SingleStepNoise, model, batch, device)
    x, y = batch
    noiseless_loss = Flux.mse(model(x), y.x)
    x.ndata.dynamic .+= (strategy.scale * randn(size(x.ndata.dynamic))) |> device
    batch_loss, grad = Flux.withgradient(model) do m
        Flux.mse(m(x), y.x)
    end
    return batch_loss, grad[1], noiseless_loss
end

function compute_loss(::NoNoise, model, batch, device)
    x, y = batch
    batch_loss, grad = Flux.withgradient(model) do m
        Flux.mse(m(x), y.x)
    end
    return batch_loss, grad[1], batch_loss
end

function compute_loss(strategy::MultiStepRollout, model, batch, device)
    x_seq = batch  # Vector{GNNGraph} of length nsteps+1

    # noiseless_loss: single first-step MSE — consistent comparator across all strategies
    noiseless_loss = Flux.mse(model(x_seq[1]), x_seq[2].ndata.dynamic)

    # Multi-step rollout: gradients flow back through the full chain
    batch_loss, grad = Flux.withgradient(model) do m
        dyn_cur = x_seq[1].ndata.dynamic
        total_loss = 0.0f0
        for k in 1:strategy.nsteps
            if strategy.noise_scale > 0
                dyn_cur = dyn_cur .+ (Float32(strategy.noise_scale) .* randn(Float32, size(dyn_cur)) |> device)
            end
            g_k = Flux.ignore_derivatives() do
                GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
            end
            yhat_k = m(g_k)
            total_loss += Flux.mse(yhat_k, x_seq[k+1].ndata.dynamic)
            dyn_cur = yhat_k
        end
        total_loss / strategy.nsteps
    end
    return batch_loss, grad[1], noiseless_loss
end

function compute_loss(strategy::PushforwardRollout, model, batch, device)
    x_seq = batch  # Vector{GNNGraph} of length nsteps+1

    # noiseless_loss: single first-step MSE — consistent comparator across all strategies
    noiseless_loss = Flux.mse(model(x_seq[1]), x_seq[2].ndata.dynamic)

    # K-1 detached steps to reach a realistic drifted state, then one gradient step
    dyn_cur = x_seq[1].ndata.dynamic
    for k in 1:(strategy.nsteps - 1)
        if strategy.noise_scale > 0
            dyn_cur = dyn_cur .+ (Float32(strategy.noise_scale) .* randn(Float32, size(dyn_cur)) |> device)
        end
        g_k = GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
        dyn_cur = Flux.ignore_derivatives(() -> model(g_k))
    end

    # Final step: single gradient computation
    g_last = GNNGraph(x_seq[end-1], ndata=(; x_seq[end-1].ndata..., dynamic=dyn_cur))
    batch_loss, grad = Flux.withgradient(model) do m
        Flux.mse(m(g_last), x_seq[end].ndata.dynamic)
    end
    return batch_loss, grad[1], noiseless_loss
end

function compute_loss(strategy::ScheduledPushforward, model, batch, device)
    x_seq = batch
    noiseless_loss = Flux.mse(model(x_seq[1]), x_seq[2].ndata.dynamic)
    nsteps = strategy.current_nsteps

    dyn_cur = x_seq[1].ndata.dynamic
    for k in 1:(nsteps - 1)
        if strategy.noise_scale > 0
            dyn_cur = dyn_cur .+ (Float32(strategy.noise_scale) .* randn(Float32, size(dyn_cur)) |> device)
        end
        g_k = GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
        dyn_cur = Flux.ignore_derivatives(() -> model(g_k))
    end

    g_last = GNNGraph(x_seq[nsteps], ndata=(; x_seq[nsteps].ndata..., dynamic=dyn_cur))
    batch_loss, grad = Flux.withgradient(model) do m
        Flux.mse(m(g_last), x_seq[nsteps+1].ndata.dynamic)
    end
    return batch_loss, grad[1], noiseless_loss
end

function compute_loss(strategy::ScheduledRollout, model, batch, device)
    x_seq = batch
    noiseless_loss = Flux.mse(model(x_seq[1]), x_seq[2].ndata.dynamic)
    nsteps = strategy.current_nsteps
    batch_loss, grad = Flux.withgradient(model) do m
        dyn_cur = x_seq[1].ndata.dynamic
        total_loss = 0.0f0
        for k in 1:nsteps
            if strategy.noise_scale > 0
                dyn_cur = dyn_cur .+ (Float32(strategy.noise_scale) .* randn(Float32, size(dyn_cur)) |> device)
            end
            g_k = Flux.ignore_derivatives() do
                GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
            end
            yhat_k = m(g_k)
            total_loss += Flux.mse(yhat_k, x_seq[k+1].ndata.dynamic)
            dyn_cur = yhat_k
        end
        total_loss / nsteps
    end
    return batch_loss, grad[1], noiseless_loss
end

# ─── Validation loss (no gradients) ──────────────────────────────────────────
# Returns (strategy_loss, loss_1step). For non-rollout strategies both are equal.

function compute_val_loss(::TrainStrategy, model, batch, device)
    x, y = batch
    loss = Flux.mse(model(x), y.x)
    return loss, loss
end

function compute_val_loss(strategy::MultiStepRollout, model, batch, device)
    x_seq      = batch
    loss_1step = Flux.mse(model(x_seq[1]), x_seq[2].ndata.dynamic)
    dyn_cur    = x_seq[1].ndata.dynamic
    total_loss = 0.0f0
    for k in 1:strategy.nsteps
        g_k    = GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
        yhat_k = model(g_k)
        total_loss += Flux.mse(yhat_k, x_seq[k+1].ndata.dynamic)
        dyn_cur = yhat_k
    end
    return total_loss / strategy.nsteps, loss_1step
end

function compute_val_loss(strategy::PushforwardRollout, model, batch, device)
    x_seq      = batch
    loss_1step = Flux.mse(model(x_seq[1]), x_seq[2].ndata.dynamic)
    dyn_cur    = x_seq[1].ndata.dynamic
    for k in 1:(strategy.nsteps - 1)
        g_k     = GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
        dyn_cur = model(g_k)
    end
    g_last = GNNGraph(x_seq[end-1], ndata=(; x_seq[end-1].ndata..., dynamic=dyn_cur))
    return Flux.mse(model(g_last), x_seq[end].ndata.dynamic), loss_1step
end

function compute_val_loss(strategy::ScheduledRollout, model, batch, device)
    x_seq      = batch
    nsteps     = strategy.current_nsteps
    loss_1step = Flux.mse(model(x_seq[1]), x_seq[2].ndata.dynamic)
    dyn_cur    = x_seq[1].ndata.dynamic
    total_loss = 0.0f0
    for k in 1:nsteps
        g_k    = GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
        yhat_k = model(g_k)
        total_loss += Flux.mse(yhat_k, x_seq[k+1].ndata.dynamic)
        dyn_cur = yhat_k
    end
    return total_loss / nsteps, loss_1step
end

function compute_val_loss(strategy::ScheduledPushforward, model, batch, device)
    x_seq      = batch
    nsteps     = strategy.current_nsteps
    loss_1step = Flux.mse(model(x_seq[1]), x_seq[2].ndata.dynamic)
    dyn_cur    = x_seq[1].ndata.dynamic
    for k in 1:(nsteps - 1)
        g_k     = GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
        dyn_cur = model(g_k)
    end
    g_last = GNNGraph(x_seq[nsteps], ndata=(; x_seq[nsteps].ndata..., dynamic=dyn_cur))
    return Flux.mse(model(g_last), x_seq[nsteps+1].ndata.dynamic), loss_1step
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

    for epoch in 1:nepochs
        Flux.adjust!(opt, eta=schedule(epoch))
        step_schedule!(train_strategy, epoch)
        loss           = 0.0f0
        noiseless_loss = 0.0f0
        n_valid        = 0

        for batch in dl_train
            batch_loss, grads, nl_loss = compute_loss(train_strategy, model, batch, device)
            if !isnan(batch_loss)
                Flux.Optimise.update!(opt, model, grads)
                loss           += batch_loss
                noiseless_loss += nl_loss
                n_valid        += 1
            end
        end

        val_strat = 0.0f0
        val_1step = 0.0f0
        for batch in dl_valid_strategy
            s_l, o_l = compute_val_loss(train_strategy, model, batch, device)
            val_strat += s_l
            val_1step += o_l
        end
        val_strat /= length(dl_valid_strategy)
        val_1step /= length(dl_valid_strategy)

        loss           /= max(1, n_valid)
        noiseless_loss /= max(1, n_valid)
        next!(pr;
            showvalues=[(:epoch, epoch), (:train_loss, loss), (:val_strategy, val_strat),
                        (:val_1step, val_1step), (:noiseless, noiseless_loss)]
        )
        train_loss[epoch]          = loss
        valid_loss_strategy[epoch] = val_strat
        valid_loss_1step[epoch]    = val_1step
        loss_noiseless[epoch]      = noiseless_loss
    end

    return train_loss, loss_noiseless, valid_loss_strategy, valid_loss_1step
end

function train_settings_from_toml(d::Dict)
    s = TrainSettings()
    for f in fieldnames(TrainSettings)
        key = string(f)
        haskey(d, key) && setfield!(s, f, convert(fieldtype(TrainSettings, f), d[key]))
    end
    return s
end