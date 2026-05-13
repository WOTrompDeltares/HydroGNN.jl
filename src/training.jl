using Flux
using JLD2
using ProgressMeter
using ParameterSchedulers
using TOML

abstract type TrainStrategy end

struct SingleStepNoise <: TrainStrategy
    scale::Float64
end

struct NoNoise <: TrainStrategy end

struct MultiStepRollout <: TrainStrategy
    nsteps::Int
    noise_scale::Float64
end

struct PushforwardRollout <: TrainStrategy
    nsteps::Int
    noise_scale::Float64
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
            g_k = GNNGraph(x_seq[k], ndata=(; x_seq[k].ndata..., dynamic=dyn_cur))
            yhat_k = m(g_k)
            total_loss += Flux.mse(yhat_k, x_seq[k+1].ndata.dynamic)
            dyn_cur = yhat_k
        end
        total_loss
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

Base.@kwdef mutable struct TrainSettings
    dhidden::Int = 32
    nhidden::Int = 5
    nepochs::Int = 250
    lr::Float64 = 3e-3
    lr_final::Float64 = 1e-5
    lr_step::Int = 10
    nbatch::Int = 32
    skip_connections::Bool = false
    train_data_path::String = "data/wave1d/train.jld2"
    valid_data_path::String = "data/wave1d/valid.jld2"
    model_name::String = "test_run3d"
    save_dir::String = "models"
end

function train_model!(model, dl_train, dl_valid, device, settings::TrainSettings, norm_strategy::NormStrategy, train_strategy::TrainStrategy)
    nepochs = settings.nepochs
    lr = settings.lr
    lr_final = settings.lr_final
    lr_step = settings.lr_step

    lr_decay = (lr_final / lr)^(lr_step / nepochs)
    opt = Flux.setup(Adam(lr), model)
    schedule = Step(start=lr, decay=lr_decay, step_sizes=lr_step)
    pr = Progress(nepochs, desc="Training Progress", showspeed=true)

    train_loss = zeros(nepochs)
    valid_loss = zeros(nepochs)
    loss_noiseless = zeros(nepochs)

    for epoch in 1:nepochs
        Flux.adjust!(opt, eta=schedule(epoch))
        loss = 0.0f0
        val_loss = 0.0f0
        noiseless_loss = 0.0f0

        for batch in dl_train
            batch_loss, grads, nl_loss = compute_loss(train_strategy, model, batch, device)
            Flux.Optimise.update!(opt, model, grads)
            loss += batch_loss
            noiseless_loss += nl_loss
        end

        for (x, y) in dl_valid
            yhat = model(x)
            val_loss += Flux.mse(yhat, y.x)
        end

        loss /= length(dl_train)
        val_loss /= length(dl_valid)
        noiseless_loss /= length(dl_train)
        next!(pr;
            showvalues=[(:epoch, epoch), (:train_loss, loss), (:val_loss, val_loss), (:loss_noiseless, noiseless_loss)]
        )
        train_loss[epoch] = loss
        valid_loss[epoch] = val_loss
        loss_noiseless[epoch] = noiseless_loss
    end

    model_dir = joinpath(settings.save_dir, settings.model_name)
    mkpath(model_dir)

    jldsave(joinpath(model_dir, "model.jld2");
        model          = model |> cpu,
        norm_strategy  = norm_strategy,
        train_strategy = train_strategy)

    open(joinpath(model_dir, "settings.toml"), "w") do io
        TOML.print(io, Dict(string(f) => getfield(settings, f) for f in fieldnames(TrainSettings)))
    end

    return train_loss, loss_noiseless, valid_loss
end