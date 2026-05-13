using Flux
using JLD2
using ProgressMeter
using ParameterSchedulers
using TOML

Base.@kwdef mutable struct TrainSettings
    dhidden::Int = 32
    nhidden::Int = 5
    nepochs::Int = 250
    noise::Float64 = 2.5e-2
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

function train_model!(model, dl_train, dl_valid, device, settings::TrainSettings, norm_strategy::NormStrategy)
    nepochs = settings.nepochs
    noise = settings.noise
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

        for (x, y) in dl_train
            yhat_noiseless = model(x, x.ndata.dynamic, x.ndata.static)
            noiseless_loss += Flux.mse(yhat_noiseless, y.x)

            x.ndata.dynamic .+= (noise * randn(size(x.ndata.dynamic))) |> device
            batch_loss, grad = Flux.withgradient(model) do m
                yhat = m(x, x.ndata.dynamic, x.ndata.static)
                Flux.mse(yhat, y.x)
            end
            Flux.Optimise.update!(opt, model, grad[1])
            loss += batch_loss
        end

        for (x, y) in dl_valid
            yhat = model(x, x.ndata.dynamic, x.ndata.static)
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
        model         = model |> cpu,
        norm_strategy = norm_strategy)

    open(joinpath(model_dir, "settings.toml"), "w") do io
        TOML.print(io, Dict(string(f) => getfield(settings, f) for f in fieldnames(TrainSettings)))
    end

    return train_loss, loss_noiseless, valid_loss
end