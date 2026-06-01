cd(@__DIR__)
using Pkg
Pkg.activate(".")

using GraphNeuralNetworks
using JLD2
using Statistics
using TOML
using CUDA, cuDNN
using Flux
using Flux: onehotbatch, DataLoader, relu, swish
using HydroGNN

if CUDA.has_cuda() && CUDA.functional()
    println("CUDA is available. Using GPU.")
    device = gpu
else
    println("CUDA is not available. Using CPU.")
    device = cpu
end

# %%

settings = TrainSettings()
settings.model_name       = "test_gpu_rollout"
settings.nepochs          = 5
settings.nbatch           = 32
settings.save_dir         = "models/lake1d_surge"
settings.train_data_path  = "data/lake1d_surge/train.jld2"
settings.valid_data_path  = "data/lake1d_surge/valid.jld2"

model_settings = ModelSettings()
model_settings.skip_connections = true
model_settings.dhidden          = 16
model_settings.nhidden          = 5

norm_strategy  = GlobalNorm(compute_norm_stats(settings.train_data_path))  # or: PerTrajectoryNorm()
train_strategy = SingleStepNoise(3e-2)  # or: NoNoise(), MultiStepRollout(5, 0.0), PushforwardRollout(5, 0.0)
# train_strategy = PushforwardRollout(3, 0.0)

val_x, val_y = load_data(settings.valid_data_path, norm_strategy)

if train_strategy isa RolloutStrategy
    train_x           = load_data_multistep(settings.train_data_path, norm_strategy, train_strategy.nsteps)
    dl_train          = DataLoader(train_x, batchsize=settings.nbatch, shuffle=true, collate=collate_multistep_batch, parallel=true)
    valid_x_ms        = load_data_multistep(settings.valid_data_path, norm_strategy, train_strategy.nsteps)
    dl_valid_strategy = DataLoader(valid_x_ms, batchsize=settings.nbatch, shuffle=false, collate=collate_multistep_batch, parallel=true)
else
    train_x, train_y  = load_data(settings.train_data_path, norm_strategy)
    dl_train          = DataLoader((train_x, train_y), batchsize=settings.nbatch, shuffle=true, collate=true, parallel=true)
    dl_valid_strategy = DataLoader((val_x, val_y), batchsize=settings.nbatch, shuffle=false, collate=true, parallel=true)
end

# %%

din = size(val_x[1].ndata.dynamic, 1) + size(val_x[1].ndata.forcing, 1) + size(val_x[1].ndata.static, 1)
dout = size(val_y[1].ndata.x, 1)

model_settings.din  = din
model_settings.dout = dout
model = GNN(model_settings.din, model_settings.dhidden, model_settings.dout, model_settings.nhidden; skip=model_settings.skip_connections)

model(val_x[1])

# %%

name = settings.model_name
model_dir = joinpath(settings.save_dir, name)

if !isdir(model_dir)
    mkpath(model_dir)
end


model             = model |> device
dl_train          = dl_train |> device
dl_valid_strategy = dl_valid_strategy |> device

train_loss, loss_noiseless, valid_loss_strategy, valid_loss_1step = train_model!(model, dl_train, dl_valid_strategy, device, settings, norm_strategy, train_strategy)

jldsave(joinpath(model_dir, "model.jld2");
    model          = model |> cpu,
    norm_strategy  = norm_strategy,
    train_strategy = train_strategy)

open(joinpath(model_dir, "train_settings.toml"), "w") do io
    d = Dict{String,Any}(string(f) => getfield(settings, f) for f in fieldnames(TrainSettings))
    d["norm_strategy"]  = strategy_to_dict(norm_strategy)
    d["train_strategy"] = strategy_to_dict(train_strategy)
    TOML.print(io, d)
end
save_model_settings(model_settings, joinpath(model_dir, "model_settings.toml"))

# %%

plot_loss(train_loss, loss_noiseless, valid_loss_strategy, joinpath(model_dir, "loss_plot.png"); val_loss_1step=valid_loss_1step)


# %%

evaluate_all_trajectories(settings.valid_data_path, model, model_dir, norm_strategy; device=device)