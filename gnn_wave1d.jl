cd(@__DIR__)
using Pkg
Pkg.activate(".")

using GraphNeuralNetworks
using JLD2
using Statistics
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

settings.model_name = "test_3"
settings.skip_connections = true
settings.nepochs = 200
settings.nbatch = 32
settings.dhidden = 16
settings.nhidden = 5
settings.save_dir = "models/lake1d_surge"


settings.train_data_path = "data/lake1d_surge/train.jld2"
settings.valid_data_path = "data/lake1d_surge/valid.jld2"

norm_strategy  = GlobalNorm(compute_norm_stats(settings.train_data_path))  # or: PerTrajectoryNorm()
train_strategy = SingleStepNoise(3e-2)  # or: NoNoise(), MultiStepRollout(5, 0.0)

val_x, val_y = load_data(settings.valid_data_path, norm_strategy)

if train_strategy isa MultiStepRollout
    train_x = load_data_multistep(settings.train_data_path, norm_strategy, train_strategy.nsteps)
    dl_train = DataLoader(train_x, batchsize=1, shuffle=true, collate=false)
else
    train_x, train_y = load_data(settings.train_data_path, norm_strategy)
    dl_train = DataLoader((train_x, train_y), batchsize=settings.nbatch, shuffle=true, collate=true)
end
dl_valid = DataLoader((val_x, val_y), batchsize=settings.nbatch, shuffle=false, collate=true)

# %%

din = size(val_x[1].ndata.dynamic, 1) + size(val_x[1].ndata.static, 1)
dout = size(val_y[1].ndata.x, 1)

model = GNN(din, settings.dhidden, dout, settings.nhidden; skip=settings.skip_connections)

model(val_x[1], val_x[1].ndata.dynamic, val_x[1].ndata.static)

# %%

name = settings.model_name
model_dir = joinpath(settings.save_dir, name)

if !isdir(model_dir)
    mkpath(model_dir)
end


model = model |> device
dl_train = dl_train |> device
dl_valid = dl_valid |> device

train_loss, loss_noiseless, valid_loss = train_model!(model, dl_train, dl_valid, device, settings, norm_strategy, train_strategy)

# %%

plot_loss(train_loss, loss_noiseless, valid_loss, joinpath(model_dir, "loss_plot.png"))


# %%

evaluate_all_trajectories(settings.valid_data_path, model |> cpu, model_dir, norm_strategy)