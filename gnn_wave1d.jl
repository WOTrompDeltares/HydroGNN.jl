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

settings.model_name = "test_run4"

train_x, train_y = load_data(settings.train_data_path)
val_x, val_y = load_data(settings.valid_data_path)

# %%

dl_train = DataLoader((train_x, train_y), batchsize=settings.nbatch, shuffle=true, collate=true)
dl_valid = DataLoader((val_x, val_y), batchsize=settings.nbatch, shuffle=false, collate=true)

din = size(train_x[1].ndata.dynamic, 1) + size(train_x[1].ndata.static, 1)
dout = size(train_y[1].ndata.x, 1)

model = GNN(din, settings.dhidden, dout, settings.nhidden)

model(first(dl_train)[1], first(dl_train)[1].ndata.dynamic, first(dl_train)[1].ndata.static)

# %%

name = settings.model_name
model_dir = joinpath(settings.save_dir, name)

if !isdir(model_dir)
    mkpath(model_dir)
end


model = model |> device
dl_train = dl_train |> device
dl_valid = dl_valid |> device

train_loss, loss_noiseless, valid_loss = train_model!(model, dl_train, dl_valid, device, settings)

# %%

plot_loss(train_loss, loss_noiseless, valid_loss, joinpath(model_dir, "loss_plot.png"))


# %%

evaluate_all_trajectories(settings.valid_data_path, model |> cpu, model_dir)