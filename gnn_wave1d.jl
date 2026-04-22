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

train_x, train_y = load_data("data/wave1d/train.jld2")
val_x, val_y = load_data("data/wave1d/valid.jld2")

# %%

nbatch = 32

dl_train = DataLoader((train_x, train_y), batchsize=nbatch, shuffle=true, collate=true)
dl_valid = DataLoader((val_x, val_y), batchsize=nbatch, shuffle=false, collate=true)

din = 10
dhidden = 32
dout = 2
nhidden = 5

model = GNN(din, dhidden, dout, nhidden)

model(first(dl_train)[1], first(dl_train)[1].ndata.dynamic, first(dl_train)[1].ndata.static)

# %%

name = "test_run3d"

if !isdir(joinpath("models", name))
    mkpath(joinpath("models", name))
end


nepochs = 250
noise = 2.5e-2

model = model |> device
dl_train = dl_train |> device
dl_valid = dl_valid |> device

train_loss, loss_noiseless, valid_loss = train_model!(model, dl_train, dl_valid, device;
    nepochs=nepochs, noise=noise, lr=3e-3, lr_final=1e-5, lr_step=10)

# %%

plot_loss(train_loss, loss_noiseless, valid_loss, "models/$(name)/loss_plot.png")


# %%

# movie_graphs(x_out, "predictions.mp4")
graph_gt, graph_pred = predict_trajectory("data/wave1d/valid.jld2", 1, model|> cpu)
movie_graphs_comp(graph_gt, graph_pred, "models/$(name)/pred_vs_gt.mp4")