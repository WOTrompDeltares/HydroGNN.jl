cd(@__DIR__)
using Pkg
Pkg.activate(".")

using GraphNeuralNetworks
using JLD2
using Statistics
using CUDA, cuDNN
using Flux
using Flux: onehotbatch, DataLoader, relu, swish
using ParameterSchedulers

using ProgressMeter
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

struct GNN
    enc
    proc
    dec
end

Flux.@layer GNN

function GNN(din::Int, dhidden::Int, dout::Int, nlayers::Int)
    return GNN(
        Dense(din=>dhidden, swish),
        GNNChain([GraphConv(dhidden=>dhidden, swish) for _ in 1:nlayers]...),
        Dense(dhidden=>dout)
    )
    # return GNN(
    #     GraphConv(din=>dhidden, swish),
    #     GraphConv(dhidden=>dhidden, swish),
    #     GraphConv(dhidden=>dout)
    # )
end

function (m::GNN)(g::GNNGraph, x_dym, x_static)
    x = cat(x_dym, x_static; dims=1)
    # x1 = m.enc(g, x)
    x1 = m.enc(x)
    x1 = m.proc(g, x1)
    # x1 = m.dec(g, x1)
    x1 = m.dec(x1)
    return x1 .+ x_dym
end

model = GNN(din, dhidden, dout, nhidden)

# AddParallel(l) = Parallel(+, identity, l)

# model = AddParallel(
#     GNNChain(
#         GraphConv(din_in=>dhidden),
#         GraphConv(dhidden=>dhidden),
#         GraphConv(dhidden=>dout))
# )

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

lr = 3e-3
lr_final = 1e-5
lr_step = 10
lr_decay = (lr_final/lr)^(lr_step/nepochs)

opt = Flux.setup(Adam(lr), model)

pr = Progress(nepochs, desc="Training Progress", showspeed=true)

schedule = Step(start = lr, decay = lr_decay, step_sizes = lr_step)

train_loss = zeros(nepochs)
valid_loss = zeros(nepochs)
loss_noiseless = zeros(nepochs)

for epoch in 1:nepochs
    Flux.adjust!(opt, eta=schedule(epoch))
    loss = 0.0f0
    val_loss = 0.0f0
    noiseless_loss = 0.0f0
    for (x,y) in dl_train
        yhat_noiseless = model(x, x.ndata.dynamic, x.ndata.static)
        noiseless_loss += Flux.mse(yhat_noiseless, y.x)

        x.ndata.dynamic .+= (noise * randn(size(x.ndata.dynamic)))|> device
        # x.x[1:2,:] .+= (noise * randn(size(x.x[1:2,:])))|> device
        batch_loss, grad = Flux.withgradient(model) do m
            yhat = m(x, x.ndata.dynamic, x.ndata.static)
            Flux.mse(yhat, y.x)
        end
        Flux.Optimise.update!(opt, model, grad[1])
        loss += batch_loss
    end

    for (x,y) in dl_valid
        yhat = model(x, x.ndata.dynamic, x.ndata.static)
        val_loss += Flux.mse(yhat, y.x)
    end

    loss /= length(dl_train)
    val_loss /= length(dl_valid)
    noiseless_loss /= length(dl_train)
    next!(pr;
        showvalues = [(:epoch, epoch), (:train_loss, loss), (:val_loss, val_loss), (:loss_noiseless, noiseless_loss)]
    )
    train_loss[epoch] = loss
    valid_loss[epoch] = val_loss
    loss_noiseless[epoch] = noiseless_loss
end

# %%

plot_loss(train_loss, loss_noiseless, valid_loss, "models/$(name)/loss_plot.png")


# %%

# movie_graphs(x_out, "predictions.mp4")
graph_gt, graph_pred = predict_trajectory("data/wave1d/valid.jld2", 1, model|> cpu)
movie_graphs_comp(graph_gt, graph_pred, "models/$(name)/pred_vs_gt.mp4")