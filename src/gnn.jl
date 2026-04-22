using GraphNeuralNetworks
using Flux
using Flux: swish

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
end

function (m::GNN)(g::GNNGraph, x_dym, x_static)
    x = cat(x_dym, x_static; dims=1)
    x1 = m.enc(x)
    x1 = m.proc(g, x1)
    x1 = m.dec(x1)
    return x1 .+ x_dym
end