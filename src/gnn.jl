using GraphNeuralNetworks
using Flux
using Flux: swish

struct ResGraphConv <: GNNLayer
    layer
end

Flux.@layer ResGraphConv

(m::ResGraphConv)(g::GNNGraph, x) = m.layer(g, x) .+ x

struct GNN
    enc
    proc
    dec
end

Flux.@layer GNN

function GNN(din::Int, dhidden::Int, dout::Int, nlayers::Int; skip::Bool=false)
    return GNN(
        Dense(din=>dhidden, swish),
        GNNChain([skip ? ResGraphConv(SAGEConv(dhidden=>dhidden, swish)) :
                         SAGEConv(dhidden=>dhidden, swish) for _ in 1:nlayers]...),
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