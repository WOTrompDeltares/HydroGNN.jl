using GraphNeuralNetworks
using Flux
using Flux: swish
using TOML

Base.@kwdef mutable struct ModelSettings
    dhidden::Int = 32
    nhidden::Int = 5
    skip_connections::Bool = false
    din::Int = 0
    dout::Int = 0
end

function Base.show(io::IO, ms::ModelSettings)
    print(io, "ModelSettings(")
    fields = fieldnames(ModelSettings)
    for (i, f) in enumerate(fields)
        print(io, f, "=", getfield(ms, f))
        i < length(fields) && print(io, ", ")
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", ms::ModelSettings)
    println(io, "ModelSettings:")
    for f in fieldnames(ModelSettings)
        println(io, "  ", rpad(string(f), 18), " = ", getfield(ms, f))
    end
end

function save_model_settings(ms::ModelSettings, path::String)
    open(path, "w") do io
        TOML.print(io, Dict(string(f) => getfield(ms, f) for f in fieldnames(ModelSettings)))
    end
end

function load_model_settings(path::String)
    d = TOML.parsefile(path)
    s = ModelSettings()
    for f in fieldnames(ModelSettings)
        key = string(f)
        haskey(d, key) && setfield!(s, f, convert(fieldtype(ModelSettings, f), d[key]))
    end
    return s
end

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
        GNNChain([skip ? ResGraphConv(GraphConv(dhidden=>dhidden, swish)) :
                         GraphConv(dhidden=>dhidden, swish) for _ in 1:nlayers]...),
        Dense(dhidden=>dout)
    )
end

function (m::GNN)(g::GNNGraph)
    x_dym    = vcat(g.ndata.dynamic, g.ndata.forcing)
    x_static = g.ndata.static
    x = cat(x_dym, x_static; dims=1)
    x1 = m.enc(x)
    x1 = m.proc(g, x1)
    x1 = m.dec(x1)
    return x1 .+ g.ndata.dynamic
end