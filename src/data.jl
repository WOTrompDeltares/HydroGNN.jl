using GraphNeuralNetworks
using Statistics
using JLD2

using Flux: onehotbatch

struct DynamicNormStats
    mu::Vector{Float32}   # mean per dynamic feature (length = ndyn_features)
    sigma::Vector{Float32} # std  per dynamic feature
end

abstract type NormStrategy end

struct GlobalNorm <: NormStrategy
    stats::DynamicNormStats
end

struct PerTrajectoryNorm <: NormStrategy end

strategy_to_dict(::GlobalNorm)       = Dict{String,Any}("name" => "GlobalNorm")
strategy_to_dict(::PerTrajectoryNorm) = Dict{String,Any}("name" => "PerTrajectoryNorm")

function compute_norm_stats(fn)
    # Accumulate sum, sum-of-squares and count per feature across all trajectories and timesteps
    s1 = nothing
    s2 = nothing
    n  = 0

    jldopen(fn, "r") do f
        for key in keys(f)
            traj = f[key]
            velocity  = traj["velocity_node"]   # (nnodes, nsteps)
            waterlevel = traj["waterlevel"]      # (nnodes, nsteps)

            has_tau = "tau" in keys(traj)
            nfeat  = has_tau ? 3 : 2
            nnodes = size(waterlevel, 1)
            nsteps = size(waterlevel, 2)

            if s1 === nothing
                s1 = zeros(Float64, nfeat)
                s2 = zeros(Float64, nfeat)
            end

            srcs = has_tau ? [waterlevel, velocity, traj["tau"]] : [waterlevel, velocity]
            for (fi, src) in enumerate(srcs)
                vals = vec(src)   # all nodes × all timesteps
                s1[fi] += sum(vals)
                s2[fi] += sum(vals .^ 2)
            end
            n += nnodes * nsteps
        end
    end

    mu    = Float32.(s1 ./ n)
    sigma = Float32.(sqrt.(max.(s2 ./ n .- (s1 ./ n).^2, 1e-8)))
    return DynamicNormStats(mu, sigma)
end

function read_trajectory(fn, traj_num::Int)
    jldopen(fn, "r") do f
        traj_group = f["trajectory_$(traj_num)"]

        velocity = traj_group["velocity_node"]
        waterlevel = traj_group["waterlevel"]
        mesh_pos = traj_group["mesh_pos"][1,:]
        node_type = traj_group["node_type"]
        bathymetry = traj_group["bathymetry"]
        tau = traj_group["tau"]

        edges = traj_group["edges"]
        edges = cat(edges, reverse(edges, dims=1), dims=2)

        return velocity, waterlevel, mesh_pos, node_type, bathymetry, edges, tau
    end
end


function _norm(x, mu, sigma)
    return (x .- mu) ./ sigma
end

function _compute_traj_norm_stats(velocity, waterlevel, tau=nothing)
    srcs = tau === nothing ? [waterlevel, velocity] : [waterlevel, velocity, tau]
    mu    = Float32[mean(vec(s)) for s in srcs]
    sigma = Float32[max(std(vec(s)), 1f-8) for s in srcs]
    return DynamicNormStats(mu, sigma)
end

resolve_stats(ns::GlobalNorm, velocity, waterlevel, tau=nothing) = ns.stats
resolve_stats(::PerTrajectoryNorm, velocity, waterlevel, tau=nothing) =
    _compute_traj_norm_stats(velocity, waterlevel, tau)

function load_data(fn, norm_strategy::NormStrategy)
    data_x = []
    data_y = []

    jldopen(fn, "r") do f
        for key in sort((keys(f)))
            println("Loading trajectory: $key")

            traj_group = f[key]

            velocity   = traj_group["velocity_node"]
            waterlevel = traj_group["waterlevel"]
            mesh_pos   = traj_group["mesh_pos"][1,:]
            node_type  = traj_group["node_type"]
            bathymetry = traj_group["bathymetry"]
            has_tau    = "tau" in keys(traj_group)
            if has_tau
                tau = traj_group["tau"]
            end

            edges = traj_group["edges"]
            edges = cat(edges, reverse(edges, dims=1), dims=2)

            bathymetry .= (bathymetry .- mean(bathymetry)) ./ std(bathymetry)
            mesh_pos   .= (mesh_pos .- minimum(mesh_pos)) ./ (maximum(mesh_pos) - minimum(mesh_pos))
            node_onehot = onehotbatch(node_type, collect(0:5))

            traj_stats = resolve_stats(norm_strategy, velocity, waterlevel, has_tau ? tau : nothing)

            mu_wl, mu_v = traj_stats.mu[1],    traj_stats.mu[2]
            s_wl,  s_v  = traj_stats.sigma[1], traj_stats.sigma[2]
            mu_t = has_tau ? traj_stats.mu[3]    : 0f0
            s_t  = has_tau ? traj_stats.sigma[3] : 1f0
            nnodes = size(waterlevel, 1)
            _forc(t) = has_tau ? reshape(_norm(tau[:,t], mu_t, s_t), 1, nnodes) :
                                 zeros(Float32, 0, nnodes)

            for ii in 1:(size(velocity,2)-1)
                data_static = hcat(bathymetry, mesh_pos, node_onehot')
                data_dym = hcat(_norm(waterlevel[:,ii], mu_wl, s_wl),
                                _norm(velocity[:,ii],   mu_v,  s_v))
                y = hcat(_norm(waterlevel[:,ii+1], mu_wl, s_wl),
                         _norm(velocity[:,ii+1],   mu_v,  s_v))

                push!(data_x, GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=data_dym', forcing=_forc(ii))))
                push!(data_y, GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=y'))
            end
        end
    end
    return data_x, data_y
end

function load_data_multistep(fn, norm_strategy::NormStrategy, nsteps::Int)
    data_x = []

    jldopen(fn, "r") do f
        for key in sort(keys(f))
            println("Loading trajectory: $key")

            traj_group = f[key]

            velocity   = traj_group["velocity_node"]
            waterlevel = traj_group["waterlevel"]
            mesh_pos   = traj_group["mesh_pos"][1,:]
            node_type  = traj_group["node_type"]
            bathymetry = traj_group["bathymetry"]
            has_tau    = "tau" in keys(traj_group)
            if has_tau
                tau = traj_group["tau"]
            end

            edges = traj_group["edges"]
            edges = cat(edges, reverse(edges, dims=1), dims=2)

            bathymetry .= (bathymetry .- mean(bathymetry)) ./ std(bathymetry)
            mesh_pos   .= (mesh_pos .- minimum(mesh_pos)) ./ (maximum(mesh_pos) - minimum(mesh_pos))
            node_onehot = onehotbatch(node_type, collect(0:5))
            data_static = hcat(bathymetry, mesh_pos, node_onehot')'

            traj_stats = resolve_stats(norm_strategy, velocity, waterlevel, has_tau ? tau : nothing)

            mu_wl, mu_v = traj_stats.mu[1], traj_stats.mu[2]
            s_wl,  s_v  = traj_stats.sigma[1], traj_stats.sigma[2]
            mu_t = has_tau ? traj_stats.mu[3]    : 0f0
            s_t  = has_tau ? traj_stats.sigma[3] : 1f0
            nnodes = size(waterlevel, 1)
            _forc(t) = has_tau ? reshape(_norm(tau[:,t], mu_t, s_t), 1, nnodes) :
                                 zeros(Float32, 0, nnodes)

            nT = size(velocity, 2)
            for ii in 1:(nT - nsteps)
                window = GNNGraph[]
                for kk in 0:nsteps
                    t = ii + kk
                    dyn = hcat(_norm(waterlevel[:, t], mu_wl, s_wl),
                               _norm(velocity[:, t],   mu_v,  s_v))
                    push!(window, GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]),
                                          ndata=(; static=data_static, dynamic=dyn', forcing=_forc(t))))
                end
                push!(data_x, window)
            end
        end
    end
    return data_x
end