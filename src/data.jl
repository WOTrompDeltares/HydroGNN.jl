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
            velocity   = traj["velocity_node"]   # (nnodes, nsteps)
            waterlevel = traj["waterlevel"]       # (nnodes, nsteps)

            has_forcing = "forcing" in keys(traj)
            has_tau     = "tau"     in keys(traj)
            nfeat  = (has_forcing || has_tau) ? 3 : 2
            nnodes = size(waterlevel, 1)
            nsteps = size(waterlevel, 2)

            if s1 === nothing
                s1 = zeros(Float64, nfeat)
                s2 = zeros(Float64, nfeat)
            end

            tau_data = has_forcing ? traj["forcing"] : (has_tau ? traj["tau"] : nothing)
            srcs = tau_data !== nothing ? [waterlevel, velocity, tau_data] : [waterlevel, velocity]
            for (fi, src) in enumerate(srcs)
                vals = vec(src)
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

        velocity   = traj_group["velocity_node"]
        waterlevel = traj_group["waterlevel"]
        mesh_pos   = traj_group["mesh_pos"][1,:]
        node_type  = traj_group["node_type"]
        bathymetry = traj_group["bathymetry"]
        tau = "forcing" in keys(traj_group) ? traj_group["forcing"] :
              "tau"     in keys(traj_group) ? traj_group["tau"]     : nothing
        bound_cond     = "bound_cond"     in keys(traj_group) ? traj_group["bound_cond"]     : nothing
        bc_dyn_indices = "bc_dyn_indices" in keys(traj_group) ? Int.(traj_group["bc_dyn_indices"]) : nothing

        edges = traj_group["edges"]
        edges = cat(edges, reverse(edges, dims=1), dims=2)

        return velocity, waterlevel, mesh_pos, node_type, bathymetry, edges, tau, bound_cond, bc_dyn_indices
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

# Build a (ndyn × nnodes) Bool mask — true where BC override applies.
# bc_node_ids: indices of boundary nodes; bc_dyn_indices: which dynamic feature rows are prescribed.
function _make_bc_mask(ndyn, nnodes, bc_node_ids, bc_dyn_indices)
    m = zeros(Bool, ndyn, nnodes)
    if !isempty(bc_node_ids) && bc_dyn_indices !== nothing
        m[bc_dyn_indices, bc_node_ids] .= true
    end
    return m
end

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
            has_forcing = "forcing" in keys(traj_group)
            has_tau     = "tau"     in keys(traj_group)
            tau = has_forcing ? traj_group["forcing"] :
                  has_tau     ? traj_group["tau"]     : nothing
            bc_dyn_indices = "bc_dyn_indices" in keys(traj_group) ?
                             Int.(traj_group["bc_dyn_indices"]) : nothing
            bound_cond     = "bound_cond" in keys(traj_group) ?
                             traj_group["bound_cond"] : nothing

            edges = traj_group["edges"]
            edges = cat(edges, reverse(edges, dims=1), dims=2)

            bathymetry .= (bathymetry .- mean(bathymetry)) ./ std(bathymetry)
            mesh_pos   .= (mesh_pos .- minimum(mesh_pos)) ./ (maximum(mesh_pos) - minimum(mesh_pos))
            node_onehot = onehotbatch(node_type, collect(0:5))

            traj_stats = resolve_stats(norm_strategy, velocity, waterlevel, tau)

            mu_wl, mu_v = traj_stats.mu[1],    traj_stats.mu[2]
            s_wl,  s_v  = traj_stats.sigma[1], traj_stats.sigma[2]
            mu_t = tau !== nothing ? traj_stats.mu[3]    : 0f0
            s_t  = tau !== nothing ? traj_stats.sigma[3] : 1f0
            nnodes      = size(waterlevel, 1)
            bc_node_ids = findall(node_type .== 1)
            bc_mask     = _make_bc_mask(2, nnodes, bc_node_ids, bc_dyn_indices)

            # Next-step BC rows: shape (n_bc_rows, nnodes), zero for non-BC nodes.
            # Appended to forcing so the model sees where the boundary is heading.
            function _bc_next_forc(t)
                bound_cond === nothing && return zeros(Float32, 0, nnodes)
                next_t = min(t + 1, size(bound_cond, 2))
                bc_row = zeros(Float32, length(bc_dyn_indices), nnodes)
                for (r, dyn_row) in enumerate(bc_dyn_indices)
                    mu_d, s_d = traj_stats.mu[dyn_row], traj_stats.sigma[dyn_row]
                    for nid in bc_node_ids
                        bc_row[r, nid] = (bound_cond[r, next_t] - mu_d) / s_d
                    end
                end
                return bc_row
            end
            _forc(t) = tau !== nothing ?
                vcat(reshape(_norm(tau[:,t], mu_t, s_t), 1, nnodes), _bc_next_forc(t)) :
                _bc_next_forc(t)

            for ii in 1:(size(velocity,2)-1)
                data_static = hcat(bathymetry, mesh_pos, node_onehot')
                data_dym = hcat(_norm(waterlevel[:,ii], mu_wl, s_wl),
                                _norm(velocity[:,ii],   mu_v,  s_v))
                y = hcat(_norm(waterlevel[:,ii+1], mu_wl, s_wl),
                         _norm(velocity[:,ii+1],   mu_v,  s_v))

                push!(data_x, GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]),
                                       ndata=(; static=data_static', dynamic=data_dym',
                                               forcing=_forc(ii), bc_mask=bc_mask)))
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
            has_forcing = "forcing" in keys(traj_group)
            has_tau     = "tau"     in keys(traj_group)
            tau = has_forcing ? traj_group["forcing"] :
                  has_tau     ? traj_group["tau"]     : nothing
            bc_dyn_indices = "bc_dyn_indices" in keys(traj_group) ?
                             Int.(traj_group["bc_dyn_indices"]) : nothing
            bound_cond     = "bound_cond" in keys(traj_group) ?
                             traj_group["bound_cond"] : nothing

            edges = traj_group["edges"]
            edges = cat(edges, reverse(edges, dims=1), dims=2)

            bathymetry .= (bathymetry .- mean(bathymetry)) ./ std(bathymetry)
            mesh_pos   .= (mesh_pos .- minimum(mesh_pos)) ./ (maximum(mesh_pos) - minimum(mesh_pos))
            node_onehot = onehotbatch(node_type, collect(0:5))
            data_static = hcat(bathymetry, mesh_pos, node_onehot')'

            traj_stats = resolve_stats(norm_strategy, velocity, waterlevel, tau)

            mu_wl, mu_v = traj_stats.mu[1], traj_stats.mu[2]
            s_wl,  s_v  = traj_stats.sigma[1], traj_stats.sigma[2]
            mu_t = tau !== nothing ? traj_stats.mu[3]    : 0f0
            s_t  = tau !== nothing ? traj_stats.sigma[3] : 1f0
            nnodes      = size(waterlevel, 1)
            bc_node_ids = findall(node_type .== 1)
            bc_mask     = _make_bc_mask(2, nnodes, bc_node_ids, bc_dyn_indices)

            # Next-step BC rows: shape (n_bc_rows, nnodes), zero for non-BC nodes.
            # Appended to forcing so the model sees where the boundary is heading.
            function _bc_next_forc(t)
                bound_cond === nothing && return zeros(Float32, 0, nnodes)
                next_t = min(t + 1, size(bound_cond, 2))
                bc_row = zeros(Float32, length(bc_dyn_indices), nnodes)
                for (r, dyn_row) in enumerate(bc_dyn_indices)
                    mu_d, s_d = traj_stats.mu[dyn_row], traj_stats.sigma[dyn_row]
                    for nid in bc_node_ids
                        bc_row[r, nid] = (bound_cond[r, next_t] - mu_d) / s_d
                    end
                end
                return bc_row
            end
            _forc(t) = tau !== nothing ?
                vcat(reshape(_norm(tau[:,t], mu_t, s_t), 1, nnodes), _bc_next_forc(t)) :
                _bc_next_forc(t)

            nT = size(velocity, 2)
            for ii in 1:(nT - nsteps)
                window = GNNGraph[]
                for kk in 0:nsteps
                    t = ii + kk
                    dyn = hcat(_norm(waterlevel[:, t], mu_wl, s_wl),
                               _norm(velocity[:, t],   mu_v,  s_v))
                    push!(window, GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]),
                                          ndata=(; static=data_static, dynamic=dyn',
                                                  forcing=_forc(t), bc_mask=bc_mask)))
                end
                push!(data_x, window)
            end
        end
    end
    return data_x
end

"""
    collate_multistep_batch(windows)

Custom DataLoader collate function for multistep datasets.

Takes a vector of B windows (each a `Vector{GNNGraph}` of length `nsteps+1`)
and returns a single `Vector{GNNGraph}` of length `nsteps+1` where each
element is a batched graph combining all B windows at that step position.

This enables proper GPU utilization by fusing B small graphs into one
large disconnected graph per step, rather than iterating one window at a time.
"""
function collate_multistep_batch(windows::Vector)
    nsteps_plus1 = length(first(windows))
    return [batch([windows[b][k] for b in eachindex(windows)]) for k in 1:nsteps_plus1]
end