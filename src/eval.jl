using GraphNeuralNetworks
using JLD2
using TOML
using Statistics
using Flux: onehotbatch, cpu

function predict_trajectory(fn, traj_ind::Int, model, norm_strategy::NormStrategy; device=cpu)

    model = model |> device

    velocity, waterlevel, mesh_pos, node_type, bathymetry, edges, tau, bound_cond, bc_dyn_indices =
        read_trajectory(fn, traj_ind)
    nsteps = size(velocity, 2) - 1

    bathymetry .= (bathymetry .- mean(bathymetry)) ./ std(bathymetry)
    mesh_pos   .= (mesh_pos .- minimum(mesh_pos)) ./ (maximum(mesh_pos) - minimum(mesh_pos))

    data_static = hcat(bathymetry, mesh_pos, onehotbatch(node_type, collect(0:5))')

    ns = resolve_stats(norm_strategy, velocity, waterlevel, tau)

    mu_wl, mu_v = ns.mu[1],    ns.mu[2]
    s_wl,  s_v  = ns.sigma[1], ns.sigma[2]
    has_tau = length(ns.mu) >= 3
    mu_t = has_tau ? ns.mu[3]    : 0f0
    s_t  = has_tau ? ns.sigma[3] : 1f0
    nnodes = size(waterlevel, 1)

    _n(x, mu, s) = (x .- mu) ./ s
    _d(x, mu, s) = x .* s .+ mu
    _forc(t) = has_tau ? reshape(_n(tau[:,t], mu_t, s_t), 1, nnodes) :
                         zeros(Float32, 0, nnodes)

    # Build BC mask and per-step normalized BC values for the override
    bc_node_ids = findall(node_type .== 1)
    bc_mask     = _make_bc_mask(2, nnodes, bc_node_ids, bc_dyn_indices)
    has_bc      = any(bc_mask)

    # Normalize bound_cond using stats of the targeted dynamic variable(s)
    dyn_stats = [(ns.mu[i], ns.sigma[i]) for i in 1:2]  # indexed by dynamic feature row
    function _norm_bc(t)
        bound_cond === nothing && return zeros(Float32, 2, nnodes)
        out = zeros(Float32, 2, nnodes)
        for (row, dyn_row) in enumerate(bc_dyn_indices)
            mu_d, s_d = dyn_stats[dyn_row]
            for (col, nid) in enumerate(bc_node_ids)
                out[dyn_row, nid] = _n(bound_cond[row, t], mu_d, s_d)
            end
        end
        return out
    end

    # Pre-compute all forcing and BC arrays; move to device as a batch
    forcings_dev = [_forc(t) |> device for t in 1:nsteps+1]
    bc_vals_dev  = has_bc ? [_norm_bc(t) |> device for t in 1:nsteps+1] : nothing
    bc_mask_dev  = bc_mask |> device

    # Build one template graph and move to device once
    dyn0     = hcat(_n(waterlevel[:,1], mu_wl, s_wl), _n(velocity[:,1], mu_v, s_v))'
    template = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]),
                        ndata=(; static=data_static', dynamic=dyn0,
                                forcing=_forc(1), bc_mask=bc_mask)) |> device
    static_dev = template.ndata.static

    # Rollout entirely on device
    dyn_cur      = template.ndata.dynamic
    pred_dyn_dev = Vector{AbstractMatrix{Float32}}(undef, nsteps + 1)
    pred_dyn_dev[1] = dyn_cur
    for ii in 1:nsteps
        g_k     = GNNGraph(template, ndata=(; static=static_dev, dynamic=dyn_cur,
                                              forcing=forcings_dev[ii], bc_mask=bc_mask_dev))
        dyn_cur = model(g_k)
        # Apply BC override: replace predicted values at prescribed positions
        if has_bc
            bc_next = bc_vals_dev[ii+1]
            dyn_cur = copy(dyn_cur)
            dyn_cur[bc_mask_dev] .= bc_next[bc_mask_dev]
        end
        pred_dyn_dev[ii+1] = dyn_cur
    end

    # Move all predicted dynamics to CPU in one batch after the loop
    pred_dyn = [d |> cpu for d in pred_dyn_dev]

    # Ground-truth dynamics (CPU, no device needed)
    gt_dyn = [hcat(_n(waterlevel[:,t], mu_wl, s_wl), _n(velocity[:,t], mu_v, s_v))' for t in 1:nsteps+1]

    # Reconstruct CPU GNNGraphs for downstream use (plotting etc.)
    edges_src = Int64.(edges[1,:])
    edges_dst = Int64.(edges[2,:])
    make_graph(dyn, t) = GNNGraph(edges_src, edges_dst,
                                  ndata=(; static=data_static', dynamic=dyn,
                                          forcing=_forc(t), bc_mask=bc_mask))

    function denorm_graph(g)
        dyn_phys  = hcat(_d(g.ndata.dynamic[1,:], mu_wl, s_wl),
                         _d(g.ndata.dynamic[2,:], mu_v,  s_v))'
        forc_phys = _d(g.ndata.forcing, mu_t, s_t)
        return GNNGraph(g, ndata=(; static=g.ndata.static, dynamic=dyn_phys, forcing=forc_phys,
                                    bc_mask=g.ndata.bc_mask))
    end

    gt   = denorm_graph.([make_graph(gt_dyn[t],   t) for t in 1:nsteps+1])
    pred = denorm_graph.([make_graph(pred_dyn[t], t) for t in 1:nsteps+1])

    return gt, pred

end

# NaN-safe per-step RMSE profile for one trajectory pair.
# Steps at and after divergence (NaN/Inf values) become Inf.
function _traj_rmse_profile(gt::Vector, pred::Vector)
    T       = length(gt)
    profile = fill(Inf, T)
    for t in 1:T
        diff = gt[t].ndata.dynamic .- pred[t].ndata.dynamic
        if any(isnan, diff) || any(isinf, diff)
            break
        end
        profile[t] = sqrt(mean(diff .^ 2))
    end
    return profile
end

"""
    evaluate_all_trajectories(data_file, model, output_dir, norm_strategy; device=cpu)
    -> (mean_final_rmse, rmse_profile)

Runs autoregressive rollout over every trajectory, saves comparison movies to
`output_dir`, and returns rollout stability metrics:
- `mean_final_rmse`: mean RMSE at the final timestep (Inf if any trajectory diverged)
- `rmse_profile`:    per-step mean RMSE vector
"""
function evaluate_all_trajectories(data_file::String, model, output_dir::String, norm_strategy::NormStrategy; device=cpu)
    mkpath(output_dir)

    trajectory_keys = String[]
    jldopen(data_file, "r") do f
        trajectory_keys = sort(filter(k -> startswith(k, "trajectory_"), collect(keys(f))))
    end

    all_profiles = Vector{Vector{Float64}}()

    for (idx, key) in enumerate(trajectory_keys)
        traj_num = parse(Int, split(key, "_")[2])
        println("\nEvaluating trajectory $idx: $key (traj_num=$traj_num)")

        gt, pred = predict_trajectory(data_file, traj_num, model, norm_strategy; device=device)
        push!(all_profiles, _traj_rmse_profile(gt, pred))

        output_file = joinpath(output_dir, "trajectory_$(traj_num)_comparison.mp4")
        println("Saving movie to: $output_file")
        movie_graphs_comp(gt, pred, output_file)
    end

    println("\nEvaluation complete. All movies saved to: $output_dir")

    isempty(all_profiles) && return (Inf, Float64[])
    T_min        = minimum(length(p) for p in all_profiles)
    rmse_profile = [mean(p[t] for p in all_profiles) for t in 1:T_min]
    return rmse_profile[end], rmse_profile
end

"""
    rollout_metrics(data_file, model, norm_strategy; device=cpu)
    -> (mean_final_rmse, rmse_profile)

Metrics-only version of `evaluate_all_trajectories` — no movies saved.
"""
function rollout_metrics(data_file::String, model, norm_strategy::NormStrategy; device=cpu)
    trajectory_keys = String[]
    jldopen(data_file, "r") do f
        trajectory_keys = sort(filter(k -> startswith(k, "trajectory_"), collect(keys(f))))
    end

    all_profiles = Vector{Vector{Float64}}()
    for key in trajectory_keys
        traj_num = parse(Int, split(key, "_")[2])
        gt, pred = predict_trajectory(data_file, traj_num, model, norm_strategy; device=device)
        push!(all_profiles, _traj_rmse_profile(gt, pred))
    end

    isempty(all_profiles) && return (Inf, Float64[])
    T_min        = minimum(length(p) for p in all_profiles)
    rmse_profile = [mean(p[t] for p in all_profiles) for t in 1:T_min]
    return rmse_profile[end], rmse_profile
end

function load_run(model_dir::String)
    d              = JLD2.load(joinpath(model_dir, "model.jld2"))
    model          = d["model"]
    norm_strategy  = d["norm_strategy"]
    train_strategy = d["train_strategy"]

    train_settings = train_settings_from_toml(TOML.parsefile(joinpath(model_dir, "train_settings.toml")))
    model_settings = load_model_settings(joinpath(model_dir, "model_settings.toml"))

    return model, model_settings, train_settings, norm_strategy, train_strategy
end