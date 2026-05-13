using GraphNeuralNetworks
using JLD2
using Statistics
using Flux: onehotbatch

function predict_trajectory(fn, traj_ind::Int, model, norm_strategy::NormStrategy)

    velocity, waterlevel, mesh_pos, node_type, bathymetry, edges, tau = read_trajectory(fn, traj_ind)
    nsteps = size(velocity,2) - 1

    bathymetry .= (bathymetry .- mean(bathymetry)) ./ std(bathymetry)
    mesh_pos   .= (mesh_pos .- minimum(mesh_pos)) ./ (maximum(mesh_pos) - minimum(mesh_pos))

    data_static = hcat(bathymetry, mesh_pos, onehotbatch(node_type, collect(0:5))')

    ns = resolve_stats(norm_strategy, velocity, waterlevel, tau)

    mu_wl, mu_v = ns.mu[1],    ns.mu[2]
    s_wl,  s_v  = ns.sigma[1], ns.sigma[2]
    has_tau = length(ns.mu) >= 3
    if has_tau
        mu_t, s_t = ns.mu[3], ns.sigma[3]
    end

    _n(x, mu, s) = (x .- mu) ./ s
    _d(x, mu, s) = x .* s .+ mu

    gt   = []
    pred = []

    # Initial state in normalized space
    wl0_norm = _n(waterlevel[:,1], mu_wl, s_wl)
    v0_norm  = _n(velocity[:,1],   mu_v,  s_v)
    if has_tau
        tau0_norm = _n(tau[:,1], mu_t, s_t)
        dyn0 = hcat(wl0_norm, v0_norm, tau0_norm)
    else
        dyn0 = hcat(wl0_norm, v0_norm)
    end

    x0 = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=dyn0'))
    push!(gt,   x0)
    push!(pred, x0)

    for ii in 1:nsteps
        println("Predicting time step: $ii")
        # yhat is in normalized space (waterlevel, velocity rows)
        yhat_norm = model(pred[end], pred[end].ndata.dynamic, pred[end].ndata.static)

        # Build next input: yhat_norm for wl/v, plus normalized tau at next step
        if has_tau
            tau_next_norm = _n(tau[:,ii+1], mu_t, s_t)
            dyn_next = hcat(yhat_norm', tau_next_norm)
        else
            dyn_next = yhat_norm'
        end
        x_next = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=dyn_next'))
        push!(pred, x_next)

        # Ground truth in normalized space for consistent comparison
        wl_gt_norm = _n(waterlevel[:,ii+1], mu_wl, s_wl)
        v_gt_norm  = _n(velocity[:,ii+1],   mu_v,  s_v)
        if has_tau
            tau_gt_norm = _n(tau[:,ii+1], mu_t, s_t)
            dyn_gt = hcat(wl_gt_norm, v_gt_norm, tau_gt_norm)
        else
            dyn_gt = hcat(wl_gt_norm, v_gt_norm)
        end
        x_gt = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=dyn_gt'))
        push!(gt, x_gt)
    end

    # Denormalize predictions and ground truth back to physical units
    function denorm_graph(g)
        dyn = g.ndata.dynamic  # (nfeat, nnodes)
        dyn_phys = similar(dyn)
        dyn_phys[1,:] = _d(dyn[1,:], mu_wl, s_wl)
        dyn_phys[2,:] = _d(dyn[2,:], mu_v,  s_v)
        if has_tau
            dyn_phys[3,:] = _d(dyn[3,:], mu_t, s_t)
        end
        return GNNGraph(g, ndata=(; static=g.ndata.static, dynamic=dyn_phys))
    end

    gt_phys   = denorm_graph.(gt)
    pred_phys = denorm_graph.(pred)

    return gt_phys, pred_phys

end

function evaluate_all_trajectories(data_file::String, model, output_dir::String, norm_strategy::NormStrategy)
    # Create output directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    # Get all trajectory keys from the data file
    trajectory_keys = []
    jldopen(data_file, "r") do f
        trajectory_keys = sort(collect(keys(f)))
    end

    # Process each trajectory
    for (idx, key) in enumerate(trajectory_keys)
        if startswith(key, "trajectory_")
            # Extract trajectory number from key
            traj_num = parse(Int, split(key, "_")[2])
            println("\nEvaluating trajectory $idx: $key (traj_num=$traj_num)")

            # Predict trajectory
            gt, pred = predict_trajectory(data_file, traj_num, model, norm_strategy)

            # Generate and save comparison movie
            output_file = joinpath(output_dir, "trajectory_$(traj_num)_comparison.mp4")
            println("Saving movie to: $output_file")
            movie_graphs_comp(gt, pred, output_file)
        end
    end

    println("\nEvaluation complete. All movies saved to: $output_dir")
end