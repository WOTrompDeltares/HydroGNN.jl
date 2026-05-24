using GraphNeuralNetworks
using JLD2
using TOML
using Statistics
using Flux: onehotbatch, cpu

function predict_trajectory(fn, traj_ind::Int, model, norm_strategy::NormStrategy; device=cpu)

    model = model |> device

    velocity, waterlevel, mesh_pos, node_type, bathymetry, edges, tau = read_trajectory(fn, traj_ind)
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

    # Pre-compute all forcing arrays and move to device as a batch (no transfers inside loop)
    forcings_dev = [_forc(t) |> device for t in 1:nsteps+1]

    # Build one template graph and move to device once — carries edge indices and static
    dyn0     = hcat(_n(waterlevel[:,1], mu_wl, s_wl), _n(velocity[:,1], mu_v, s_v))'
    template = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]),
                        ndata=(; static=data_static', dynamic=dyn0, forcing=_forc(1))) |> device
    static_dev = template.ndata.static  # reused every step, already on device

    # Rollout entirely on device — zero CPU↔GPU transfers inside the loop
    dyn_cur      = template.ndata.dynamic
    pred_dyn_dev = Vector{AbstractMatrix{Float32}}(undef, nsteps + 1)
    pred_dyn_dev[1] = dyn_cur
    for ii in 1:nsteps
        g_k     = GNNGraph(template, ndata=(; static=static_dev, dynamic=dyn_cur, forcing=forcings_dev[ii]))
        dyn_cur = model(g_k)
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
                                  ndata=(; static=data_static', dynamic=dyn, forcing=_forc(t)))

    function denorm_graph(g)
        dyn_phys  = hcat(_d(g.ndata.dynamic[1,:], mu_wl, s_wl),
                         _d(g.ndata.dynamic[2,:], mu_v,  s_v))'
        forc_phys = _d(g.ndata.forcing, mu_t, s_t)
        return GNNGraph(g, ndata=(; static=g.ndata.static, dynamic=dyn_phys, forcing=forc_phys))
    end

    gt   = denorm_graph.([make_graph(gt_dyn[t],   t) for t in 1:nsteps+1])
    pred = denorm_graph.([make_graph(pred_dyn[t], t) for t in 1:nsteps+1])

    return gt, pred

end

function evaluate_all_trajectories(data_file::String, model, output_dir::String, norm_strategy::NormStrategy; device=cpu)
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
            gt, pred = predict_trajectory(data_file, traj_num, model, norm_strategy; device=device)

            # Generate and save comparison movie
            output_file = joinpath(output_dir, "trajectory_$(traj_num)_comparison.mp4")
            println("Saving movie to: $output_file")
            movie_graphs_comp(gt, pred, output_file)
        end
    end

    println("\nEvaluation complete. All movies saved to: $output_dir")
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