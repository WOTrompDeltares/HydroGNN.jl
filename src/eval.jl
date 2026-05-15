using GraphNeuralNetworks
using JLD2
using TOML
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
    mu_t = has_tau ? ns.mu[3]    : 0f0
    s_t  = has_tau ? ns.sigma[3] : 1f0
    nnodes = size(waterlevel, 1)

    _n(x, mu, s) = (x .- mu) ./ s
    _d(x, mu, s) = x .* s .+ mu
    _forc(t) = has_tau ? reshape(_n(tau[:,t], mu_t, s_t), 1, nnodes) :
                         zeros(Float32, 0, nnodes)

    gt   = []
    pred = []

    dyn0 = hcat(_n(waterlevel[:,1], mu_wl, s_wl), _n(velocity[:,1], mu_v, s_v))'
    x0 = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=dyn0, forcing=_forc(1)))
    push!(gt,   x0)
    push!(pred, x0)

    for ii in 1:nsteps
        println("Predicting time step: $ii")
        yhat_norm = model(pred[end])

        x_next = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=yhat_norm, forcing=_forc(ii+1)))
        push!(pred, x_next)

        dyn_gt = hcat(_n(waterlevel[:,ii+1], mu_wl, s_wl), _n(velocity[:,ii+1], mu_v, s_v))'
        x_gt = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=dyn_gt, forcing=_forc(ii+1)))
        push!(gt, x_gt)
    end

    function denorm_graph(g)
        dyn_phys = hcat(_d(g.ndata.dynamic[1,:], mu_wl, s_wl),
                        _d(g.ndata.dynamic[2,:], mu_v,  s_v))'
        forc_phys = _d(g.ndata.forcing, mu_t, s_t)
        return GNNGraph(g, ndata=(; static=g.ndata.static, dynamic=dyn_phys, forcing=forc_phys))
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

function load_run(model_dir::String)
    d              = JLD2.load(joinpath(model_dir, "model.jld2"))
    model          = d["model"]
    norm_strategy  = d["norm_strategy"]
    train_strategy = d["train_strategy"]

    train_settings = train_settings_from_toml(TOML.parsefile(joinpath(model_dir, "train_settings.toml")))
    model_settings = model_settings_from_toml(TOML.parsefile(joinpath(model_dir, "model_settings.toml")))

    return model, model_settings, train_settings, norm_strategy, train_strategy
end