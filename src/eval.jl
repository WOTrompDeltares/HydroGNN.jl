using GraphNeuralNetworks
using JLD2

function predict_trajectory(fn, traj_ind::Int, model)

    velocity, waterlevel, mesh_pos, node_type, bathymetry, edges = read_trajectory(fn, traj_ind)
    nsteps = size(velocity,2) - 1


    bathymetry .= (bathymetry .- mean(bathymetry)) ./ std(bathymetry)
    mesh_pos .= (mesh_pos .- minimum(mesh_pos)) ./ (maximum(mesh_pos) - minimum(mesh_pos))

    data_static = hcat(bathymetry, mesh_pos, onehotbatch(node_type, collect(0:5))')

    gt = []
    pred = []

    x0 = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=hcat(waterlevel[:,1], velocity[:,1])'))
    push!(gt, x0)
    push!(pred, x0)

    for ii in 1:nsteps
        println("Predicting time step: $ii")
        yhat = model(pred[end], pred[end].ndata.dynamic, pred[end].ndata.static)

        x_next = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=yhat))

        push!(pred, x_next)

        x_gt = GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=hcat(waterlevel[:,ii+1], velocity[:,ii+1])'))
        push!(gt, x_gt)
    end

    return gt, pred

end

function evaluate_all_trajectories(data_file::String, model, output_dir::String)
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
            gt, pred = predict_trajectory(data_file, traj_num, model)

            # Generate and save comparison movie
            output_file = joinpath(output_dir, "trajectory_$(traj_num)_comparison.mp4")
            println("Saving movie to: $output_file")
            movie_graphs_comp(gt, pred, output_file)
        end
    end

    println("\nEvaluation complete. All movies saved to: $output_dir")
end