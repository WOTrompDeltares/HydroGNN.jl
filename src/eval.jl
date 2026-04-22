using GraphNeuralNetworks

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