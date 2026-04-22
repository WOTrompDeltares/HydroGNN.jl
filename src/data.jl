using GraphNeuralNetworks
using Statistics
using JLD2

using Flux: onehotbatch

function read_trajectory(fn, traj_num::Int)
    jldopen(fn, "r") do f
        traj_group = f["trajectory_$(traj_num)"]

        velocity = traj_group["velocity_node"]
        waterlevel = traj_group["waterlevel"]
        mesh_pos = traj_group["mesh_pos"][1,:]
        node_type = traj_group["node_type"]
        bathymetry = traj_group["bathymetry"]

        edges = traj_group["edges"]
        edges = cat(edges, reverse(edges, dims=1), dims=2)

        return velocity, waterlevel, mesh_pos, node_type, bathymetry, edges
    end
end


function load_data(fn)
    data_x = []
    data_y = []


    jldopen(fn, "r") do f
        for key in sort((keys(f)))
            println("Loading trajectory: $key")

            traj_group = f[key]

            velocity = traj_group["velocity_node"]
            waterlevel = traj_group["waterlevel"]
            mesh_pos = traj_group["mesh_pos"][1,:]
            node_type = traj_group["node_type"]
            bathymetry = traj_group["bathymetry"]

            edges = traj_group["edges"]
            edges = cat(edges, reverse(edges, dims=1), dims=2)

            bathymetry .= (bathymetry .- mean(bathymetry)) ./ std(bathymetry)
            mesh_pos .= (mesh_pos .- minimum(mesh_pos)) ./ (maximum(mesh_pos) - minimum(mesh_pos))
            node_onehot = onehotbatch(node_type, collect(0:5))

            for ii in 1:(size(velocity,2)-1)

                # x = hcat(waterlevel[:,ii], velocity[:,ii], bathymetry, mesh_pos, node_onehot')
                data_static = hcat(bathymetry, mesh_pos, node_onehot')
                data_dym= hcat(waterlevel[:,ii], velocity[:,ii])
                y = hcat(waterlevel[:,ii+1], velocity[:,ii+1])

                push!(data_x, GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=(; static=data_static', dynamic=data_dym')))
                push!(data_y, GNNGraph(Int64.(edges[1,:]), Int64.(edges[2,:]), ndata=y'))

            end

        end

    end
    return data_x, data_y
end