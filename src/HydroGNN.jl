module HydroGNN

include("data.jl")
export load_data

include("gnn.jl")
export GNN

include("training.jl")
export TrainSettings, train_model!

include("plot.jl")
export plot_loss, movie_graphs, movie_graphs_comp

include("eval.jl")
export predict_trajectory
end