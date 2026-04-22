module HydroGNN

include("data.jl")
export load_data

include("plot.jl")
export plot_loss, movie_graphs, movie_graphs_comp

include("eval.jl")
export predict_trajectory
end