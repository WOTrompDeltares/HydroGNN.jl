module HydroGNN

include("data.jl")
export load_data, load_data_multistep, DynamicNormStats, compute_norm_stats,
       NormStrategy, GlobalNorm, PerTrajectoryNorm

include("gnn.jl")
export GNN

include("training.jl")
export TrainSettings, train_model!,
       TrainStrategy, SingleStepNoise, NoNoise, MultiStepRollout, PushforwardRollout

include("plot.jl")
export plot_loss, movie_graphs, movie_graphs_comp

include("eval.jl")
export predict_trajectory, evaluate_all_trajectories
end