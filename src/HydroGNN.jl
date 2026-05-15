module HydroGNN

include("data.jl")
export load_data, load_data_multistep, DynamicNormStats, compute_norm_stats,
       NormStrategy, GlobalNorm, PerTrajectoryNorm, strategy_to_dict

include("gnn.jl")
export GNN

include("training.jl")
export ModelSettings, TrainSettings, train_model!,
       TrainStrategy, RolloutStrategy, SingleStepNoise, NoNoise,
       MultiStepRollout, PushforwardRollout, ScheduledRollout, ScheduledPushforward

include("plot.jl")
export plot_loss, movie_graphs, movie_graphs_comp

include("eval.jl")
export predict_trajectory, evaluate_all_trajectories, load_run
end