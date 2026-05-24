module HydroGNN

include("data.jl")
export load_data, load_data_multistep, DynamicNormStats, compute_norm_stats,
       NormStrategy, GlobalNorm, PerTrajectoryNorm, strategy_to_dict

include("gnn.jl")
export GNN, ModelSettings, save_model_settings, load_model_settings

include("training.jl")
export TrainSettings, train_model!,
       TrainStrategy, RolloutStrategy, SingleStepNoise, NoNoise,
       MultiStepRollout, PushforwardRollout, ScheduledRollout, ScheduledPushforward,
       train_settings_from_toml

include("plot.jl")
export plot_loss, movie_graphs, movie_graphs_comp

include("eval.jl")
export predict_trajectory, evaluate_all_trajectories, load_run
end