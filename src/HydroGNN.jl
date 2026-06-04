module HydroGNN

include("data.jl")
export load_data, load_data_multistep, collate_multistep_batch, DynamicNormStats, compute_norm_stats,
       NormStrategy, GlobalNorm, PerTrajectoryNorm, strategy_to_dict

include("gnn.jl")
export GNN, ModelSettings, save_model_settings, load_model_settings, model_settings_from_dict

include("training.jl")
export TrainSettings, train_model!,
       TrainStrategy, RolloutStrategy, SingleStepNoise, NoNoise,
       MultiStepRollout, PushforwardRollout, ScheduledRollout, ScheduledPushforward,
       train_settings_from_toml, save_train_strategy, load_train_strategy,
       build_strategy_from_config, strategy_to_dict

include("plot.jl")
export plot_loss, movie_graphs, movie_graphs_comp

include("eval.jl")
export predict_trajectory, evaluate_all_trajectories, rollout_metrics, load_run

include("hpar_search.jl")
export run_hparam_search, save_csv, load_csv, best
end