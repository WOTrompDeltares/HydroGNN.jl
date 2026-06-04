cd(@__DIR__)
using Pkg
Pkg.activate(".")

using HydroGNN

length(ARGS) == 1 || error("Usage: julia run_experiment.jl <config.toml>")
run_hparam_search(ARGS[1])
