#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

module load 2024
module load CUDA/12.6.0

julia --threads auto hparam_search.jl