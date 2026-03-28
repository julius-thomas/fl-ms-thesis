#!/bin/bash
#SBATCH --job-name=fl
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=log/slurm/out/slurm-%j.out
#SBATCH --error=log/slurm/err/slurm-%j.err

# Usage: sbatch commands/submit_a100.sh commands/chexpert/explore/runs/iid.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fl

source "$1"
