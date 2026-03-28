#!/bin/bash
#SBATCH --job-name=fl
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/out/slurm-%j.out
#SBATCH --error=slurm/err/slurm-%j.err

# Usage: sbatch commands/submit_a100.sh commands/chexpert/explore/runs/iid.sh

module load cuda/12.8
conda activate fl

source "$1"
