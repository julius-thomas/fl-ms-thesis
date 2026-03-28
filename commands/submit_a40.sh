#!/bin/bash
#SBATCH --job-name=fl
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/out/slurm-%j.out
#SBATCH --error=slurm/err/slurm-%j.err

# Usage: sbatch commands/submit_a100.sh commands/chexpert/explore/runs/iid.sh

module load pytorch/2.2.2
source ~/envs/fl/bin/activate

source "$1"
