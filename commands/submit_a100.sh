#!/bin/bash
#SBATCH --job-name=fl
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Usage: sbatch commands/submit_a100.sh commands/chexpert/explore/runs/iid.sh

# Conda setup (change 'fl' to your env name)
eval "$(conda shell.bash hook)"
conda activate fl

REPO_ROOT="$(git rev-parse --show-toplevel)"

if [ -z "$1" ]; then
    echo "Usage: sbatch commands/submit_a100.sh <run_script.sh>"
    exit 1
fi

cd "$REPO_ROOT"
source "$1"
