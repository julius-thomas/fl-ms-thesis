#!/bin/bash
#SBATCH --job-name=fl
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/out/slurm-%j.out
#SBATCH --error=slurm/err/slurm-%j.err
#SBATCH --mail-user=nsw510@alumni.ku.dk
#SBATCH --mail-type=FAIL

# Usage: sbatch commands/submit_l40s.sh commands/chexpert/explore/runs/iid.sh

module load cuda/12.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$1"
