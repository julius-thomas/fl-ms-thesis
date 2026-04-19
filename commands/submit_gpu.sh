#!/bin/bash
#SBATCH --job-name=fl
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=hendrixgpu03fl
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/out/slurm-%x-%A_%a.out
#SBATCH --error=slurm/err/slurm-%x-%A_%a.err
#SBATCH --mail-user=nsw510@alumni.ku.dk
#SBATCH --mail-type=FAIL

# Usage:
#   sbatch commands/submit_gpu.sh commands/mimic/vanilla.sh                  # single run, seed=42
#   sbatch --array=0-4 commands/submit_gpu.sh commands/mimic/vanilla.sh      # 5 reps, seeds 42..46

module load cuda/12.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
export SEED=$((42 + TASK_ID))
export EXP_SUFFIX="_seed${SEED}"
echo "[submit] run=$1 array_task_id=${TASK_ID} SEED=${SEED} EXP_SUFFIX=${EXP_SUFFIX}"

source "$1"
