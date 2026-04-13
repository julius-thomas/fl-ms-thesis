#!/bin/bash
#SBATCH --job-name=fl
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hendrixfut01fl,hendrixgpu07fl,hendrixfut03fl,hendrixgpu02fl,hendrixgpu08fl,hendrixgpu15fl,hendrixgpu04fl,hendrixgpu14fl,hendrixgpu21fl,hendrixgpu01fl,hendrixgpu23fl,hendrixgpu24fl,hendrixgpu25fl,hendrixgpu26fl,hendrixgpu16fl
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/out/slurm-%j.out
#SBATCH --error=slurm/err/slurm-%j.err

# Usage: sbatch commands/submit_a100.sh commands/chexpert/explore/runs/iid.sh

module load cuda/12.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$1"
