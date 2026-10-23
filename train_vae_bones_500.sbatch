#!/bin/bash
#SBATCH --job-name=500-vae-bone
#SBATCH --partition=drl-titanrtx
#SBATCH --qos=drl-main
#SBATCH --account=drl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodelist=gpu20-3.drl,gpu20-4.drl

set -euo pipefail

source /data/scratch/cgoyal/env.sh
source /data/scratch/cgoyal/LATENT-local/.env
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export BONES_SET=pool1000
export BONES_NUM_CLIPS=500

cd /data/scratch/cgoyal/LATENT-local

JOB_TMP="/tmp/$USER/$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$JOB_TMP/venv"
export UV_LINK_MODE=copy
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

uv sync --frozen

uv run python -m latent_mj.learning.train.train_vae_distill \
    --task G1TrackingTennisBones \
    --teacher_exp_name tracker_bones_pool1000_n500_500M_s0 \
    --exp_name vae_bones_500 \
    --lambda_kl 0.01 \
    --num_iterations 25000 \
    --no_wandb


