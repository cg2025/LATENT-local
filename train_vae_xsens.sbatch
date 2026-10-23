#!/bin/bash
#SBATCH --job-name=vae-xsens
#SBATCH --partition=drl-titanrtx
#SBATCH --qos=drl-main
#SBATCH --account=drl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source /data/scratch/cgoyal/env.sh
source /data/scratch/cgoyal/LATENT-local/.env
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /data/scratch/cgoyal/LATENT-local

JOB_TMP="/tmp/$USER/$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$JOB_TMP/venv"
export UV_LINK_MODE=copy
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

uv sync --frozen

uv run python -m latent_mj.learning.train.train_vae_distill \
    --task G1TrackingTennisXsens \
    --teacher_exp_name tracker_xsens_s16_v1 \
    --exp_name vae_xsens_s16_v1 \
    --lambda_kl 0.01 \
    --num_iterations 2000 \
    --no_wandb

# Eval: student fall rate + tracking error, then teacher baseline on the same task
uv run python -m latent_mj.eval.vae.eval_vae_batched2 \
    --task G1TrackingTennisXsens \
    --exp_name vae_xsens_s16_v1 \
    --num_envs 512 --deterministic

uv run python -m latent_mj.eval.tracking.eval_teacher_batched \
    --task G1TrackingTennisXsens \
    --exp_name tracker_xsens_s16_v1 \
    --num_envs 512 --deterministic
