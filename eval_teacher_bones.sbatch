#!/bin/bash
#SBATCH --job-name=eval-bones
#SBATCH --partition=drl-titanrtx
#SBATCH --qos=drl-main
#SBATCH --account=drl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source /data/scratch/cgoyal/env.sh
source /data/scratch/cgoyal/LATENT-local/.env

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /data/scratch/cgoyal/LATENT-local

echo "GLI_PATH=${GLI_PATH:?GLI_PATH is not set}"
echo "Starting batched teacher evaluation"

uv run --frozen python -m latent_mj.eval.tracking.eval_teacher_batched \
    --exp_name tracker_bones_v1 \
    --task G1TrackingTennisBones \
    --num_envs 512 \
    --deterministic
