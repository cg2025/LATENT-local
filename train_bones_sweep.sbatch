#!/bin/bash
#SBATCH --job-name=bones-sweep
#SBATCH --partition=drl-titanrtx
#SBATCH --qos=drl-main
#SBATCH --account=drl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source /data/scratch/cgoyal/env.sh
source /data/scratch/cgoyal/LATENT-local/.env
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export BONES_SET=${BONES_SET:-pool}
: "${BONES_NUM_CLIPS:?pass --export=ALL,BONES_NUM_CLIPS=<n>}"
SEED=${SEED:-0}
cd /data/scratch/cgoyal/LATENT-local

JOB_TMP="/tmp/$USER/$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$JOB_TMP/venv"
export UV_LINK_MODE=copy
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

uv sync --frozen

echo "BONES_SET=$BONES_SET BONES_NUM_CLIPS=$BONES_NUM_CLIPS SEED=$SEED"
uv run python -m latent_mj.learning.train.train_ppo_track_tennis \
    --task G1TrackingTennisBones \
    --exp_name tracker_bones_${BONES_SET}_n${BONES_NUM_CLIPS}_500M_s${SEED} \
    --num_timesteps 500000000 \
    --num_envs 16384 \
    --batch_size 2048 \
    --num_evals 10 \
    --seed $SEED \
    --no_wandb
