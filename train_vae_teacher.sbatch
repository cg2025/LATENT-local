#!/bin/bash
#SBATCH --job-name=vae-teacher
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
LAMBDA_KL=${1:?usage: sbatch train_vae_teacher.sbatch <lambda_kl> <task> <tag>}
TASK=${2:?task}
TAG=${3:?tag}

source /data/scratch/cgoyal/env.sh
source /data/scratch/cgoyal/LATENT-local/.env
cd /data/scratch/cgoyal/LATENT-local

JOB_TMP="/tmp/$USER/$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$JOB_TMP/venv"
export UV_LINK_MODE=copy
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

uv sync --frozen

uv run python -m latent_mj.learning.train.train_vae_distill \
    --teacher_exp_name tracker_bones_v1 \
    --task $TASK \
    --exp_name vae_distill_${TAG} \
    --lambda_kl $LAMBDA_KL \
    --num_iterations 20000 \
    --no_wandb
