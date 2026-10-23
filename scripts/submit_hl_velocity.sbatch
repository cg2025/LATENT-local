#!/bin/bash
#SBATCH --job-name=hl_velocity
#SBATCH --account=drl
#SBATCH --partition=drl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/scratch/cgoyal/LATENT-local/storage/logs/slurm/%x_%j.out
#SBATCH --error=/data/scratch/cgoyal/LATENT-local/storage/logs/slurm/%x_%j.err
# Usage: sbatch scripts/submit_hl_velocity.sbatch <exp_name> <abs_vae_ckpt> [extra train args]
set -euo pipefail
REPO_ROOT=/data/scratch/cgoyal/LATENT-local
cd "$REPO_ROOT"
export GLI_PATH="$REPO_ROOT"
export UV_CACHE_DIR=/data/scratch/cgoyal/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/scratch/cgoyal/.local/share/uv/python
export PATH=/data/scratch/cgoyal/uv:$PATH
mkdir -p storage/logs/slurm
EXP_NAME="${1:?exp_name}"; VAE_CKPT="${2:?abs vae ckpt}"; shift 2
uv run python -m latent_mj.learning.train.train_ppo_hl_tennis \
    --task G1VelocityVAE --exp_name "$EXP_NAME" \
    --vae_checkpoint_path "$VAE_CKPT" --no_wandb "$@"
