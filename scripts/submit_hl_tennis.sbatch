#!/bin/bash
#SBATCH --job-name=hl_tennis
#SBATCH --account=drl
#SBATCH --partition=drl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/data/scratch/cgoyal/LATENT-local/storage/logs/slurm/%x_%j.out
#SBATCH --error=/data/scratch/cgoyal/LATENT-local/storage/logs/slurm/%x_%j.err
#SBATCH --requeue

# ---------------------------------------------------------------------------
# Stage 3 high-level PPO training (pi_planner), long cluster run.
#
# Usage:
#   sbatch scripts/submit_hl_tennis.sbatch <exp_name> <vae_checkpoint_path> [extra train_ppo_hl_tennis.py args...]
#
# Example:
#   sbatch scripts/submit_hl_tennis.sbatch \
#       hl_run_full \
#       storage/logs/vae/vae_distill_kl0.01_long/checkpoints/00010000_final
#
# Resuming after preemption/timeout (SLURM --requeue will resubmit this same
# script automatically on preemption; for a manual resume after --time expires,
# just resubmit with the same exp_name -- restore_checkpoint_path picks up
# from wherever it left off, matched via --restore_exp_name below):
#   sbatch scripts/submit_hl_tennis.sbatch hl_run_full <vae_ckpt> --restore_exp_name hl_run_full
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT=/data/scratch/cgoyal/LATENT-local
cd "$REPO_ROOT"

# --- environment (fixes the jaredb-scratch leakage from earlier sessions) ---
export GLI_PATH="$REPO_ROOT"
export UV_CACHE_DIR=/data/scratch/cgoyal/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/scratch/cgoyal/.local/share/uv/python
export PATH=/data/scratch/cgoyal/uv:$PATH
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
mkdir -p "$REPO_ROOT/storage/logs/slurm"

# sanity check: make sure nothing is still pointing at jaredb's scratch space
if env | grep -qi jaredb; then
    echo "ERROR: found a jaredb-scoped env var still set -- fix before submitting:" >&2
    env | grep -i jaredb >&2
    exit 1
fi

echo "GLI_PATH=$GLI_PATH"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
which uv
uv --version

# --- args ---
EXP_NAME="${1:?Usage: sbatch submit_hl_tennis.sbatch <exp_name> <vae_checkpoint_path> [extra args...]}"
VAE_CKPT="${2:?Usage: sbatch submit_hl_tennis.sbatch <exp_name> <vae_checkpoint_path> [extra args...]}"
shift 2
EXTRA_ARGS=("$@")

# orbax requires an absolute checkpoint path
case "$VAE_CKPT" in
    /*) ;;  # already absolute
    *) VAE_CKPT="$REPO_ROOT/$VAE_CKPT" ;;
esac
if [ ! -e "$VAE_CKPT" ]; then
    echo "ERROR: vae_checkpoint_path does not exist: $VAE_CKPT" >&2
    exit 1
fi

# --- wandb ---
# Requires `wandb login` to have been run once on this account already.
# Set WANDB_MODE=offline below instead if the compute node has no internet
# access, then `wandb sync` the run directory afterward from a node that does.
export WANDB_PROJECT="${WANDB_PROJECT:-latent}"
# export WANDB_MODE=offline

echo "JAX / GPU visibility check:"
uv run python -c "import jax; print(jax.devices())"

echo "Starting Stage 3 HL training: exp_name=$EXP_NAME  vae_ckpt=$VAE_CKPT"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"

uv run python -m latent_mj.learning.train.train_ppo_hl_tennis \
    --exp_name "$EXP_NAME" \
    --vae_checkpoint_path "$VAE_CKPT" \
    "${EXTRA_ARGS[@]}"
