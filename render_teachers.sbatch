#!/bin/bash
#SBATCH --job-name=render-teachers
#SBATCH --partition=drl-titanrtx
#SBATCH --qos=drl-main
#SBATCH --account=drl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
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
uv run python -c "import jax; d = jax.devices(); print(d); assert d[0].platform != 'cpu', d" || exit 1
uv run python latent_mj/app/mj_playground_init.py

for EXP in "$@"; do
  echo "=== $EXP ==="
  uv run python -m latent_mj.app.brax2onnx_tracking --task G1TrackingTennis --exp_name "$EXP" \
    && uv run python -m latent_mj.eval.tracking.mj_onnx_video --task G1TrackingTennis --exp_name "$EXP" --use_renderer \
    || echo "FAILED: $EXP"
done
