#!/bin/bash
#SBATCH --job-name=download-bones-g1
#SBATCH --partition=drl-titanrtx
#SBATCH --qos=drl-main
#SBATCH --account=drl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

source /data/scratch/cgoyal/env.sh
source /data/scratch/cgoyal/LATENT-local/.env

cd /data/scratch/cgoyal/LATENT-local

JOB_TMP="/tmp/$USER/$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$JOB_TMP/venv"
export UV_LINK_MODE=copy
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

uv sync --frozen

# hf download resumes automatically from the .incomplete file already on disk --
# no need to delete anything first. Using the project's own uv-managed
# huggingface_hub (via Python API) instead of the `hf` CLI, since the CLI
# binary lives under an AFS home directory that compute nodes can't reach.
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='bones-studio/seed',
    repo_type='dataset',
    allow_patterns=['g1.tar.gz', 'metadata/*'],
    local_dir='/data/scratch/cgoyal/bones-seed-download',
)
"

echo "Download finished. Verifying file size..."
ls -la /data/scratch/cgoyal/bones-seed-download/g1.tar.gz
