#!/bin/bash
# PPO tracker on the Xsens S16 tennis retarget (task G1TrackingTennisXsens, dataset Tennis/xsens_s16).
# Data: scripts/process_motion/xsens_npz_to_latent.py -> storage/data/mocap/Tennis/xsens_s16/.
# Partition: aia-h200 (1 H200, 4-day cap). The owner's 3e9-step runs took ~51 h on a Titan RTX
# (~16.5k sps); an H200 is several times faster, and aia-h200-shortjobs (8 h cap) is too short.
# Resume: sbatch scripts/submit_track_xsens.sbatch --restore_exp_name track/tracker_xsens_s16_v1
# (extra args are passed to the train script; loads the latest checkpoint of that run).
#SBATCH --job-name=tracker-xsens
#SBATCH --partition=aia-h200
#SBATCH --account=aia
#SBATCH --qos=aia-main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/scratch/jaredb/sbatch/logs/%x_%j.out
#SBATCH --error=/data/scratch/jaredb/sbatch/logs/%x_%j.err

set -euo pipefail

REPO=/data/scratch/jaredb/LATENT-local
source /data/scratch/jaredb/env.sh
set -a; source "$REPO/.env"; set +a

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$REPO"

echo "host:     $(hostname)"
echo "git:      $(git rev-parse --short HEAD)"
echo "GLI_PATH: $GLI_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --no-sync python -m latent_mj.learning.train.train_ppo_track_tennis \
    --task G1TrackingTennisXsens \
    --exp_name tracker_xsens_s16_v1 \
    --num_timesteps 3000000000 \
    --num_envs 16384 \
    --batch_size 2048 \
    --num_evals 50 \
    "$@"
