"""Evaluate and render videos of the trained VAE student policy.

Deploys the VAE student in simulation using the learnable prior P(z|s)
for inference (no motion target needed at deployment time).

Usage:
    python -m latent_mj.eval.vae.eval_vae \
        --exp_name vae_run_1 \
        --use_renderer

    # Compare student vs teacher side by side
    python -m latent_mj.eval.vae.eval_vae \
        --exp_name vae_run_1 \
        --teacher_exp_name short_run_1 \
        --use_renderer
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import time
import json
import functools
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import imageio.v2 as imageio
from tqdm import tqdm
from absl import logging

import latent_mj as lmj
from latent_mj.constant import WANDB_PATH_LOG
from latent_mj.envs.g1_tracking.play.play_g1_env_tracking_tennis import PlayG1TrackingTennisEnv
from latent_mj.learning.policy.vae.networks import VAEPolicy
import orbax.checkpoint as ocp


def get_latest_ckpt(vae_exp_name: str):
    ckpt_dir = WANDB_PATH_LOG / "vae" / vae_exp_name / "checkpoints"
    ckpts = [c for c in ckpt_dir.glob("*") if not c.name.endswith(".json")]
    ckpts.sort(key=lambda x: x.name)
    return ckpts[-1] if ckpts else None


def load_vae_params(ckpt_path, vae):
    """Load VAE params from orbax checkpoint."""
    checkpointer = ocp.StandardCheckpointer()
    # Create dummy params to restore into
    dummy_state = jnp.zeros((1, vae.state_dim))
    dummy_priv = jnp.zeros((1, vae.privileged_dim))
    rng = jax.random.PRNGKey(0)
    dummy_params = vae.init(rng, dummy_state, dummy_priv, rng)
    params = checkpointer.restore(str(ckpt_path), dummy_params)
    return params


def load_teacher(teacher_exp_name: str, env, trajectory_data):
    """Load pretrained PPO teacher for comparison."""
    from latent_mj.envs.g1_tracking.utils.wrapper import wrap_fn
    from latent_mj.learning.policy.ppo import train_tracking as ppo
    from brax.training.agents.ppo.networks import make_ppo_networks

    task_cfg = lmj.registry.get("G1TrackingTennis", "tracking_config")
    policy_cfg = task_cfg.policy_config

    teacher_ckpt = lmj.get_latest_ckpt(f"track/{teacher_exp_name}")
    config_path = WANDB_PATH_LOG / "track" / teacher_exp_name / "checkpoints" / "config.json"
    with open(config_path) as f:
        teacher_config = json.load(f)
    policy_cfg.update(teacher_config["policy_config"])

    network_factory = functools.partial(make_ppo_networks, **policy_cfg.network_factory)
    make_teacher_fn, teacher_params, _ = ppo.train(
        environment=env,
        trajectory_data=trajectory_data,
        num_timesteps=0,
        num_envs=1,
        episode_length=policy_cfg.episode_length,
        wrap_env=True,
        wrap_env_fn=wrap_fn,
        network_factory=network_factory,
        restore_checkpoint_path=str(teacher_ckpt),
        normalize_observations=policy_cfg.normalize_observations,
    )
    teacher_fn = make_teacher_fn(teacher_params, deterministic=True)
    return teacher_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True,
                        help="VAE experiment name")
    parser.add_argument("--teacher_exp_name", type=str, default=None,
                        help="Optional: PPO teacher exp name for side-by-side comparison")
    parser.add_argument("--task", type=str, default="G1TrackingTennis")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--use_renderer", action="store_true",
                        help="Save video to file")
    parser.add_argument("--use_viewer", action="store_true",
                        help="Open interactive MuJoCo viewer")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use prior mean (no sampling) for inference")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # ── Load env ─────────────────────────────────────────────────────────────
    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config

    # Play env for visualization (CPU, no JAX)
    play_env = PlayG1TrackingTennisEnv(
        with_racket=True,
        config=env_cfg,
        play_ref_motion=False,
        use_viewer=args.use_viewer,
        use_renderer=args.use_renderer,
        exp_name=f"vae_{args.exp_name}",
    )

    # Get dims from config directly
    train_env_class = lmj.registry.get(args.task, "tracking_train_env_class")
    train_env = train_env_class(config=env_cfg)
    train_env.prepare_trajectory(env_cfg.reference_traj_config.name)
    state_dim = train_env.observation_size["state"][0]
    privileged_dim = train_env.observation_size["privileged_state"][0]
    action_dim = train_env.action_size

    logging.info("state_dim=%d, privileged_dim=%d, action_dim=%d", state_dim, privileged_dim, action_dim)

    # ── Load VAE ──────────────────────────────────────────────────────────────
    vae = VAEPolicy(
        state_dim=state_dim,
        privileged_dim=privileged_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        encoder_hidden=(512, 256),
        decoder_hidden=(512, 256, 128),
        prior_hidden=(256, 128),
    )

    ckpt_path = get_latest_ckpt(args.exp_name)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for VAE exp: {args.exp_name}")
    logging.info("Loading VAE checkpoint: %s", ckpt_path)
    vae_params = load_vae_params(ckpt_path, vae)
    logging.info("VAE loaded successfully.")

    # JIT inference function (sample from prior)
    @jax.jit
    def vae_inference(state, rng):
        action = vae.apply(vae_params, state[None], method=vae.inference,
                           rng=rng, deterministic=args.deterministic)[0]
        return action

    # ── Optionally load teacher ───────────────────────────────────────────────
    teacher_fn = None
    if args.teacher_exp_name:
        logging.info("Loading teacher for comparison: %s", args.teacher_exp_name)
        teacher_fn = load_teacher(args.teacher_exp_name, train_env,
                                  train_env.th.traj.data)
        logging.info("Teacher loaded.")

    # ── Run evaluation ────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(args.seed)
    state = play_env.reset()

    total_steps = play_env.th.traj.data.qpos.shape[0]
    logging.info("Running evaluation for %d steps...", total_steps)

    rewards = []
    for i in tqdm(range(total_steps - 1)):
        rng, step_rng = jax.random.split(rng)

        # Get VAE student action from prior
        obs_state = jnp.array(state.obs["state"], dtype=jnp.float32)
        student_action = np.array(vae_inference(obs_state, step_rng))

        state = play_env.step(state, student_action)

    play_env.close()

    if args.use_renderer:
        video_dir = f"storage/videos/vae/{args.exp_name}"
        logging.info("Videos saved to %s", video_dir)
        # Find and report saved videos
        import glob
        videos = glob.glob(f"{video_dir}/**/*.mp4", recursive=True)
        for v in videos:
            logging.info("  %s", v)

    logging.info("Evaluation complete!")


if __name__ == "__main__":
    main()
