"""Evaluate and render videos of the trained VAE student policy.

Deploys the VAE student in simulation using the learnable prior P(z|s)
for inference (no motion target at deployment time).

Usage:
    python -m latent_mj.eval.vae.eval_vae \
        --exp_name vae_run_1 \
        --use_renderer
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse

import numpy as np
import jax
import jax.numpy as jnp
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
    # restore into a matching param tree
    dummy_proprio = jnp.zeros((1, vae.proprio_dim))
    dummy_dif = jnp.zeros((1, vae.dif_dim))
    rng = jax.random.PRNGKey(0)
    dummy_params = vae.init(rng, dummy_proprio, dummy_dif, rng)
    params = checkpointer.restore(str(ckpt_path), dummy_params)
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True,
                        help="VAE experiment name")
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

    # Play env for visualization; videos land in storage/videos/vae/<exp_name>/
    play_env = PlayG1TrackingTennisEnv(
        with_racket=True,
        config=env_cfg,
        play_ref_motion=False,
        use_viewer=args.use_viewer,
        use_renderer=args.use_renderer,
        exp_name=args.exp_name,
        video_subdir="vae",
    )

    # Get dims from config
    train_env_class = lmj.registry.get(args.task, "tracking_train_env_class")
    train_env = train_env_class(config=env_cfg)
    train_env.prepare_trajectory(env_cfg.reference_traj_config.name)
    proprio_dim = train_env.observation_size["proprio_state"][0]
    dif_dim = train_env.observation_size["dif_state"][0]
    action_dim = train_env.action_size

    logging.info("proprio_dim=%d, dif_dim=%d, action_dim=%d", proprio_dim, dif_dim, action_dim)

    # Load VAE
    vae = VAEPolicy(
        proprio_dim=proprio_dim,
        dif_dim=dif_dim,
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

    # JIT inference function: sample z from the prior P(z|proprio), decode (no reference)
    @jax.jit
    def vae_inference(proprio, rng):
        action = vae.apply(vae_params, proprio[None], method=vae.inference,
                           rng=rng, deterministic=args.deterministic)[0]
        return action

    #  run evaluation
    rng = jax.random.PRNGKey(args.seed)
    state = play_env.reset()

    total_steps = play_env.th.traj.data.qpos.shape[0]
    logging.info("Running evaluation for %d steps...", total_steps)

    for _ in tqdm(range(total_steps - 1)):
        rng, step_rng = jax.random.split(rng)

        # Deployment: student acts from proprioception only, sampling z from the prior
        obs_proprio = jnp.array(state.obs["proprio_state"], dtype=jnp.float32)
        student_action = np.array(vae_inference(obs_proprio, step_rng))

        state = play_env.step(state, student_action)

    play_env.close()

    if args.use_renderer:
        video_dir = f"storage/videos/{play_env.video_subdir}/{play_env.exp_name}"
        logging.info("Videos saved to %s", video_dir)
        import glob
        videos = glob.glob(f"{video_dir}/**/*.mp4", recursive=True)
        for v in videos:
            logging.info("  %s", v)

    logging.info("Evaluation complete!")


if __name__ == "__main__":
    main()
