"""Batched VAE eval: fall rate + raw tracking error + fall-timing histogram.

Usage:
    python -m latent_mj.eval.vae.eval_vae_batched \
        --exp_name vae_distill_kl0.01_long \
        --num_envs 512 \
        --deterministic
"""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from tqdm import tqdm

import latent_mj as lmj
from latent_mj.constant import WANDB_PATH_LOG
from latent_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from latent_mj.learning.policy.vae.networks import VAEPolicy
import orbax.checkpoint as ocp


def get_latest_ckpt(vae_exp_name):
    ckpt_dir = WANDB_PATH_LOG / "vae" / vae_exp_name / "checkpoints"
    ckpts = [c for c in ckpt_dir.glob("*") if not c.name.endswith(".json")]
    ckpts.sort(key=lambda x: x.name)
    return ckpts[-1] if ckpts else None


def load_vae_params(ckpt_path, vae):
    checkpointer = ocp.StandardCheckpointer()
    dummy_proprio = jnp.zeros((1, vae.proprio_dim))
    dummy_dif = jnp.zeros((1, vae.dif_dim))
    rng = jax.random.PRNGKey(0)
    dummy_params = vae.init(rng, dummy_proprio, dummy_dif, rng)
    return checkpointer.restore(str(ckpt_path), dummy_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--task", type=str, default="G1TrackingTennis")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_bins", type=int, default=10,
                        help="Number of histogram bins across the episode length")
    args = parser.parse_args()

    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config
    aux = env_cfg.reward_config.auxiliary

    EnvClass = lmj.registry.get(args.task, "tracking_train_env_class")
    env = EnvClass(config=env_cfg)
    trajectory_data = env.prepare_trajectory(env_cfg.reference_traj_config.name)

    proprio_dim = env.observation_size["proprio_state"][0]
    dif_dim = env.observation_size["dif_state"][0]
    action_dim = env.action_size
    logging.info("proprio_dim=%d dif_dim=%d action_dim=%d", proprio_dim, dif_dim, action_dim)

    vae = VAEPolicy(
        proprio_dim=proprio_dim, dif_dim=dif_dim, action_dim=action_dim,
        latent_dim=args.latent_dim,
        encoder_hidden=(512, 256), decoder_hidden=(512, 256, 128), prior_hidden=(256, 128),
    )
    ckpt_path = get_latest_ckpt(args.exp_name)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for VAE exp: {args.exp_name}")
    logging.info("Loading VAE checkpoint: %s", ckpt_path)
    vae_params = load_vae_params(ckpt_path, vae)

    @jax.jit
    def get_student_action(proprio, rng):
        return vae.apply(vae_params, proprio, method=vae.inference,
                          rng=rng, deterministic=args.deterministic)

    wrapped_env = wrap_fn(env, episode_length=policy_cfg.episode_length)
    reset_fn = jax.jit(wrapped_env.reset)
    step_fn = jax.jit(wrapped_env.step)

    rng = jax.random.PRNGKey(args.seed)
    rng, env_rng = jax.random.split(rng)
    state = reset_fn(jax.random.split(env_rng, args.num_envs), trajectory_data)

    fell_ever = np.zeros(args.num_envs, dtype=bool)
    fall_step = np.full(args.num_envs, -1, dtype=np.int32)  # -1 = never fell
    root_height_err, upper_pos_err, lower_pos_err = [], [], []

    episode_length = policy_cfg.episode_length
    for t in tqdm(range(episode_length)):
        rng, step_rng, act_rng = jax.random.split(rng, 3)
        proprio = state.obs["proprio_state"]
        action = get_student_action(proprio, act_rng)
        state = step_fn(state, action, trajectory_data)

        truncation = np.array(state.info["truncation"])
        termination_only = np.clip(np.array(state.done) - truncation, 0.0, 1.0) > 0.5

        # Record first fall step per env, only for envs that haven't fallen yet
        newly_fallen = termination_only & (~fell_ever)
        fall_step[newly_fallen] = t
        fell_ever = fell_ever | termination_only

        r_h = jnp.clip(state.metrics["reward/root_height_tracking"], 1e-6, 1.0)
        r_u = jnp.clip(state.metrics["reward/rigid_body_pos_tracking_upper"], 1e-6, 1.0)
        r_l = jnp.clip(state.metrics["reward/rigid_body_pos_tracking_lower"], 1e-6, 1.0)
        root_height_err.append(-aux.root_height_sigma * jnp.log(r_h))
        upper_pos_err.append(-aux.upper_body_sigma * jnp.log(r_u))
        lower_pos_err.append(-aux.lower_body_sigma * jnp.log(r_l))

    fall_rate = float(fell_ever.mean()) * 100
    mean_root_h_err = float(jnp.stack(root_height_err).mean())
    mean_upper_err = float(jnp.stack(upper_pos_err).mean())
    mean_lower_err = float(jnp.stack(lower_pos_err).mean())

    print(f"\n=== Batched VAE Eval: {args.exp_name} (deterministic={args.deterministic}) ===")
    print(f"Envs: {args.num_envs}  |  Episode length: {episode_length}")
    print(f"Fall rate: {fall_rate:.1f}%  ({int(fell_ever.sum())}/{args.num_envs} envs fell)")
    print(f"Mean root height error: {mean_root_h_err:.4f} m")
    print(f"Mean upper body pos error (summed L1): {mean_upper_err:.4f}")
    print(f"Mean lower body pos error (summed L1): {mean_lower_err:.4f}")

    # Fall-timing histogram: which part of the episode do falls cluster in?
    fallen_steps = fall_step[fell_ever]
    if len(fallen_steps) > 0:
        bin_edges = np.linspace(0, episode_length, args.num_bins + 1)
        counts, _ = np.histogram(fallen_steps, bins=bin_edges)
        print(f"\n--- Fall-timing histogram ({len(fallen_steps)} falls across {args.num_bins} bins) ---")
        for i in range(args.num_bins):
            lo, hi = int(bin_edges[i]), int(bin_edges[i + 1])
            bar = "#" * counts[i]
            print(f"step [{lo:4d}-{hi:4d}): {counts[i]:3d} {bar}")
    else:
        print("\nNo falls recorded.")


if __name__ == "__main__":
    main()
