"""Batched teacher-tracker eval (no VAE): fall rate + tracking error, split by
healthy vs. near-fall steps.

This is the "no VAE" baseline: it rolls out the raw PPO tracker (teacher)
policy directly against the reference motion, with no VAE bottleneck /
distillation in the loop. Metrics and print layout intentionally mirror
`latent_mj.eval.vae.eval_vae_batched2` so the numbers can be diffed directly
against a VAE-distilled student run.

Usage:
    python -m latent_mj.eval.tracking.eval_teacher_batched \
        --exp_name my_tracker_run \
        --num_envs 512 \
        --deterministic
"""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import functools
import json

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from tqdm import tqdm

import latent_mj as lmj
from latent_mj.constant import WANDB_PATH_LOG
from latent_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from latent_mj.learning.policy.ppo import train_tracking as ppo
from brax.training.agents.ppo.networks import make_ppo_networks


def load_teacher_inference_fn(env, trajectory_data, task_cfg, args):
    """Restore the pretrained PPO tracker (teacher) and return a policy fn.

    Mirrors `train_vae_distill.load_teacher`, so this eval uses exactly the
    same restore path / network config as the VAE distillation pipeline does
    when it queries the teacher for labels.
    """
    teacher_ckpt = lmj.get_latest_ckpt(f"track/{args.exp_name}")
    if teacher_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found for tracker exp: {args.exp_name}")
    logging.info("Loading teacher tracker checkpoint: %s", teacher_ckpt)

    config_path = WANDB_PATH_LOG / "track" / args.exp_name / "checkpoints" / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    policy_cfg = task_cfg.policy_config
    policy_cfg.update(config["policy_config"])

    network_factory = functools.partial(make_ppo_networks, **policy_cfg.network_factory)
    make_policy_fn, params, _ = ppo.train(
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
    inference_fn = make_policy_fn(params, deterministic=args.deterministic)
    return inference_fn, policy_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Exp name of the pretrained PPO tracker (teacher / no-VAE baseline)")
    parser.add_argument("--task", type=str, default="G1TrackingTennis")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--near_fall_window", type=int, default=10,
                        help="Steps before an env's termination to count as 'near-fall'")
    args = parser.parse_args()

    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    aux = env_cfg.reward_config.auxiliary

    EnvClass = lmj.registry.get(args.task, "tracking_train_env_class")
    env = EnvClass(config=env_cfg)
    trajectory_data = env.prepare_trajectory(env_cfg.reference_traj_config.name)

    logging.info("action_dim=%d", env.action_size)

    inference_fn, policy_cfg = load_teacher_inference_fn(env, trajectory_data, task_cfg, args)

    wrapped_env = wrap_fn(env, episode_length=policy_cfg.episode_length)
    reset_fn = jax.jit(wrapped_env.reset)
    step_fn = jax.jit(wrapped_env.step)

    rng = jax.random.PRNGKey(args.seed)
    rng, env_rng = jax.random.split(rng)
    state = reset_fn(jax.random.split(env_rng, args.num_envs), trajectory_data)

    episode_length = policy_cfg.episode_length
    fell_ever = np.zeros(args.num_envs, dtype=bool)
    fall_step = np.full(args.num_envs, -1, dtype=np.int32)

    # Store per-env, per-step error (episode_length x num_envs), not pre-averaged.
    root_h_err_steps = np.zeros((episode_length, args.num_envs), dtype=np.float32)
    upper_err_steps = np.zeros((episode_length, args.num_envs), dtype=np.float32)
    lower_err_steps = np.zeros((episode_length, args.num_envs), dtype=np.float32)

    for t in tqdm(range(episode_length)):
        rng, act_rng = jax.random.split(rng)
        # Teacher policy consumes the full ("state") observation -- no VAE
        # bottleneck / proprio-only restriction here.
        action, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, action, trajectory_data)

        truncation = np.array(state.info["truncation"])
        termination_only = np.clip(np.array(state.done) - truncation, 0.0, 1.0) > 0.5
        newly_fallen = termination_only & (~fell_ever)
        fall_step[newly_fallen] = t
        fell_ever = fell_ever | termination_only

        r_h = jnp.clip(state.metrics["reward/root_height_tracking"], 1e-6, 1.0)
        r_u = jnp.clip(state.metrics["reward/rigid_body_pos_tracking_upper"], 1e-6, 1.0)
        r_l = jnp.clip(state.metrics["reward/rigid_body_pos_tracking_lower"], 1e-6, 1.0)
        root_h_err_steps[t] = np.array(-aux.root_height_sigma * jnp.log(r_h))
        upper_err_steps[t] = np.array(-aux.upper_body_sigma * jnp.log(r_u))
        lower_err_steps[t] = np.array(-aux.lower_body_sigma * jnp.log(r_l))

    fall_rate = float(fell_ever.mean()) * 100

    # Build a (episode_length x num_envs) mask: True where step t is within
    # near_fall_window steps before that env's eventual fall.
    steps_idx = np.arange(episode_length)[:, None]  # (T, 1)
    fall_step_b = fall_step[None, :]                # (1, N)
    has_fall = fell_ever[None, :]
    dist_to_fall = fall_step_b - steps_idx           # (T, N), can be negative
    near_fall_mask = has_fall & (dist_to_fall >= 0) & (dist_to_fall <= args.near_fall_window)
    healthy_mask = ~near_fall_mask

    def masked_mean(arr, mask):
        vals = arr[mask]
        return float(vals.mean()) if vals.size > 0 else float("nan")

    print(f"\n=== Batched Teacher (no VAE) Eval: {args.exp_name} (deterministic={args.deterministic}) ===")
    print(f"Envs: {args.num_envs}  |  Episode length: {episode_length}")
    print(f"Fall rate: {fall_rate:.1f}%  ({int(fell_ever.sum())}/{args.num_envs} envs fell)")
    print(f"Near-fall window: last {args.near_fall_window} steps before each env's termination\n")

    print("--- Overall (all steps, all envs) ---")
    print(f"Mean root height error: {root_h_err_steps.mean():.4f} m")
    print(f"Mean upper body pos error (summed L1): {upper_err_steps.mean():.4f}")
    print(f"Mean lower body pos error (summed L1): {lower_err_steps.mean():.4f}")

    print("\n--- Healthy steps only (excludes near-fall window) ---")
    print(f"Mean root height error: {masked_mean(root_h_err_steps, healthy_mask):.4f} m")
    print(f"Mean upper body pos error (summed L1): {masked_mean(upper_err_steps, healthy_mask):.4f}")
    print(f"Mean lower body pos error (summed L1): {masked_mean(lower_err_steps, healthy_mask):.4f}")

    print("\n--- Near-fall steps only ---")
    print(f"Mean root height error: {masked_mean(root_h_err_steps, near_fall_mask):.4f} m")
    print(f"Mean upper body pos error (summed L1): {masked_mean(upper_err_steps, near_fall_mask):.4f}")
    print(f"Mean lower body pos error (summed L1): {masked_mean(lower_err_steps, near_fall_mask):.4f}")

    print(f"\n--- Error over episode progression (healthy steps only, {args.num_bins} bins) ---")
    bin_edges = np.linspace(0, episode_length, args.num_bins + 1).astype(int)
    for i in range(args.num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_mask = np.zeros_like(healthy_mask)
        bin_mask[lo:hi, :] = True
        bin_mask = bin_mask & healthy_mask
        u_err = masked_mean(upper_err_steps, bin_mask)
        l_err = masked_mean(lower_err_steps, bin_mask)
        r_err = masked_mean(root_h_err_steps, bin_mask)
        print(f"step [{lo:4d}-{hi:4d}): upper={u_err:.4f}  lower={l_err:.4f}  root_h={r_err:.4f}")

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
