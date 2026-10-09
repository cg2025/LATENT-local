"""Compare actions predicted by (A) the teacher tracker, (B) deterministic VAE
decoding, and (C) stochastic VAE decoding -- all queried at the SAME states.

Design: the environment is stepped forward using the TEACHER's action only.
This keeps every visited state on the teacher's trajectory manifold, so all
three action predictions are conditioned on identical states. If we instead
stepped with the VAE's action, the rollout would diverge from the teacher's
trajectory after a few steps and states downstream would no longer be
comparable across the three policies -- this is the same "teacher forcing"
principle `train_vae_distill.py` uses to collect DAgger labels.

At each step we record:
  A. teacher_action      = deterministic PPO tracker action, given full state
  B. vae_det_action      = VAE decode with z = prior mean (deterministic)
  C. vae_stoch_action    = VAE decode with z ~ N(prior mean, prior std) (sampled)

...and report, both overall and per action-dimension:
  - mean L2 distance and MAE between each pair (A vs B, A vs C, B vs C)
  - cosine similarity between each pair
  - per-dimension correlation (how consistently each joint's action tracks)
  - the stochastic policy's own self-consistency (spread across independent
    samples at the same state), to separate "VAE decode is biased away from
    teacher" from "VAE decode is just noisy due to sampling"

Usage:
    python -m latent_mj.eval.vae.compare_actions_teacher_vs_vae \
        --teacher_exp_name tracker_3B_v2 \
        --vae_exp_name vae_distill_1 \
        --num_envs 256 \
        --num_stoch_samples 8 \
        --out_path storage/logs/compare_actions_tracker_3B_v2_vs_vae_distill_1.npz
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
from latent_mj.learning.policy.vae.networks import VAEPolicy
from brax.training.agents.ppo.networks import make_ppo_networks
import orbax.checkpoint as ocp


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_teacher_inference_fn(env, trajectory_data, task_cfg, teacher_exp_name):
    """Restore the pretrained PPO tracker (teacher). Same pattern as
    `train_vae_distill.load_teacher` / `eval_teacher_batched.py`."""
    teacher_ckpt = lmj.get_latest_ckpt(f"track/{teacher_exp_name}")
    if teacher_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found for teacher exp: {teacher_exp_name}")
    logging.info("Loading teacher tracker checkpoint: %s", teacher_ckpt)

    config_path = WANDB_PATH_LOG / "track" / teacher_exp_name / "checkpoints" / "config.json"
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
    inference_fn = make_policy_fn(params, deterministic=True)
    return inference_fn, policy_cfg


def get_vae_ckpt(vae_exp_name):
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


# ─────────────────────────────────────────────────────────────────────────────
# Comparison metrics
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_stats(a, b, name_a, name_b):
    """a, b: (T, N, action_dim) arrays. Returns dict of scalar + per-dim stats."""
    diff = a - b
    l2 = np.linalg.norm(diff, axis=-1)          # (T, N)
    mae = np.mean(np.abs(diff), axis=-1)         # (T, N)

    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    denom = (np.linalg.norm(a_flat, axis=-1) * np.linalg.norm(b_flat, axis=-1))
    denom = np.clip(denom, 1e-8, None)
    cos_sim = np.sum(a_flat * b_flat, axis=-1) / denom

    per_dim_mae = np.mean(np.abs(a_flat - b_flat), axis=0)  # (action_dim,)
    per_dim_corr = np.array([
        np.corrcoef(a_flat[:, d], b_flat[:, d])[0, 1] if np.std(a_flat[:, d]) > 1e-8 and np.std(b_flat[:, d]) > 1e-8 else np.nan
        for d in range(a_flat.shape[-1])
    ])

    return {
        "pair": f"{name_a} vs {name_b}",
        "mean_l2": float(l2.mean()),
        "mean_mae": float(mae.mean()),
        "mean_cosine_sim": float(np.nanmean(cos_sim)),
        "per_dim_mae": per_dim_mae,
        "per_dim_corr": per_dim_corr,
    }


def print_pair_report(stats):
    print(f"\n--- {stats['pair']} ---")
    print(f"Mean L2 distance:   {stats['mean_l2']:.4f}")
    print(f"Mean MAE:           {stats['mean_mae']:.4f}")
    print(f"Mean cosine sim:    {stats['mean_cosine_sim']:.4f}")
    worst = np.argsort(-stats["per_dim_mae"])[:5]
    print("Top-5 most divergent action dims (by MAE):")
    for d in worst:
        corr = stats["per_dim_corr"][d]
        corr_str = f"{corr:.3f}" if not np.isnan(corr) else "n/a"
        print(f"  dim {d:2d}: MAE={stats['per_dim_mae'][d]:.4f}  corr={corr_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_exp_name", type=str, required=True)
    parser.add_argument("--vae_exp_name", type=str, required=True)
    parser.add_argument("--task", type=str, default="G1TrackingTennis")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_stoch_samples", type=int, default=4,
                        help="Independent stochastic VAE samples per step, to measure the "
                             "stochastic policy's own self-consistency at a fixed state.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_path", type=str, default=None,
                        help="Optional .npz path to dump raw per-step action arrays.")
    args = parser.parse_args()

    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config

    EnvClass = lmj.registry.get(args.task, "tracking_train_env_class")
    env = EnvClass(config=env_cfg)
    trajectory_data = env.prepare_trajectory(env_cfg.reference_traj_config.name)

    proprio_dim = env.observation_size["proprio_state"][0]
    dif_dim = env.observation_size["dif_state"][0]
    action_dim = env.action_size
    logging.info("proprio_dim=%d dif_dim=%d action_dim=%d", proprio_dim, dif_dim, action_dim)

    teacher_inference_fn, policy_cfg = load_teacher_inference_fn(
        env, trajectory_data, task_cfg, args.teacher_exp_name
    )

    vae = VAEPolicy(
        proprio_dim=proprio_dim, dif_dim=dif_dim, action_dim=action_dim,
        latent_dim=args.latent_dim,
        encoder_hidden=(512, 256), decoder_hidden=(512, 256, 128), prior_hidden=(256, 128),
    )
    vae_ckpt = get_vae_ckpt(args.vae_exp_name)
    if vae_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found for VAE exp: {args.vae_exp_name}")
    logging.info("Loading VAE checkpoint: %s", vae_ckpt)
    vae_params = load_vae_params(vae_ckpt, vae)

    @functools.partial(jax.jit, static_argnums=(2,))
    def vae_decode(proprio, rng, deterministic):
        return vae.apply(vae_params, proprio, method=vae.inference, rng=rng, deterministic=deterministic)

    @functools.partial(jax.jit, static_argnums=(2,))
    def vae_decode_multi(proprio, rng, num_samples):
        """Independent stochastic samples at the same proprio, for self-consistency."""
        rngs = jax.random.split(rng, num_samples)
        decode_one = lambda r: vae.apply(vae_params, proprio, method=vae.inference, rng=r, deterministic=False)
        return jax.vmap(decode_one)(rngs)  # (num_samples, num_envs, action_dim)

    wrapped_env = wrap_fn(env, episode_length=policy_cfg.episode_length)
    reset_fn = jax.jit(wrapped_env.reset)
    step_fn = jax.jit(wrapped_env.step)

    rng = jax.random.PRNGKey(args.seed)
    rng, env_rng = jax.random.split(rng)
    state = reset_fn(jax.random.split(env_rng, args.num_envs), trajectory_data)

    episode_length = policy_cfg.episode_length
    teacher_actions = np.zeros((episode_length, args.num_envs, action_dim), dtype=np.float32)
    vae_det_actions = np.zeros((episode_length, args.num_envs, action_dim), dtype=np.float32)
    vae_stoch_actions = np.zeros((episode_length, args.num_envs, action_dim), dtype=np.float32)
    # Self-consistency spread: std across independent stochastic samples at the same state.
    stoch_self_std = np.zeros((episode_length, args.num_envs), dtype=np.float32)
    fell_ever = np.zeros(args.num_envs, dtype=bool)

    for t in tqdm(range(episode_length)):
        rng, teacher_rng, det_rng, stoch_rng, multi_rng = jax.random.split(rng, 5)

        proprio = state.obs["proprio_state"]

        teacher_action, _ = teacher_inference_fn(state.obs, teacher_rng)
        vae_det_action = vae_decode(proprio, det_rng, True)
        vae_stoch_action = vae_decode(proprio, stoch_rng, False)
        multi_samples = vae_decode_multi(proprio, multi_rng, args.num_stoch_samples)  # (S, N, A)

        teacher_actions[t] = np.array(teacher_action)
        vae_det_actions[t] = np.array(vae_det_action)
        vae_stoch_actions[t] = np.array(vae_stoch_action)
        # Per-env std across samples, averaged over action dims.
        stoch_self_std[t] = np.array(jnp.std(multi_samples, axis=0).mean(axis=-1))

        # Step using the TEACHER action -> keeps states on-trajectory for all three.
        state = step_fn(state, teacher_action, trajectory_data)

        truncation = np.array(state.info["truncation"])
        termination_only = np.clip(np.array(state.done) - truncation, 0.0, 1.0) > 0.5
        fell_ever = fell_ever | termination_only

    print(f"\n=== Action comparison: teacher={args.teacher_exp_name}  vae={args.vae_exp_name} ===")
    print(f"Envs: {args.num_envs}  |  Episode length: {episode_length}")
    print(f"Teacher fall rate during this rollout: {100 * fell_ever.mean():.1f}%")
    print("(Rollout was stepped with the TEACHER action throughout, so all three "
          "policies are compared at identical states.)")

    stats_teacher_vs_det = pairwise_stats(teacher_actions, vae_det_actions, "teacher", "vae_det")
    stats_teacher_vs_stoch = pairwise_stats(teacher_actions, vae_stoch_actions, "teacher", "vae_stoch")
    stats_det_vs_stoch = pairwise_stats(vae_det_actions, vae_stoch_actions, "vae_det", "vae_stoch")

    print_pair_report(stats_teacher_vs_det)
    print_pair_report(stats_teacher_vs_stoch)
    print_pair_report(stats_det_vs_stoch)

    print(f"\n--- Stochastic VAE self-consistency ({args.num_stoch_samples} independent samples/state) ---")
    print(f"Mean per-dim std across samples at the same state: {stoch_self_std.mean():.4f}")
    print("(If this is small relative to 'teacher vs vae_stoch' MAE above, the stochastic "
          "decoder is consistently biased away from the teacher, not just noisy. If it's "
          "comparable in magnitude, most of the teacher/VAE gap is sampling noise.)")

    if args.out_path:
        np.savez(
            args.out_path,
            teacher_actions=teacher_actions,
            vae_det_actions=vae_det_actions,
            vae_stoch_actions=vae_stoch_actions,
            stoch_self_std=stoch_self_std,
            fell_ever=fell_ever,
        )
        print(f"\nSaved raw per-step action arrays to {args.out_path}")


if __name__ == "__main__":
    main()
