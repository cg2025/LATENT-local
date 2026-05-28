"""Train G1 motion-tracking policy via PPO.

Usage:
    python -m latent_mj.learning.train.train_ppo_track_tennis --exp_name my_first_run
    python -m latent_mj.learning.train.train_ppo_track_tennis --exp_name my_first_run --num_envs 512 --no_wandb
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import functools
import json
import time

import jax
from absl import logging

import latent_mj as lmj
from latent_mj.constant import WANDB_PATH_LOG
from latent_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from latent_mj.learning.policy.ppo import train_tracking as ppo
from brax.training.agents.ppo.networks import make_ppo_networks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="G1TrackingTennis")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_updates_per_batch", type=int, default=4)
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--restore_exp_name", type=str, default=None)
    args = parser.parse_args()

    exp_name = args.exp_name or f"tennis_{int(time.time())}"
    ckpt_dir = WANDB_PATH_LOG / "track" / exp_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load config and env
    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    # Apply CLI overrides to policy_cfg before saving
    policy_cfg.num_envs = args.num_envs
    policy_cfg.batch_size = args.batch_size
    policy_cfg.num_minibatches = args.num_minibatches
    policy_cfg.num_updates_per_batch = args.num_updates_per_batch
    if args.num_timesteps is not None:
        policy_cfg.num_timesteps = args.num_timesteps

    # Save config for eval/export scripts
    def _to_serializable(v):
        if hasattr(v, "to_dict"):
            return {k2: _to_serializable(v2) for k2, v2 in v.to_dict().items()}
        if isinstance(v, tuple):
            return list(v)
        return v

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({
            "env_config": env_cfg.to_dict(),
            "policy_config": {
                k: _to_serializable(v) for k, v in policy_cfg.items()
                if k not in ("progress_fn", "wrap_env_fn", "randomization_fn")
            },
        }, f, indent=2)

    EnvClass = lmj.registry.get(args.task, "tracking_train_env_class")
    env = EnvClass(config=env_cfg)
    trajectory_data = env.prepare_trajectory(env_cfg.reference_traj_config.name)

    # Network factory
    network_factory = functools.partial(make_ppo_networks, **policy_cfg.network_factory)

    # WandB
    if not args.no_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "latent"),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=exp_name,
        )

    def progress_fn(step, metrics):
        flat = {k: float(v) for k, v in metrics.items()}
        reward = flat.get("eval/episode_reward", flat.get("training/episode_reward", float("nan")))
        logging.info("step=%d  reward=%.3f", step, reward)
        if not args.no_wandb:
            import wandb
            wandb.log({"step": step, **flat}, step=step)

    def policy_params_fn(step, make_policy, params):
        logging.info("Checkpoint at step %d -> %s", step, ckpt_dir / str(step))

    # Restore checkpoint if requested
    restore_ckpt = None
    if args.restore_exp_name:
        restore_ckpt = lmj.get_latest_ckpt(args.restore_exp_name)

    logging.info("JAX devices: %s", jax.devices())
    logging.info("Starting training: exp=%s", exp_name)

    make_policy, params, metrics = ppo.train(
        environment=env,
        trajectory_data=trajectory_data,
        num_timesteps=policy_cfg.num_timesteps,
        max_devices_per_host=policy_cfg.max_devices_per_host,
        num_envs=policy_cfg.num_envs,
        episode_length=policy_cfg.episode_length,
        action_repeat=policy_cfg.action_repeat,
        wrap_env=True,
        wrap_env_fn=wrap_fn,
        randomization_fn=None,
        learning_rate=policy_cfg.learning_rate,
        entropy_cost=policy_cfg.entropy_cost,
        discounting=policy_cfg.discounting,
        unroll_length=policy_cfg.unroll_length,
        batch_size=policy_cfg.batch_size,
        num_minibatches=policy_cfg.num_minibatches,
        num_updates_per_batch=policy_cfg.num_updates_per_batch,
        normalize_observations=policy_cfg.normalize_observations,
        reward_scaling=policy_cfg.reward_scaling,
        clipping_epsilon=policy_cfg.clipping_epsilon,
        gae_lambda=policy_cfg.gae_lambda,
        max_grad_norm=policy_cfg.max_grad_norm,
        normalize_advantage=policy_cfg.normalize_advantage,
        network_factory=network_factory,
        seed=args.seed,
        num_evals=policy_cfg.num_evals,
        log_training_metrics=True,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        save_checkpoint_path=str(ckpt_dir),
        restore_checkpoint_path=str(restore_ckpt) if restore_ckpt else None,
    )

    logging.info("Training complete. metrics: %s", metrics)
    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
