"""Online distillation training with variational bottleneck.

Implements Section 3.2.2:
  - Loads a pretrained PPO tracker as teacher
  - Trains a student VAE policy via DAgger-style online distillation
  - Student uses encoder-decoder with learnable prior (conditional VAE)

Usage:
    python -m latent_mj.learning.train.train_vae_distill \
        --teacher_exp_name short_run_1 \
        --exp_name vae_distill_1 \
        --no_wandb

    # With more envs for faster data collection
    python -m latent_mj.learning.train.train_vae_distill \
        --teacher_exp_name short_run_1 \
        --exp_name vae_distill_1 \
        --num_envs 1024 \
        --no_wandb
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import functools
import time
from pathlib import Path
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from absl import logging

import latent_mj as lmj
from latent_mj.constant import WANDB_PATH_LOG
from latent_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from latent_mj.learning.policy.ppo import train_tracking as ppo
from latent_mj.learning.policy.vae.networks import VAEPolicy, vae_loss
from brax.training.agents.ppo.networks import make_ppo_networks


# DAgger replay buffer

class DAggerBuffer:
    """Simple replay buffer for DAgger distillation data."""

    def __init__(self, max_size: int, state_dim: int, privileged_dim: int, action_dim: int):
        self.max_size = max_size
        self.state_dim = state_dim
        self.privileged_dim = privileged_dim
        self.action_dim = action_dim
        self._states = np.zeros((max_size, state_dim), dtype=np.float32)
        self._privileged = np.zeros((max_size, privileged_dim), dtype=np.float32)
        self._teacher_actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, states, privileged, teacher_actions):
        """Add batch of transitions."""
        n = states.shape[0]
        if self._ptr + n > self.max_size:
            # Wrap around
            end = self.max_size - self._ptr
            self._states[self._ptr:] = states[:end]
            self._privileged[self._ptr:] = privileged[:end]
            self._teacher_actions[self._ptr:] = teacher_actions[:end]
            remainder = n - end
            self._states[:remainder] = states[end:]
            self._privileged[:remainder] = privileged[end:]
            self._teacher_actions[:remainder] = teacher_actions[end:]
            self._ptr = remainder
        else:
            self._states[self._ptr:self._ptr+n] = states
            self._privileged[self._ptr:self._ptr+n] = privileged
            self._teacher_actions[self._ptr:self._ptr+n] = teacher_actions
            self._ptr = (self._ptr + n) % self.max_size
        self._size = min(self._size + n, self.max_size)

    def sample(self, batch_size: int):
        """Sample a random batch."""
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self._states[idx],
            self._privileged[idx],
            self._teacher_actions[idx],
        )

    @property
    def size(self):
        return self._size



# Training state
class VAETrainState(flax.struct.PyTreeNode):
    params: Any
    opt_state: Any
    step: int


# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_exp_name", type=str, required=True,
                        help="Exp name of the pretrained PPO tracker (teacher)")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name for this VAE distillation run")
    parser.add_argument("--task", type=str, default="G1TrackingTennis")
    parser.add_argument("--num_envs", type=int, default=1024,
                        help="Number of parallel envs for data collection")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimension of latent skill code z")
    parser.add_argument("--buffer_size", type=int, default=500000,
                        help="Max size of DAgger replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for VAE gradient updates")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lambda_action", type=float, default=1.0,
                        help="Weight for action reconstruction loss")
    parser.add_argument("--lambda_kl", type=float, default=0.01,
                        help="Weight for KL divergence loss")
    parser.add_argument("--num_iterations", type=int, default=1000,
                        help="Number of DAgger iterations")
    parser.add_argument("--collect_steps", type=int, default=20,
                        help="Env steps per DAgger collection round")
    parser.add_argument("--grad_updates_per_iter", type=int, default=32,
                        help="Gradient updates per DAgger round")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    exp_name = args.exp_name or f"vae_{int(time.time())}"
    ckpt_dir = WANDB_PATH_LOG / "vae" / exp_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logging.info("JAX devices: %s", jax.devices())

    # Load task config and env 
    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    EnvClass = lmj.registry.get(args.task, "tracking_train_env_class")
    env = EnvClass(config=env_cfg)
    trajectory_data = env.prepare_trajectory(env_cfg.reference_traj_config.name)

    state_dim = env.observation_size["state"][0]
    privileged_dim = env.observation_size["privileged_state"][0]
    action_dim = env.action_size

    logging.info("state_dim=%d, privileged_dim=%d, action_dim=%d, latent_dim=%d",
                 state_dim, privileged_dim, action_dim, args.latent_dim)

    # Load pretrained teacher (PPO tracker) 
    logging.info("Loading teacher from exp: %s", args.teacher_exp_name)
    teacher_ckpt = lmj.get_latest_ckpt(f"track/{args.teacher_exp_name}")
    if teacher_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found for teacher: {args.teacher_exp_name}")
    logging.info("Teacher checkpoint: %s", teacher_ckpt)

    import json
    config_path = WANDB_PATH_LOG / "track" / args.teacher_exp_name / "checkpoints" / "config.json"
    with open(config_path) as f:
        teacher_config = json.load(f)

    teacher_policy_cfg = task_cfg.policy_config
    teacher_policy_cfg.update(teacher_config["policy_config"])

    network_factory = functools.partial(make_ppo_networks, **teacher_policy_cfg.network_factory)

    # Restore teacher policy
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
        normalize_observations=teacher_policy_cfg.normalize_observations,
    )
    teacher_inference_fn = make_teacher_fn(teacher_params, deterministic=True)
    logging.info("Teacher loaded successfully.")

    #  Build student VAE policy 
    vae = VAEPolicy(
        state_dim=state_dim,
        privileged_dim=privileged_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        encoder_hidden=(512, 256),
        decoder_hidden=(512, 256, 128),
        prior_hidden=(256, 128),
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    dummy_state = jnp.zeros((1, state_dim))
    dummy_privileged = jnp.zeros((1, privileged_dim))
    vae_params = vae.init(init_rng, dummy_state, dummy_privileged, init_rng)
    logging.info("VAE initialized. Param count: %d",
                 sum(x.size for x in jax.tree_util.tree_leaves(vae_params)))

    #  Optimizer (clip global grad norm to tame KL-loss spikes, then Adam)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(args.learning_rate),
    )
    opt_state = optimizer.init(vae_params)

    #  JIT-compiled training step 
    @jax.jit
    def train_step(params, opt_state, states, privileged, teacher_actions, rng):
        def loss_fn(params):
            action_pred, mu, log_sigma, mu_prior, log_sigma_prior = vae.apply(
                params, states, privileged, rng
            )
            total, l_action, l_kl = vae_loss(
                action_pred, teacher_actions,
                mu, log_sigma, mu_prior, log_sigma_prior,
                lambda_action=args.lambda_action,
                lambda_kl=args.lambda_kl,
            )
            return total, (l_action, l_kl)

        (loss, (l_action, l_kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, l_action, l_kl

    #   teacher inference
    @jax.jit
    def get_teacher_action(obs, rng):
        action, _ = teacher_inference_fn(obs, rng)
        return action

    #   student inference
    @jax.jit
    def get_student_action(params, states, rng):
        return vae.apply(params, states, rng, method=VAEPolicy.inference)

    #  Environment setup for data collection 
    wrapped_env = wrap_fn(env, episode_length=policy_cfg.episode_length)
    reset_fn = jax.jit(wrapped_env.reset)
    step_fn = jax.jit(wrapped_env.step)

    rng, env_rng = jax.random.split(rng)
    env_rngs = jax.random.split(env_rng, args.num_envs)
    env_state = reset_fn(env_rngs, trajectory_data)
    # DAgger replay buffer 
    buffer = DAggerBuffer(
        max_size=args.buffer_size,
        state_dim=state_dim,
        privileged_dim=privileged_dim,
        action_dim=action_dim,
    )

    if not args.no_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "latent"),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=exp_name,
            config=vars(args),
        )

    # Training loop 
    logging.info("Starting VAE distillation training...")
    logging.info("num_iterations=%d, collect_steps=%d, grad_updates=%d",
                 args.num_iterations, args.collect_steps, args.grad_updates_per_iter)

    for iteration in range(args.num_iterations):
        # Collect data with student policy (DAgger)
        # Roll out student in env, query teacher for labels
        for _ in range(args.collect_steps):
            rng, step_rng, teacher_rng = jax.random.split(rng, 3)

            # Get current observations
            states = env_state.obs["state"]          # (num_envs, state_dim)
            privileged = env_state.obs["privileged_state"]  # (num_envs, privileged_dim)

            # Query teacher for action labels
            teacher_actions = get_teacher_action(env_state.obs, teacher_rng)

            # Add to buffer
            buffer.add(
                np.array(states),
                np.array(privileged),
                np.array(teacher_actions),
            )

            # Step env with student actions using the deployment policy
            student_actions = get_student_action(vae_params, states, step_rng)

            env_state = step_fn(env_state, student_actions, trajectory_data)

        # Gradient updates on buffer
        if buffer.size < args.batch_size:
            continue

        grad_losses = []
        for _ in range(args.grad_updates_per_iter):
            rng, update_rng = jax.random.split(rng)
            batch_states, batch_privileged, batch_teacher_actions = buffer.sample(args.batch_size)

            batch_states = jnp.array(batch_states)
            batch_privileged = jnp.array(batch_privileged)
            batch_teacher_actions = jnp.array(batch_teacher_actions)

            vae_params, opt_state, loss, l_action, l_kl = train_step(
                vae_params, opt_state,
                batch_states, batch_privileged, batch_teacher_actions,
                update_rng,
            )
            grad_losses.append((float(loss), float(l_action), float(l_kl)))

        avg_loss = np.mean([x[0] for x in grad_losses])
        avg_action_loss = np.mean([x[1] for x in grad_losses])
        avg_kl_loss = np.mean([x[2] for x in grad_losses])

        logging.info("iter=%d  buffer=%d  loss=%.4f  l_action=%.4f  l_kl=%.4f",
                     iteration, buffer.size, avg_loss, avg_action_loss, avg_kl_loss)

        if not args.no_wandb:
            import wandb
            wandb.log({
                "iteration": iteration,
                "buffer_size": buffer.size,
                "loss/total": avg_loss,
                "loss/action": avg_action_loss,
                "loss/kl": avg_kl_loss,
            })

        #  Save checkpoint every 100 iterations 
        if iteration % 100 == 0 and iteration > 0:
            ckpt_path = ckpt_dir / f"{iteration:08d}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            import orbax.checkpoint as ocp
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(str(ckpt_path), vae_params, force=True)
            logging.info("Saved checkpoint at iteration %d -> %s", iteration, ckpt_path)

    
    final_ckpt = ckpt_dir / f"{args.num_iterations:08d}_final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    import orbax.checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(str(final_ckpt), vae_params, force=True)
    logging.info("Training complete. Final checkpoint: %s", final_ckpt)

    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
