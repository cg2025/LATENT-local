"""Evaluate the distilled VAE on the simple velocity task (no tennis MDP).

Conditions (each run over a grid of FIXED commands):
  prior mean   : a_latent = 0  -> z = mu_p(proprio)       (VAE alone, no steering)
  random z     : a_latent ~ N(0, I) -> z = mu_p + lam*sigma_p*tanh(a)
  HL policy    : trained pi_planner (only if --hl_exp_name is given)

Read: achieved vs commanded speed, fall rate, survival time.

Usage:
  python -m latent_mj.eval.vae.eval_velocity_vae \
      --vae_checkpoint_path /abs/path/storage/logs/vae/vae_xsens_s16_v1/checkpoints/00002000_final \
      --action_mode absolute            # then again with default_residual
  ... --hl_exp_name vel_hl_v1           # adds the trained steering policy
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

import latent_mj as lmj
from latent_mj.constant import WANDB_PATH_LOG
from latent_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from latent_mj.learning.policy.ppo import train_tracking as ppo
from brax.training.agents.ppo.networks import make_ppo_networks

TASK = "G1VelocityVAE"


def load_hl_policy(env, task_cfg, exp_name, deterministic=True):
    ckpt = lmj.get_latest_ckpt(f"return/{exp_name}")
    if ckpt is None:
        raise FileNotFoundError(f"no HL checkpoint for return/{exp_name}")
    policy_cfg = task_cfg.policy_config
    with open(WANDB_PATH_LOG / "return" / exp_name / "checkpoints" / "config.json") as f:
        policy_cfg.update(json.load(f)["policy_config"])
    network_factory = functools.partial(make_ppo_networks, **policy_cfg.network_factory)
    make_policy, params, _ = ppo.train(
        environment=env, trajectory_data=None, num_timesteps=0, num_envs=1,
        episode_length=policy_cfg.episode_length, wrap_env=True, wrap_env_fn=wrap_fn,
        network_factory=network_factory, restore_checkpoint_path=str(ckpt),
        normalize_observations=policy_cfg.normalize_observations,
    )
    return make_policy(params, deterministic=deterministic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_checkpoint_path", required=True, help="absolute path")
    ap.add_argument("--hl_exp_name", default="")
    ap.add_argument("--action_mode", default="absolute", choices=["absolute", "default_residual"])
    ap.add_argument("--lab_lambda", type=float, default=None)
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--vx", default="-0.4,0.0,0.4,0.8")
    ap.add_argument("--yaw", default="-0.5,0.0,0.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    task_cfg = lmj.registry.get(TASK, "return_config")
    env_cfg = task_cfg.env_config
    if args.hl_exp_name:  # match the obs/LAB settings the policy was trained with
        with open(WANDB_PATH_LOG / "return" / args.hl_exp_name / "checkpoints" / "config.json") as f:
            env_cfg.update(json.load(f)["env_config"])
    env_cfg.vae_checkpoint_path = args.vae_checkpoint_path
    env_cfg.action_mode = args.action_mode
    if args.lab_lambda is not None:
        env_cfg.lab_lambda = args.lab_lambda
    env_cfg.command_resample_steps = 10**9  # commands are fixed by this script

    env = lmj.registry.get(TASK, "return_train_env_class")(config=env_cfg)
    T = int(round(args.seconds / env.dt))
    N = args.num_envs
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    get_obs = jax.jit(jax.vmap(env._get_obs))

    conditions = {
        "prior mean (z=mu_p)": lambda obs, k: jnp.zeros((N, env.action_size)),
        "random residual a~N(0,1)": lambda obs, k: jax.random.normal(k, (N, env.action_size)),
    }
    if args.hl_exp_name:
        pol = jax.jit(load_hl_policy(env, task_cfg, args.hl_exp_name))
        conditions["HL policy"] = lambda obs, k: pol(obs, k)[0]

    vxs = [float(x) for x in args.vx.split(",")]
    yaws = [float(x) for x in args.yaw.split(",")]
    results = []
    print(f"mode={args.action_mode} lam={env_cfg.lab_lambda} envs={N} T={T} steps ({args.seconds}s)\n")
    print(f"{'condition':<26}{'cmd vx':>7}{'cmd yaw':>8} | {'vx':>6}{'|lin err|':>10}{'yaw':>7}{'fell':>6}{'surv s':>8}")
    for name, act in conditions.items():
        for vx in vxs:
            for yaw in yaws:
                key = jax.random.PRNGKey(args.seed)
                key, rk = jax.random.split(key)
                state = reset(jax.random.split(rk, N))
                cmd = jnp.tile(jnp.array([vx, 0.0, yaw]), (N, 1))
                info = {**state.info, "command": cmd}
                state = state.replace(info=info, obs=get_obs(state.data, info))
                alive = np.ones(N, bool)
                sums = {k: np.zeros(N) for k in ("achieved_vx", "lin_error", "achieved_yaw")}
                n_alive = np.zeros(N)
                for t in range(T):
                    key, ak = jax.random.split(key)
                    state = step(state, act(state.obs, ak))
                    for k in sums:
                        sums[k] += np.asarray(state.metrics[k]) * alive
                    n_alive += alive
                    alive &= ~(np.asarray(state.done) > 0.5)
                m = {k: float(v.sum() / max(n_alive.sum(), 1)) for k, v in sums.items()}
                fell = float((~alive).mean()) * 100
                surv = float(n_alive.mean() * env.dt)
                print(f"{name:<26}{vx:7.2f}{yaw:8.2f} | {m['achieved_vx']:6.2f}{m['lin_error']:10.2f}"
                      f"{m['achieved_yaw']:7.2f}{fell:5.0f}%{surv:8.2f}")
                results.append(dict(condition=name, cmd_vx=vx, cmd_yaw=yaw, fell_pct=fell,
                                    survival_s=surv, **m))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(args=vars(args), results=results), f, indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
