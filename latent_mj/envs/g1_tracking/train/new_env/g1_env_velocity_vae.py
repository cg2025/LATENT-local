"""G1 velocity-command task on top of the frozen distilled VAE (+ LAB).

Simple stand-in for the tennis-return MDP, used to evaluate the VAE's latent
space: the high-level policy outputs only a_latent (32-d); LAB + the frozen
decoder turn it into joint PD targets (Eq. 4); reward = track a commanded
(vx, vy, yaw_rate) in the pelvis heading frame. No ball, no court.

Registered as "G1VelocityVAE" under the same registry keys the tennis-return
task uses, so train_ppo_hl_tennis.py works with `--task G1VelocityVAE`.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

import latent_mj as lmj
from latent_mj.envs.g1_tracking.train import base_env as g1_base
from latent_mj.envs.g1_tracking import g1_tracking_constants_tennis as consts
from latent_mj.envs.g1_tracking.train.new_env.g1_env_return_tennis import (
    get_collision_contact,
    torque_step,
)
from latent_mj.learning.policy.high_level.latent_action_barrier import LatentActionBarrier

EPISODE_LENGTH_STEPS = 500  # 10 s at 50 Hz

EXTRA_METRICS = ("achieved_vx", "achieved_vy", "achieved_yaw", "lin_error", "yaw_error", "latent_sq")


def g1_velocity_vae_task_config() -> config_dict.ConfigDict:
    env_config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,  # SAME as the tracker the VAE was distilled in (return env uses 0.0005)
        episode_length=EPISODE_LENGTH_STEPS,
        action_repeat=1,
        soft_joint_pos_limit_factor=0.95,
        # --- LAB ---
        latent_dim=32,
        vae_action_dim=26,
        vae_checkpoint_path=None,
        lab_lambda=2.0,
        # How the decoder output becomes a PD target. The tracker defined its
        # action as a deviation from the REFERENCE joint pos (not available here).
        #   "absolute":         target = a_body            (what G1ReturnTennis does)
        #   "default_residual": target = default_pose + a_body
        action_mode="absolute",
        # --- command ---
        cmd_vx_min=-0.5, cmd_vx_max=1.0,
        cmd_vy_min=-0.4, cmd_vy_max=0.4,
        cmd_yaw_min=-0.6, cmd_yaw_max=0.6,
        command_resample_steps=250,   # 5 s
        zero_command_prob=0.1,
        termination_config=config_dict.create(root_height_threshold=0.3),
        obs_scales_config=config_dict.create(joint_vel=0.05),
        reward_config=config_dict.create(
            scales=config_dict.create(
                track_lin_vel=10.0,
                track_ang_vel=5.0,
                alive=1.0,
                high_level_action=-5e-3,
                torque_penalty=-2e-5,
                whole_body_action_rate=-0.5,
                joint_smoothness=-2e-6,
                joint_position_limit=-10.0,
                joint_velocity_limit=-5.0,
                self_collision=-10.0,
                fall=-200.0,
            ),
            auxiliary=config_dict.create(lin_sigma=0.25, ang_sigma=0.25),
            penalize_collision_on=[
                ["left_hand_collision", "left_thigh"],
                ["right_hand_collision", "right_thigh"],
                ["left_hand_collision", "right_hand_collision"],
                ["left_hand_collision", "right_wrist_pitch_collision"],
                ["right_hand_collision", "left_wrist_pitch_collision"],
            ],
        ),
    )
    policy_config = config_dict.create(
        num_timesteps=300_000_000,
        max_devices_per_host=8,
        wrap_env=True,
        num_envs=8192,
        episode_length=EPISODE_LENGTH_STEPS,
        action_repeat=1,
        wrap_env_fn=None,
        randomization_fn=None,
        learning_rate=3e-4,
        entropy_cost=0.01,
        discounting=0.97,
        unroll_length=20,
        batch_size=256,          # 256 * 32 * 20 == 8192 * 20
        num_minibatches=32,
        num_updates_per_batch=4,
        num_resets_per_eval=0,
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        normalize_advantage=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="state",
        ),
        seed=0,
        num_evals=20,
        log_training_metrics=True,
        training_metrics_steps=int(1e6),
        progress_fn=lambda *args: None,
        save_checkpoint_path=None,
        restore_checkpoint_path=None,
        restore_params=None,
        restore_value_fn=True,
    )
    return config_dict.create(env_config=env_config, policy_config=policy_config)


lmj.registry.register("G1VelocityVAE", "return_config")(g1_velocity_vae_task_config())


@lmj.registry.register("G1VelocityVAE", "return_train_env_class")
class G1VelocityVAEEnv(g1_base.G1Env):
    """pi_planner outputs a_latent only; LAB + frozen VAE decoder -> PD targets."""

    @property
    def action_size(self) -> int:
        return self._config.latent_dim

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(with_racket=True, with_ball=False).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        cfg = self._config
        assert cfg.action_mode in ("absolute", "default_residual"), cfg.action_mode
        excluded = consts.EXCLUDED_ACTION_JOINTs
        self.all_actuator_names = consts.ACTION_JOINT_NAMES.copy()
        self.all_actuator_ids = jp.array([self.mj_model.actuator(j).id for j in self.all_actuator_names])
        self.active_actuator_names = [j for j in self.all_actuator_names if j not in excluded]
        self.active_actuator_ids = jp.array([self.mj_model.actuator(j).id for j in self.active_actuator_names])
        assert len(self.active_actuator_names) == cfg.vae_action_dim
        self._active_qpos_to_full = jp.array(
            [(self.mj_model.joint(j).qposadr - 7).item() for j in self.active_actuator_names]
        )
        self._active_actuator_to_full = jp.array(
            [self.all_actuator_names.index(j) for j in self.active_actuator_names]
        )
        self.obs_joint_ids = jp.array([self.mj_model.actuator(j).id for j in consts.OBS_JOINT_NAMES])
        self._default_qpos = jp.array(consts.DEFAULT_QPOS[7:])
        self._kps = jp.array(consts.KPs)
        self._kds = jp.array(consts.KDs)
        self.torque_limit = jp.array(consts.TORQUE_LIMIT)
        lowers, uppers = self.mj_model.jnt_range[1:].T
        c, r = (lowers + uppers) / 2, uppers - lowers
        self._soft_lowers = c - 0.5 * r * cfg.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * cfg.soft_joint_pos_limit_factor
        self.dof_vel_limit = jp.array(consts.DOF_VEL_LIMITS)
        self.penalize_collision_pair = jp.array(
            [[self.mj_model.geom(p[0]).id, self.mj_model.geom(p[1]).id]
             for p in cfg.reward_config.penalize_collision_on]
        )
        self._pelvis_imu_site_id = self.mj_model.site("imu_in_pelvis").id

        if cfg.vae_checkpoint_path is None:
            raise ValueError("config.vae_checkpoint_path is None; pass a train_vae_distill.py checkpoint.")
        self._lab = LatentActionBarrier.from_checkpoint(
            ckpt_path=cfg.vae_checkpoint_path,
            proprio_dim=self._proprio_dim(),
            dif_dim=58,  # dif_joint_pos(29) + dif_joint_vel(29), as in Stage 2
            action_dim=cfg.vae_action_dim,
            latent_dim=cfg.latent_dim,
            lam=cfg.lab_lambda,
        )

    def _proprio_dim(self) -> int:
        n = len(consts.OBS_JOINT_NAMES)
        return 3 + 3 + n + n + len(self.all_actuator_names)

    # ------------------------------------------------------------------ #
    def _sample_command(self, rng: jax.Array) -> jax.Array:
        c = self._config
        r1, r2, r3, r4 = jax.random.split(rng, 4)
        cmd = jp.stack([
            jax.random.uniform(r1, (), minval=c.cmd_vx_min, maxval=c.cmd_vx_max),
            jax.random.uniform(r2, (), minval=c.cmd_vy_min, maxval=c.cmd_vy_max),
            jax.random.uniform(r3, (), minval=c.cmd_yaw_min, maxval=c.cmd_yaw_max),
        ])
        return jp.where(jax.random.uniform(r4) < c.zero_command_prob, jp.zeros(3), cmd)

    def _heading_velocities(self, data: mjx.Data):
        """(v_forward, v_lateral, yaw_rate) of the pelvis in its heading frame."""
        xmat = data.site_xmat[self._pelvis_imu_site_id]  # (3,3) body -> world
        yaw = jp.arctan2(xmat[1, 0], xmat[0, 0])
        c, s = jp.cos(yaw), jp.sin(yaw)
        vw = data.qvel[:3]
        v_x = c * vw[0] + s * vw[1]
        v_y = -s * vw[0] + c * vw[1]
        yaw_rate = (xmat @ self.get_gyro(data, "pelvis"))[2]
        return v_x, v_y, yaw_rate

    # ------------------------------------------------------------------ #
    def reset(self, rng: jax.Array, trajectory_data=None) -> mjx_env.State:
        rng, rng_cmd = jax.random.split(rng)
        data = mjx_env.make_data(
            self.mj_model,
            qpos=jp.array(consts.DEFAULT_QPOS),
            qvel=jp.zeros(self.mjx_model.nv),
            ctrl=jp.zeros(self.mjx_model.nu),
        )
        data = mjx.forward(self.mjx_model, data)
        info = {
            "rng": rng,
            "step": 0,
            "command": self._sample_command(rng_cmd),
            "last_motor_targets": data.qpos[7:][self.all_actuator_ids],
            "last_action": jp.zeros(self.action_size),
            "last_joint_vel": jp.zeros(self.mjx_model.nv - 6),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        metrics.update({k: jp.zeros(()) for k in EXTRA_METRICS})
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array, trajectory_data=None) -> mjx_env.State:
        cfg = self._config
        state.info["rng"], rng_step, rng_cmd = jax.random.split(state.info["rng"], 3)

        a_latent = action[: cfg.latent_dim]
        proprio = self._proprio_from_data(state.data, state.info)
        a_body = self._lab.decode(proprio[None, :], a_latent[None, :])["a_body"][0]
        if cfg.action_mode == "default_residual":
            a_body = self._default_qpos[self.active_actuator_ids] + a_body
        motor_targets = self._default_qpos.copy().at[self.active_actuator_ids].set(a_body)

        state.info["rng"], data, torque = torque_step(
            rng_step, self.mjx_model, state.data, motor_targets,
            kps=self._kps, kds=self._kds, torque_limit=self.torque_limit,
            n_substeps=self.n_substeps,
        )

        state.info["step"] += 1
        resample = (state.info["step"] % cfg.command_resample_steps) == 0
        state.info["command"] = jp.where(resample, self._sample_command(rng_cmd), state.info["command"])

        rewards, fall, extras = self._get_reward(data, a_latent, action, torque, state.info)
        scales = cfg.reward_config.scales
        scaled = {k: scales[k] * v for k, v in rewards.items()}
        reward = jp.clip(sum(scaled.values()) * self.dt, max=10000.0)
        for k, v in scaled.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics.update(extras)

        state.info["last_motor_targets"] = motor_targets
        state.info["last_action"] = action
        state.info["last_joint_vel"] = data.qvel[6:]

        obs = self._get_obs(data, state.info)
        return state.replace(data=data, obs=obs, reward=reward, done=fall.astype(jp.float32))

    # ------------------------------------------------------------------ #
    def _proprio_from_data(self, data: mjx.Data, info: dict) -> jax.Array:
        """Identical to the tracker's proprio_obs_keys (gvec, gyro, joint_pos, joint_vel, last targets)."""
        gyro = self.get_gyro(data, "pelvis")
        gvec = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        joint_pos = data.qpos[7 : 7 + consts.NUM_JOINT]
        joint_vel = data.qvel[6 : 6 + consts.NUM_JOINT]
        s = self._config.obs_scales_config.joint_vel
        return jp.concatenate([
            gvec,
            gyro * s,
            (joint_pos - self._default_qpos)[self.obs_joint_ids],
            joint_vel[self.obs_joint_ids] * s,
            info["last_motor_targets"],
        ])

    def _get_obs(self, data: mjx.Data, info: dict):
        proprio = self._proprio_from_data(data, info)
        v_x, v_y, _ = self._heading_velocities(data)
        state = jp.concatenate([proprio, jp.stack([v_x, v_y]), info["command"]])
        return {"state": jp.nan_to_num(state)}

    def _get_reward(self, data, a_latent, action, torque, info):
        cfg = self._config
        aux = cfg.reward_config.auxiliary
        v_x, v_y, yaw_rate = self._heading_velocities(data)
        cmd = info["command"]
        lin_err2 = (v_x - cmd[0]) ** 2 + (v_y - cmd[1]) ** 2
        yaw_err = yaw_rate - cmd[2]

        qvel_a = data.qvel[6:][self._active_qpos_to_full]
        qacc_a = (qvel_a - info["last_joint_vel"][self._active_qpos_to_full]) / self.dt
        pos_a = data.qpos[7:][self._active_qpos_to_full]
        oob = -jp.clip(pos_a - self._soft_lowers[self._active_qpos_to_full], None, 0.0)
        oob += jp.clip(pos_a - self._soft_uppers[self._active_qpos_to_full], 0.0, None)
        vel_lim = self.dof_vel_limit[self._active_qpos_to_full]
        pair = self.penalize_collision_pair
        collided = jax.vmap(lambda a, b: get_collision_contact(data.contact, a, b)[0])(pair[:, 0], pair[:, 1])
        fall = data.qpos[2] < cfg.termination_config.root_height_threshold

        rewards = dict(
            track_lin_vel=jp.exp(-lin_err2 / aux.lin_sigma),
            track_ang_vel=jp.exp(-(yaw_err ** 2) / aux.ang_sigma),
            alive=jp.ones(()),
            high_level_action=jp.sum(jp.square(a_latent)),
            torque_penalty=jp.sum(jp.square(torque[self._active_actuator_to_full])),
            whole_body_action_rate=jp.sum(jp.square(action - info["last_action"])),
            joint_smoothness=jp.sum(0.02 * jp.square(qvel_a) + jp.square(qacc_a)),
            joint_position_limit=jp.clip(jp.sum(oob), 0.0, 100.0),
            joint_velocity_limit=jp.sum(jp.clip(jp.abs(qvel_a) - vel_lim, 0.0, 1.0)),
            self_collision=jp.sum(collided),
            fall=fall.astype(jp.float32),
        )
        extras = dict(
            achieved_vx=v_x, achieved_vy=v_y, achieved_yaw=yaw_rate,
            lin_error=jp.sqrt(lin_err2), yaw_error=jp.abs(yaw_err),
            latent_sq=jp.mean(jp.square(a_latent)),
        )
        return rewards, fall, extras
