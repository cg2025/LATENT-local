"""G1 tennis-RETURN task 

  - target locstion implmented here as second sampled point that is stored per ball
    in info["return_target_xy"].
  - Ball position/velocity observation uses a 4-frame sliding window. Velocity is finite-differenced from
    successive position observations, then averaged over the  window
"""

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict
import jax
import jax.numpy as jp
import numpy as np

import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

import latent_mj as lmj
from latent_mj.envs.g1_tracking.train import base_env as g1_base
from latent_mj.envs.g1_tracking import g1_tracking_constants_tennis as consts
from latent_mj.envs.g1_tracking.train import ball_launch as bl
from latent_mj.envs.g1_tracking.train import sample_ball_launch_nn as bln
from latent_mj.learning.policy.high_level.latent_action_barrier import LatentActionBarrier
from latent_mj.utils.dataset.traj_class import Trajectory, TrajectoryData

EPISODE_LENGTH_STEPS = 900 



def torque_step(
    rng: jax.Array,
    model: mjx.Model,
    data: mjx.Data,
    qpos_des: jax.Array,
    kps: jax.Array,
    kds: jax.Array,
    torque_limit: jax.Array,
    n_substeps: int = 1,
) -> tuple[jax.Array, mjx.Data, jax.Array]:
    
    n_joint = consts.NUM_JOINT

    def single_step(carry, _):
        rng, data, _ = carry
        rng, rng_rfi = jax.random.split(rng, 2)
        pos_err = qpos_des - data.qpos[7 : 7 + n_joint]
        vel_err = -data.qvel[6 : 6 + n_joint]
        torque = kps * pos_err + kds * vel_err
        torque = jp.clip(torque, -torque_limit, torque_limit)
        data = data.replace(ctrl=torque)
        data = mjx.step(model, data)
        return (rng, data, torque), None

    initial_torque = jp.zeros_like(torque_limit)
    (final_rng, final_data, final_torque), _ = jax.lax.scan(
        single_step, (rng, data, initial_torque), (), n_substeps
    )
    return final_rng, final_data, final_torque


def get_collision_contact(contact: Any, geom1: int, geom2: int):
    """Whether geom1/geom2 are currently in contact, and the contact point."""
    mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jp.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = jp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    pos = contact.pos[idx]
    return dist < 0, pos


def g1_return_tennis_task_config() -> config_dict.ConfigDict:
    env_config = config_dict.create(
        ctrl_dt=0.02,     # 50 Hz
        sim_dt=0.0005,    # 2000 Hz
        episode_length=EPISODE_LENGTH_STEPS,
        action_repeat=1,
        soft_joint_pos_limit_factor=0.95,

        # --- LAB (Eq. 4) ---
        
        latent_dim=32,
        vae_action_dim=26,   # = len(active_actuator_names)
        vae_checkpoint_path=None,   
        lab_lambda=2.0,       # paper gives no value

        #ball task
        num_balls_per_episode=8,
        ball_launch_interval_s=2.0,
        ball_region="mixed",   # forecourt | backcourt | mixed, see ball_launch.py
        ball_history_len=4,    
        nominal_air_drag_k=0.025,

        termination_config=config_dict.create(
            root_height_threshold=0.3,
        ),

        # Random State Initialization (RSI): sample the robot's initial
        # qpos/qvel from real motion-capture frames  instead of always resetting to the same
        # static default pose. 
        use_rsi=False,
        rsi_dataset_paths=[],

        obs_scales_config=config_dict.create(joint_vel=0.05),
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                gyro=0.2,
                ball_pos=0.01,   
            ),
        ),

        # --- Table 1 reward weights---
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Task
                approach_to_ball=10.0,
                ball_landing=30.0,
                hit_success=200.0,
                # Regularization
                high_level_action=-5e-3,
                torque_penalty=-2e-5,
                lower_body_action_rate=-1.0,
                whole_body_action_rate=-0.5,
                racket_acceleration=-1e-6,
                joint_smoothness=-2e-6,
                correction_action=-5.0,
                correction_action_rate=-1.0,
                wrist_torque=-4e-5,
                wrist_joint_smoothness=-4e-6,
                joint_position_limit=-10.0,
                joint_velocity_limit=-5.0,
                self_collision=-10.0,
                net_clearance=100.0,
                ball_velocity_constraint=50.0,
                racket_velocity_constraint=50.0,
                pelvis_facing_forward=1.0,
                # Termination 
                fall=-200.0,
                miss_ball=-200.0,
                ball_net_collision=-50.0,
                ball_out_of_bounds=-50.0,
                stroke_style_violation=-50.0,
            ),
            auxiliary=config_dict.create(
                # Shaping constants NOT given by the paper (not sure what to put)
                approach_sigma=1.0,
                landing_sigma=1.5,
                net_clearance_margin=0.15,     # extra clearance above net height that saturates the bonus
                ball_speed_cap=35.0,            # ball_velocity_constraint saturation point (m/s)
                racket_speed_cap=20.0,          # racket_velocity_constraint saturation point (m/s)
                wrist_correction_limit=1.2,     # stroke_style_violation threshold on |a_correct| (rad)
            ),
            # Same geom pairs G1TrackingTennisEnv checks -- self-collision
            # hazards (hand clipping torso/thigh/opposite arm mid-swing)
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
        num_timesteps=3_000_000_000,
        max_devices_per_host=8,
        wrap_env=True,
        num_envs=32768,
        episode_length=EPISODE_LENGTH_STEPS,
        action_repeat=1,
        wrap_env_fn=None,
        randomization_fn=None,
        learning_rate=3e-4,
        entropy_cost=0.01,
        discounting=0.97,
        unroll_length=20,
        batch_size=1024,
        num_minibatches=32,
        num_updates_per_batch=4,
        num_resets_per_eval=0,
        normalize_observations=False,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        normalize_advantage=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 512, 256, 256, 128),
            value_hidden_layer_sizes=(512, 512, 256, 256, 128),
            policy_obs_key="state",
            value_obs_key="state",  # no separate privileged obs in this task yet
        ),
        seed=0,
        num_evals=0,
        log_training_metrics=True,
        training_metrics_steps=int(1e6),
        progress_fn=lambda *args: None,
        save_checkpoint_path=None,
        restore_checkpoint_path=None,
        restore_params=None,
        restore_value_fn=True,
    )

    config = config_dict.create(env_config=env_config, policy_config=policy_config)
    return config


lmj.registry.register("G1ReturnTennis", "return_config")(g1_return_tennis_task_config())


@lmj.registry.register("G1ReturnTennis", "return_train_env_class")
class G1TennisReturnEnv(g1_base.G1Env):
    """pi_planner outputs [a_latent, a_correct],
    LAB decodes a_latent through the frozen VAE, a_correct drives the right wrist
    directly, and the ball is launched via ball_launch.sample_ball_launch()."""

    @property
    def action_size(self) -> int:
        # a_latent (LAB residual) + a_correct (3 right-wrist joints)
        return self._config.latent_dim + len(consts.EXCLUDED_ACTION_JOINTs)

    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(with_racket=True, with_ball=True).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        cfg = self._config

        # Actuator/joint index bookkeeping (copied this from G1TrackingTennisEnv) =======
        excluded_joints = consts.EXCLUDED_ACTION_JOINTs
        self._excluded_actuator_ids = jp.array(
            [self.mj_model.actuator(j).id for j in excluded_joints]
        )
        self.all_actuator_names = consts.ACTION_JOINT_NAMES.copy()
        self.all_actuator_ids = jp.array(
            [self.mj_model.actuator(j).id for j in self.all_actuator_names]
        )
        self.active_actuator_names = [j for j in self.all_actuator_names if j not in excluded_joints]
        self.active_actuator_ids = jp.array(
            [self.mj_model.actuator(j).id for j in self.active_actuator_names]
        )
        assert len(self.active_actuator_names) == cfg.vae_action_dim, (
            f"active actuator count {len(self.active_actuator_names)} != "
            f"config.vae_action_dim {cfg.vae_action_dim} -- LAB's decoder output "
            f"width must match exactly."
        )
        self._active_qpos_to_full = jp.array(
            [(self.mj_model.joint(j).qposadr - 7).item() for j in self.active_actuator_names]
        )
        self._excluded_qpos_to_full = jp.array(
            [(self.mj_model.joint(j).qposadr - 7).item() for j in excluded_joints]
        )
        
        self._active_actuator_to_full = jp.array(
            [self.all_actuator_names.index(j) for j in self.active_actuator_names]
        )
        self._excluded_actuator_to_full = jp.array(
            [self.all_actuator_names.index(j) for j in excluded_joints]
        )
        self.obs_joint_ids = jp.array(
            [self.mj_model.actuator(j).id for j in consts.OBS_JOINT_NAMES]
        )
        self._default_qpos = jp.array(consts.DEFAULT_QPOS[7:])
        self._kps = jp.array(consts.KPs)
        self._kds = jp.array(consts.KDs)
        self.torque_limit = jp.array(consts.TORQUE_LIMIT)
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * cfg.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * cfg.soft_joint_pos_limit_factor
        self.dof_vel_limit = jp.array(consts.DOF_VEL_LIMITS)
        self.penalize_collision_pair = jp.array(
            [
                [self.mj_model.geom(pair[0]).id, self.mj_model.geom(pair[1]).id]
                for pair in cfg.reward_config.penalize_collision_on
            ]
        )

        #Site/geom/body ids
        self._pelvis_imu_site_id = self.mj_model.site("imu_in_pelvis").id
        self._floor_geom_id = self.mj_model.geom("floor").id
        self._net_geom_id = self.mj_model.geom(consts.NET_GEOM).id
        self._ball_geom_id = self.mj_model.geom(consts.TENNIS_BALL_GEOM).id
        self._racket_geom_id = self.mj_model.geom(consts.TENNIS_RACKET_COLLISION_GEOM).id
        self._racket_site_id = self.mj_model.site(consts.TENNIS_RACKET_CENTER_SITE).id
        self._ball_site_id = self.mj_model.site(consts.TENNIS_BALL_SITE).id
        self._ball_joint_id = self.mj_model.joint(consts.TENNIS_BALL_JOINT).id
        self._ball_qpos_adr = self.mj_model.jnt_qposadr[self._ball_joint_id]
        self._ball_dof_adr = self.mj_model.jnt_dofadr[self._ball_joint_id]

        # ======= LAB (frozen decoder/prior) =======
        if cfg.vae_checkpoint_path is None:
            raise ValueError(
                "config.vae_checkpoint_path is None. Set it to a train_vae_distill.py "
                )
        self._lab = LatentActionBarrier.from_checkpoint(
            ckpt_path=cfg.vae_checkpoint_path,
            proprio_dim=self._proprio_dim(),
            dif_dim=1,  
            action_dim=cfg.vae_action_dim,
            latent_dim=cfg.latent_dim,
            lam=cfg.lab_lambda,
        )

        # ======= Court geometry =======
        self._net_x = bl.NET_X
        self._net_height = bl.NET_HEIGHT
        self._court_half_length = bl.COURT_HALF_LENGTH
        self._singles_half_width = bl.SINGLES_HALF_WIDTH

        # ======= RSI pool =======
        self._rsi_qpos_pool = None
        self._rsi_qvel_pool = None
        if cfg.use_rsi:
            if not cfg.rsi_dataset_paths:
                raise ValueError(
                    "config.use_rsi=True but rsi_dataset_paths is empty."
                )
            trajectories = [Trajectory.load(p, backend=np) for p in cfg.rsi_dataset_paths]
            self._rsi_qpos_pool, self._rsi_qvel_pool = self._build_rsi_pool(trajectories)

    @staticmethod
    def _build_rsi_pool(trajectories: list) -> tuple[jax.Array, jax.Array]:
        """Flatten a list of loaded Trajectory objects into one (N, nq)/(N, nv)
        pool of individual frames for uniform random sampling at reset.
        """
        all_qpos, all_qvel = [], []
        for traj in trajectories:
            qpos = np.asarray(traj.data.qpos)
            qvel = np.asarray(traj.data.qvel)
            
            if qpos.ndim == 3:
                qpos = qpos.reshape(-1, qpos.shape[-1])
                qvel = qvel.reshape(-1, qvel.shape[-1])
            all_qpos.append(qpos)
            all_qvel.append(qvel)
        pool_qpos = jp.array(np.concatenate(all_qpos, axis=0))
        pool_qvel = jp.array(np.concatenate(all_qvel, axis=0))
        assert pool_qpos.shape[0] == pool_qvel.shape[0], (
            f"qpos pool has {pool_qpos.shape[0]} frames but qvel pool has "
            f"{pool_qvel.shape[0]} -- mismatched trajectory data."
        )
        return pool_qpos, pool_qvel

    def _sample_rsi_state(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Random frame from the RSI pool -> (qpos, qvel) for the robot's 36
        dims (7 root + 29 joints)."""
        idx = jax.random.randint(rng, (), 0, self._rsi_qpos_pool.shape[0])
        return self._rsi_qpos_pool[idx], self._rsi_qvel_pool[idx]

    def _proprio_dim(self) -> int:
        """proprio vector width, needed to
        build the VAEPolicy with the right input shape before any jax tracing."""
        n_active = len(self.active_actuator_names)
        n_obs_joints = len(consts.OBS_JOINT_NAMES)
        # gvec_pelvis(3) + gyro_pelvis(3) + joint_pos(n_obs_joints) +
        # joint_vel(n_obs_joints) + last_motor_targets(len(all_actuator_ids))
        return 3 + 3 + n_obs_joints + n_obs_joints + len(self.all_actuator_names)

    # ------------------------------------------------------------------ #
    # Ball state helpers

    def _ball_qpos(self, data: mjx.Data) -> jax.Array:
        return jax.lax.dynamic_slice(data.qpos, (self._ball_qpos_adr,), (7,))

    def _ball_qvel(self, data: mjx.Data) -> jax.Array:
        return jax.lax.dynamic_slice(data.qvel, (self._ball_dof_adr,), (6,))

    def _set_ball_state(self, data: mjx.Data, pos: jax.Array, vel: jax.Array) -> mjx.Data:
        quat = jp.array([1.0, 0.0, 0.0, 0.0])
        qpos = jax.lax.dynamic_update_slice(data.qpos, jp.concatenate([pos, quat]), (self._ball_qpos_adr,))
        qvel = jax.lax.dynamic_update_slice(data.qvel, jp.concatenate([vel, jp.zeros(3)]), (self._ball_dof_adr,))
        return data.replace(qpos=qpos, qvel=qvel)

    def _launch_next_ball(self, data: mjx.Data, info: dict) -> tuple[mjx.Data, dict]:
        info["rng"], rng_origin, rng_target, rng_time, rng_return_target = jax.random.split(info["rng"], 5)
        
        k = self._config.nominal_air_drag_k

        
        p0 = bl.sample_service_origin(rng_origin)
        target_xy = bl.sample_landing_target(rng_target, region=self._config.ball_region)
        T = bl.sample_flight_time(rng_time, p0, target_xy)
        v0 = bln.solve_launch_velocity_nn_4feat(p0, target_xy, T, k)
        data = self._set_ball_state(data, p0, v0)

        info["ball_pos_history"] = jp.tile(p0, (self._config.ball_history_len, 1))

    
        info["return_target_xy"] = bl.sample_landing_target(rng_return_target, region="mixed")
        info["balls_launched"] += 1
        info["ball_hit_this_ball"] = jp.array(False)
        info["ball_bounced_this_ball"] = jp.array(False)
        info["next_launch_step"] = info["step"] + int(
            round(self._config.ball_launch_interval_s / self.dt)
        )
        return data, info

    # ------------------------------------------------------------------ #

    def reset(self, rng: jax.Array, trajectory_data=None) -> mjx_env.State:
        rng, rng_pose, rng_phase = jax.random.split(rng, 3)

        if self._config.use_rsi:
            robot_qpos, robot_qvel = self._sample_rsi_state(rng_pose)
        else:
            robot_qpos = jp.array(consts.DEFAULT_QPOS)
            robot_qvel = jp.zeros(self.mjx_model.nv - 6)  # nv-6 = robot's 35 tangent-space dof

       
        ball_placeholder_qpos = jp.array([0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0])
        full_qpos = jp.concatenate([robot_qpos, ball_placeholder_qpos])
        full_qvel = jp.concatenate([robot_qvel, jp.zeros(6)])
        data = mjx_env.make_data(
            self.mj_model,
            qpos=full_qpos,
            qvel=full_qvel,
            ctrl=jp.zeros(self.mjx_model.nu),
        )
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "step": 0,
            "last_motor_targets": data.qpos[7:][self.all_actuator_ids],
            "last_action": jp.zeros(self.action_size),
            "last_joint_vel": jp.zeros(self.mjx_model.nv - 6),
            "last_racket_linvel": jp.zeros(3),
            "balls_launched": 0,
            "ball_hit_this_ball": jp.array(False),
            "ball_bounced_this_ball": jp.array(False),
            "return_target_xy": jp.zeros(2),
            "next_launch_step": 0,
            
            "ball_pos_history": jp.zeros((self._config.ball_history_len, 3)),
        }
        data, info = self._launch_next_ball(data, info)

       
        phase_jitter_steps = jax.random.randint(
            rng_phase, (), 0, int(round(self._config.ball_launch_interval_s / self.dt))
        )
        info["next_launch_step"] += phase_jitter_steps

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array, trajectory_data=None) -> mjx_env.State:
        cfg = self._config
        state.info["rng"], rng_step = jax.random.split(state.info["rng"])

        a_latent = action[: cfg.latent_dim]
        a_correct = action[cfg.latent_dim :]

        proprio = self._proprio_from_data(state.data, state.info)
        lab_out = self._lab.decode(proprio[None, :], a_latent[None, :])
        a_body = lab_out["a_body"][0]  # (vae_action_dim,) active-joint PD targets (Eq. 4)

        motor_targets = self._default_qpos.copy()
        motor_targets = motor_targets.at[self.active_actuator_ids].set(a_body)
        motor_targets = motor_targets.at[self._excluded_actuator_ids].set(a_correct)

        state.info["rng"], data, torque = torque_step(
            rng_step,
            self.mjx_model,
            state.data,
            motor_targets,
            kps=self._kps,
            kds=self._kds,
            torque_limit=self.torque_limit,
            n_substeps=self.n_substeps,
        )

        # ball-launch bookkeeping (new ball every ball_launch_interval_s,
        #     up to num_balls_per_episode) ---
        should_launch = (
            (state.info["step"] + 1 >= state.info["next_launch_step"])
            & (state.info["balls_launched"] < cfg.num_balls_per_episode)
        )
        data, info_after_launch = jax.lax.cond(
            should_launch,
            lambda d: self._launch_next_ball(d, dict(state.info)),
            lambda d: (d, dict(state.info)),
            data,
        )
        state.info.update(info_after_launch)

        full_action = jp.zeros(self.action_size)
        full_action = full_action.at[:].set(action)

        rewards, term_flags = self._get_reward(data, a_latent, a_correct, full_action, torque, state.info)
        scales = cfg.reward_config.scales
        scaled = {k: scales[k] * v for k, v in rewards.items()}
        reward = jp.clip(sum(scaled.values()) * self.dt, max=10000.0)

        for k, v in scaled.items():
            state.metrics[f"reward/{k}"] = v

        state.info["last_motor_targets"] = motor_targets
        state.info["last_action"] = full_action
        state.info["last_joint_vel"] = data.qvel[6:]
        state.info["last_racket_linvel"] = mjx_env.get_sensor_data(
            self.mj_model, data, f"{consts.TENNIS_RACKET_CENTER_SITE}_global_linvel"
        )
        state.info["step"] += 1

        done = jp.any(jp.array(list(term_flags.values())))
        obs = self._get_obs(data, state.info)

        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(jp.float32))

    # ------------------------------------------------------------------ #
    # Observation

    def _proprio_from_data(self, data: mjx.Data, info: dict) -> jax.Array:
        """ same construction as other env. noise is added only to the outer policy observation
        below."""
        gyro_pelvis = self.get_gyro(data, "pelvis")
        gvec_pelvis = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        joint_pos = data.qpos[7 : 7 + consts.NUM_JOINT]
        joint_vel = data.qvel[6 : 6 + consts.NUM_JOINT]
        return jp.concatenate([
            gvec_pelvis,
            gyro_pelvis * self._config.obs_scales_config.joint_vel,
            (joint_pos - self._default_qpos)[self.obs_joint_ids],
            joint_vel[self.obs_joint_ids] * self._config.obs_scales_config.joint_vel,
            info["last_motor_targets"],
        ])

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        """Full policy observation: proprio (s_t) + global root pose (g_t) +
        ball/racket state."""
        proprio = self._proprio_from_data(data, info)

        root_pos = data.qpos[:3]
        root_quat = data.qpos[3:7]
        root_linvel = data.qvel[:3]
        root_angvel = data.qvel[3:6]

        racket_pos = data.site_xpos[self._racket_site_id]
        racket_linvel = mjx_env.get_sensor_data(
            self.mj_model, data, f"{consts.TENNIS_RACKET_CENTER_SITE}_global_linvel"
        )

        # Ball position/velocity: four-frame sliding window ...
        # and use the averaged velocity within this window
        ball_pos_true = data.site_xpos[self._ball_site_id]
        info["rng"], rng_bp = jax.random.split(info["rng"])
        noise = self._config.noise_config
        ball_pos_noisy = (
            ball_pos_true + (2 * jax.random.uniform(rng_bp, (3,)) - 1) * noise.level * noise.scales.ball_pos
        )
        history = jp.concatenate([info["ball_pos_history"][1:], ball_pos_noisy[None, :]], axis=0)
        info["ball_pos_history"] = history  # carried forward to the next _get_obs call

        frame_velocities = (history[1:] - history[:-1]) / self.dt  # (history_len - 1, 3)
        ball_vel_avg = jp.mean(frame_velocities, axis=0)
        ball_pos_current = history[-1]

        state = jp.concatenate([
            proprio,
            root_pos,
            root_quat,
            root_linvel,
            root_angvel,
            ball_pos_current - root_pos,   # ball_pos_rel: relative to robot, more learnable than world-abs?
            ball_vel_avg,
            racket_pos - root_pos,          # racket_pos_rel
            racket_linvel,
        ])
        return jp.nan_to_num(state)

    # ------------------------------------------------------------------ #
    # Reward (Table 1). 

    def _reward_collision(self, data: mjx.Data) -> jax.Array:
        """Sum of currently-colliding pairs among penalize_collision_on.

        Uses get_collision_contact (a pair is "colliding" if any
        active contact between them has dist < 0). mujoco playgroudn not working."""
        pair_geom1 = self.penalize_collision_pair[:, 0]
        pair_geom2 = self.penalize_collision_pair[:, 1]

        def is_colliding(g1, g2):
            touching, _ = get_collision_contact(data.contact, g1, g2)
            return touching

        collided_values = jax.vmap(is_colliding)(pair_geom1, pair_geom2)
        return jp.sum(collided_values, axis=-1)

    def _get_reward(self, data, a_latent, a_correct, full_action, torque, info):
        aux = self._config.reward_config.auxiliary
        racket_pos = data.site_xpos[self._racket_site_id]
        ball_pos = data.site_xpos[self._ball_site_id]
        ball_vel = mjx_env.get_sensor_data(self.mj_model, data, "tennis_ball_global_linvel")
        racket_linvel = mjx_env.get_sensor_data(
            self.mj_model, data, f"{consts.TENNIS_RACKET_CENTER_SITE}_global_linvel"
        )

        racket_ball_dist = jp.linalg.norm(racket_pos - ball_pos)
        touching_racket, _ = get_collision_contact(data.contact, self._ball_geom_id, self._racket_geom_id)
        newly_hit = touching_racket & (~info["ball_hit_this_ball"])
        info["ball_hit_this_ball"] = info["ball_hit_this_ball"] | touching_racket

        # --- Task ---
        # approach_to_ball: only active before the ball is hit (afterwards the
        # racket chasing a departed ball would be meaningless).
        approach = jp.exp(-racket_ball_dist / aux.approach_sigma) * (~info["ball_hit_this_ball"])
        hit_success = newly_hit.astype(jp.float32)
        # ball_landing: shaped once the ball is on the ground again after being
        # hit (a bounce following a hit == the return shot has landed).
        touching_floor, _ = get_collision_contact(data.contact, self._ball_geom_id, self._floor_geom_id)
        ball_landed_after_hit = touching_floor & info["ball_hit_this_ball"] & (~info["ball_bounced_this_ball"])
        landing_dist = jp.linalg.norm(ball_pos[:2] - info["return_target_xy"])
        ball_landing = jp.exp(-landing_dist / aux.landing_sigma) * ball_landed_after_hit
        info["ball_bounced_this_ball"] = info["ball_bounced_this_ball"] | touching_floor

        # --- Regularization ---
        high_level_action = jp.sum(jp.square(a_latent))
        torque_penalty = jp.sum(jp.square(torque[self._active_actuator_to_full]))
        lower_body_action_rate = jp.sum(
            jp.square(full_action[: self._config.latent_dim] - info["last_action"][: self._config.latent_dim])
        )
        whole_body_action_rate = jp.sum(jp.square(full_action - info["last_action"]))
        racket_accel = jp.linalg.norm(racket_linvel - info["last_racket_linvel"]) / self.dt
        racket_acceleration = jp.square(racket_accel)
        qvel_active = data.qvel[6:][self._active_qpos_to_full]
        last_vel_active = info["last_joint_vel"][self._active_qpos_to_full]
        qacc_active = (qvel_active - last_vel_active) / self.dt
        joint_smoothness = jp.sum(0.02 * jp.square(qvel_active) + jp.square(qacc_active))
        correction_action = jp.sum(jp.square(a_correct))
        correction_action_rate = jp.sum(
            jp.square(full_action[self._config.latent_dim :] - info["last_action"][self._config.latent_dim :])
        )
        wrist_torque_val = jp.sum(jp.square(torque[self._excluded_actuator_to_full]))
        wrist_qvel = data.qvel[6:][self._excluded_qpos_to_full]
        wrist_joint_smoothness = jp.sum(jp.square(wrist_qvel))

        dof_pos = data.qpos[7:]
        active_dof_pos = dof_pos[self._active_qpos_to_full]
        active_soft_lowers = self._soft_lowers[self._active_qpos_to_full]
        active_soft_uppers = self._soft_uppers[self._active_qpos_to_full]
        out_of_limits = -jp.clip(active_dof_pos - active_soft_lowers, None, 0.0)
        out_of_limits += jp.clip(active_dof_pos - active_soft_uppers, 0.0, None)
        joint_position_limit = jp.clip(jp.sum(out_of_limits), 0.0, 100.0)

        active_dof_vel = data.qvel[6:][self._active_qpos_to_full]
        active_vel_limit = self.dof_vel_limit[self._active_qpos_to_full]
        joint_velocity_limit = jp.sum(jp.clip(jp.abs(active_dof_vel) - active_vel_limit, 0.0, 1.0))

        self_collision = self._reward_collision(data)

        # net_clearance: bonus for clearing the net with margin, evaluated at
        # the step the ball crosses x=net_x while airborne post-hit.
        ball_prev_x = ball_pos[0] - ball_vel[0] * self.sim_dt * self.n_substeps
        crossing_net = (
            info["ball_hit_this_ball"]
            & (jp.sign(ball_pos[0] - self._net_x) != jp.sign(ball_prev_x - self._net_x))
        )
        clearance = ball_pos[2] - self._net_height
        net_clearance = jp.clip(clearance / aux.net_clearance_margin, -1.0, 1.0) * crossing_net

        ball_speed = jp.linalg.norm(ball_vel)
        ball_velocity_constraint = jp.clip(1.0 - ball_speed / aux.ball_speed_cap, -1.0, 1.0)
        racket_speed = jp.linalg.norm(racket_linvel)
        racket_velocity_constraint = jp.clip(1.0 - racket_speed / aux.racket_speed_cap, -1.0, 1.0)

        # pelvis_facing_forward: bonus for facing the net (+x) (not in paper)
        pelvis_forward = data.site_xmat[self._pelvis_imu_site_id][:, 0]
        pelvis_facing_forward = pelvis_forward[0]

        # --- Termination (Table 1) ---
        fall = data.qpos[2] < self._config.termination_config.root_height_threshold
        # miss_ball: incoming ball reaches the robot's baseline without ever
        # being hit (the point is over, unreturned).
        miss_ball = (
            (ball_pos[0] > self._court_half_length * 0.98)
            & (~info["ball_hit_this_ball"])
        )
        touching_net, _ = get_collision_contact(data.contact, self._ball_geom_id, self._net_geom_id)
        ball_net_collision = touching_net & info["ball_hit_this_ball"]  # the RETURN shot hit the net
        out_of_bounds = ball_landed_after_hit & (
            (jp.abs(ball_pos[1]) > self._singles_half_width)
            | (ball_pos[0] < 0.0)
            | (ball_pos[0] > self._court_half_length)
        )
        # stroke_style_violation: NOT a paper-specified check 
        # implemented here as a simple wrist-correction-magnitude guard
        stroke_style_violation = jp.any(jp.abs(a_correct) > aux.wrist_correction_limit)

        rewards = dict(
            approach_to_ball=approach,
            ball_landing=ball_landing,
            hit_success=hit_success,
            high_level_action=high_level_action,
            torque_penalty=torque_penalty,
            lower_body_action_rate=lower_body_action_rate,
            whole_body_action_rate=whole_body_action_rate,
            racket_acceleration=racket_acceleration,
            joint_smoothness=joint_smoothness,
            correction_action=correction_action,
            correction_action_rate=correction_action_rate,
            wrist_torque=wrist_torque_val,
            wrist_joint_smoothness=wrist_joint_smoothness,
            joint_position_limit=joint_position_limit,
            joint_velocity_limit=joint_velocity_limit,
            self_collision=self_collision,
            net_clearance=net_clearance,
            ball_velocity_constraint=ball_velocity_constraint,
            racket_velocity_constraint=racket_velocity_constraint,
            pelvis_facing_forward=pelvis_facing_forward,
            fall=fall.astype(jp.float32),
            miss_ball=miss_ball.astype(jp.float32),
            ball_net_collision=ball_net_collision.astype(jp.float32),
            ball_out_of_bounds=out_of_bounds.astype(jp.float32),
            stroke_style_violation=stroke_style_violation.astype(jp.float32),
        )
        term_flags = dict(
            fall=fall,
            miss_ball=miss_ball,
            ball_net_collision=ball_net_collision,
            ball_out_of_bounds=out_of_bounds,
            stroke_style_violation=stroke_style_violation,
            episode_done=info["step"] >= self._config.episode_length,
            all_balls_done=(
                (info["balls_launched"] >= self._config.num_balls_per_episode)
                & info["ball_bounced_this_ball"]
            ),
        )
        return rewards, term_flags
