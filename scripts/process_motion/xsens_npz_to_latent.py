"""Convert the Xsens S16 tennis retarget release (qpos-only, 60 Hz) to LATENT NPZ clips.

Writes one file per clip with the same keys/dtypes as csv_to_npz.py, computed with the
no-racket model (FLAT_TERRAIN_WO_RACKET_XML) to match the native Tennis data format.

The clips are resampled to 50 Hz (the env's control rate): prepare_trajectory recomputes
reference velocities with frequency=1/ctrl_dt *before* the TrajectoryHandler interpolates,
so a 60 Hz file would get velocities scaled by 50/60. xyz and joints are interpolated
linearly, the root quaternion with slerp. qvel uses mj_differentiatePos (forward
difference, last frame repeats the previous one).

Usage:
    python scripts/process_motion/xsens_npz_to_latent.py \
        --npz /data/drl/humanoid-sports/jaredb/retargeting/xsens-g1-retarget/outputs/release/tennis_s16_g1_v1/tennis_s16_g1_v1.npz \
        --out-dir storage/data/mocap/Tennis/xsens_s16
"""

import argparse
import os

import mujoco
import numpy as np

from latent_mj.envs.g1_tracking import g1_tracking_constants_tennis as consts

DEFAULT_NPZ = ("/data/drl/humanoid-sports/jaredb/retargeting/xsens-g1-retarget/outputs/release/"
               "tennis_s16_g1_v1/tennis_s16_g1_v1.npz")
OUT_FPS = 50.0


def slerp(q0, q1, t):
    """Row-wise slerp between unit wxyz quaternions q0, q1 (N, 4) at fractions t (N,)."""
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, sin_theta))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin_theta))
    q = w0[:, None] * q0 + w1[:, None] * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample(qpos, fps_in, fps_out):
    """Resample (T, 36) qpos [xyz, wxyz, joints] from fps_in to fps_out over the same time span."""
    n_in = qpos.shape[0]
    n_out = int(np.floor((n_in - 1) * fps_out / fps_in + 1e-9)) + 1
    x = np.arange(n_out) * (fps_in / fps_out)          # positions in input-frame units
    i0 = np.minimum(np.floor(x).astype(int), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    t = x - i0
    out = (1.0 - t)[:, None] * qpos[i0] + t[:, None] * qpos[i1]
    out[:, 3:7] = slerp(qpos[i0, 3:7], qpos[i1, 3:7], t)
    return out


def convert_clip(mj_model, mj_data, qpos, fps):
    n_frames = qpos.shape[0]
    nv, nbody, nsite = mj_model.nv, mj_model.nbody, mj_model.nsite
    dt = 1.0 / fps

    qvel = np.zeros((n_frames, nv))
    for i in range(n_frames - 1):
        mujoco.mj_differentiatePos(mj_model, qvel[i], dt, qpos[i], qpos[i + 1])
    if n_frames > 1:
        qvel[-1] = qvel[-2]

    xpos = np.zeros((n_frames, nbody, 3), dtype=np.float32)
    xquat = np.zeros((n_frames, nbody, 4), dtype=np.float32)
    cvel = np.zeros((n_frames, nbody, 6), dtype=np.float32)
    subtree_com = np.zeros((n_frames, nbody, 3), dtype=np.float32)
    site_xpos = np.zeros((n_frames, nsite, 3), dtype=np.float32)
    site_xmat = np.zeros((n_frames, nsite, 9), dtype=np.float32)
    for i in range(n_frames):
        mj_data.qpos[:] = qpos[i]
        mj_data.qvel[:] = qvel[i]
        mujoco.mj_forward(mj_model, mj_data)
        xpos[i] = mj_data.xpos
        xquat[i] = mj_data.xquat
        cvel[i] = mj_data.cvel
        subtree_com[i] = mj_data.subtree_com
        site_xpos[i] = mj_data.site_xpos
        site_xmat[i] = mj_data.site_xmat.reshape(nsite, 9)

    return dict(
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        xpos=xpos, xquat=xquat, cvel=cvel, subtree_com=subtree_com,
        site_xpos=site_xpos, site_xmat=site_xmat,
    )


def model_fields(mj_model):
    """Static model metadata, as written by csv_to_npz.py."""
    return dict(
        joint_names=np.array([mj_model.joint(j).name for j in range(mj_model.njnt)]),
        body_names=np.array([mj_model.body(b).name for b in range(mj_model.nbody)]),
        site_names=np.array([mj_model.site(s).name for s in range(mj_model.nsite)]),
        metadata=None,
        njnt=np.int64(mj_model.njnt),
        jnt_type=mj_model.jnt_type.copy(),
        nbody=np.int64(mj_model.nbody),
        body_rootid=mj_model.body_rootid.copy(),
        body_weldid=mj_model.body_weldid.copy(),
        body_mocapid=mj_model.body_mocapid.copy(),
        body_pos=mj_model.body_pos.copy().astype(np.float32),
        body_quat=mj_model.body_quat.copy().astype(np.float32),
        body_ipos=mj_model.body_ipos.copy().astype(np.float32),
        body_iquat=mj_model.body_iquat.copy().astype(np.float32),
        nsite=np.int64(mj_model.nsite),
        site_bodyid=mj_model.site_bodyid.copy(),
        site_pos=mj_model.site_pos.copy().astype(np.float32),
        site_quat=mj_model.site_quat.copy().astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=DEFAULT_NPZ)
    parser.add_argument("--out-dir", type=str, default="storage/data/mocap/Tennis/xsens_s16")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N clips")
    args = parser.parse_args()

    src = np.load(args.npz)
    fps_in = float(src["fps"])
    qpos_all = src["qpos"].astype(np.float64)
    names = [str(n) for n in src["clip_names"]]
    starts, lengths = src["clip_start"], src["clip_length"]

    mj_model = mujoco.MjModel.from_xml_path(str(consts.FLAT_TERRAIN_WO_RACKET_XML))
    mj_data = mujoco.MjData(mj_model)

    # The release stores [xyz, wxyz, 29 joints in ACTION_JOINT_NAMES order]; check it maps 1:1
    # onto the model's qpos (free joint + 29 hinges in the same order).
    src_joints = [str(n) for n in src["joint_names"][7:]]
    assert src_joints == list(consts.ACTION_JOINT_NAMES), "joint order mismatch"
    assert mj_model.nq == 7 + len(src_joints) and mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    for k, jname in enumerate(src_joints):
        assert mj_model.joint(jname).qposadr[0] == 7 + k, f"{jname} qpos address mismatch"

    static = model_fields(mj_model)
    os.makedirs(args.out_dir, exist_ok=True)

    n_clips = len(names) if args.limit is None else min(args.limit, len(names))
    total_in = total_out = 0
    for c in range(n_clips):
        qpos = qpos_all[starts[c]:starts[c] + lengths[c]].copy()
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
        qpos = resample(qpos, fps_in, OUT_FPS)
        data = convert_clip(mj_model, mj_data, qpos, OUT_FPS)
        n = qpos.shape[0]
        np.savez(
            os.path.join(args.out_dir, f"{names[c]}.npz"),
            **data,
            split_points=np.array([0, n], dtype=np.int32),
            frequency=np.float64(OUT_FPS),
            **static,
        )
        total_in += int(lengths[c])
        total_out += n
        print(f"[{c + 1}/{n_clips}] {names[c]}: {lengths[c]} frames @ {fps_in:g} Hz -> {n} @ {OUT_FPS:g} Hz")

    print(f"Done: {n_clips} clips, {total_in} frames @ {fps_in:g} Hz -> {total_out} frames @ {OUT_FPS:g} Hz, "
          f"{total_out / OUT_FPS:.1f} s ({total_out / OUT_FPS / 60:.2f} min) -> {args.out_dir}")


if __name__ == "__main__":
    main()
