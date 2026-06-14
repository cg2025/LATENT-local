"""Convert BONES-SEED G1 CSV files to LATENT NPZ format.

Usage:
    python scripts/process_motion/csv_to_npz.py \
        --csv_path /data/scratch-fast/cgoyal/g1/csv/240327/play_tennis_R_002__A533.csv \
        --output_path storage/data/mocap/Tennis/p1/bones_tennis_001.npz
"""

import os
import argparse
import numpy as np
import mujoco
from tqdm import tqdm

from latent_mj.envs.g1_tracking import g1_tracking_constants_tennis as consts


# CSV columns (degrees) to  MuJoCo joint names
CSV_JOINT_COLUMNS = [
    "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof", "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof", "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof", "right_wrist_yaw_joint_dof",
]
 
MUJOCO_JOINT_NAMES = consts.ACTION_JOINT_NAMES  # 29 joints


def euler_to_quat(rx, ry, rz):
    """Convert XYZ euler angles (degrees) to quaternion (w, x, y, z)."""
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    cx, cy, cz = np.cos(rx/2), np.cos(ry/2), np.cos(rz/2)
    sx, sy, sz = np.sin(rx/2), np.sin(ry/2), np.sin(rz/2)
    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz
    return np.array([w, x, y, z])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--fps", type=float, default=50.0,
                        help="Output FPS (CSV is 120fps, downsample to 50)")
    args = parser.parse_args()

    # Load the base model (no racket) to match native Tennis data format (8 sites).
    mj_model = mujoco.MjModel.from_xml_path(str(consts.FLAT_TERRAIN_WO_RACKET_XML))
    mj_data = mujoco.MjData(mj_model)

    # Get joint qpos addresses
    joint_qpos_adrs = []
    for jname in MUJOCO_JOINT_NAMES:
        adr = mj_model.joint(jname).qposadr
        joint_qpos_adrs.append(adr)

    # Load CSV
    print(f"Loading CSV: {args.csv_path}")
    import pandas as pd
    df = pd.read_csv(args.csv_path)
    print(f"CSV shape: {df.shape}, columns: {list(df.columns[:5])}...")

    # Downsample from 120fps to target fps
    csv_fps = 120.0
    step = int(csv_fps / args.fps)
    df = df.iloc[::step].reset_index(drop=True)
    n_frames = len(df)
    print(f"Downsampled to {n_frames} frames at {args.fps}fps")

    # Get model dimensions
    nq = mj_model.nq  # qpos size
    nv = mj_model.nv  # qvel size
    nbody = mj_model.nbody
    nsite = mj_model.nsite

    # Allocate arrays
    qpos_arr = np.zeros((n_frames, nq), dtype=np.float32)
    qvel_arr = np.zeros((n_frames, nv), dtype=np.float32)
    xpos_arr = np.zeros((n_frames, nbody, 3), dtype=np.float32)
    xquat_arr = np.zeros((n_frames, nbody, 4), dtype=np.float32)
    cvel_arr = np.zeros((n_frames, nbody, 6), dtype=np.float32)
    subtree_com_arr = np.zeros((n_frames, nbody, 3), dtype=np.float32)
    site_xpos_arr = np.zeros((n_frames, nsite, 3), dtype=np.float32)
    site_xmat_arr = np.zeros((n_frames, nsite, 9), dtype=np.float32)

    print("Running forward kinematics...")
    for i, row in enumerate(tqdm(df.itertuples(), total=n_frames)):
        # Root position (cm -> m)
        root_x = row.root_translateX / 100.0
        root_y = row.root_translateY / 100.0
        root_z = row.root_translateZ / 100.0

        # Root rotation (euler degrees to quaternion)
        root_quat = euler_to_quat(row.root_rotateX, row.root_rotateY, row.root_rotateZ)

        # Set qpos: [x, y, z, qw, qx, qy, qz, joint1, joint2, ...]
        mj_data.qpos[0] = root_x
        mj_data.qpos[1] = root_y
        mj_data.qpos[2] = root_z
        mj_data.qpos[3:7] = root_quat  # w, x, y, z

        # Set joint angles (degrees -> radians)
        for j, (col, adr) in enumerate(zip(CSV_JOINT_COLUMNS, joint_qpos_adrs)):
            val = getattr(row, col)
            mj_data.qpos[adr] = np.radians(val)

        # Compute velocities via finite differences (except first frame)
        if i > 0:
            dt = 1.0 / args.fps
            mj_data.qvel[:3] = (mj_data.qpos[:3] - qpos_arr[i-1][:3]) / dt
            mj_data.qvel[3:6] = 0.0
            mj_data.qvel[6:] = (mj_data.qpos[7:] - qpos_arr[i-1][7:]) / dt
            # Handle quaternion velocity differently for root
            mj_data.qvel[3:6] = 0.0  # simplified: zero angular vel

        # Run forward kinematics
        mujoco.mj_forward(mj_model, mj_data)

        # Store results
        qpos_arr[i] = mj_data.qpos.copy()
        qvel_arr[i] = mj_data.qvel.copy()
        xpos_arr[i] = mj_data.xpos.copy()
        xquat_arr[i] = mj_data.xquat.copy()
        cvel_arr[i] = mj_data.cvel.copy()
        subtree_com_arr[i] = mj_data.subtree_com.copy()
        site_xpos_arr[i] = mj_data.site_xpos.copy()
        site_xmat_arr[i] = mj_data.site_xmat.reshape(nsite, 9).copy()

    # Get model metadata
    joint_names = np.array([mj_model.joint(j).name for j in range(mj_model.njnt)])
    body_names = np.array([mj_model.body(b).name for b in range(nbody)])
    site_names = np.array([mj_model.site(s).name for s in range(nsite)])

    # Save NPZ
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"Saving to {args.output_path}")
    np.savez(
        args.output_path,
        qpos=qpos_arr,
        qvel=qvel_arr,
        xpos=xpos_arr,
        xquat=xquat_arr,
        cvel=cvel_arr,
        subtree_com=subtree_com_arr,
        site_xpos=site_xpos_arr,
        site_xmat=site_xmat_arr,
        split_points=np.array([0, n_frames], dtype=np.int32),
        joint_names=joint_names,
        frequency=np.float64(args.fps),
        body_names=body_names,
        site_names=site_names,
        metadata=None,
        njnt=np.int64(mj_model.njnt),
        jnt_type=mj_model.jnt_type.copy(),
        nbody=np.int64(nbody),
        body_rootid=mj_model.body_rootid.copy(),
        body_weldid=mj_model.body_weldid.copy(),
        body_mocapid=mj_model.body_mocapid.copy(),
        body_pos=mj_model.body_pos.copy().astype(np.float32),
        body_quat=mj_model.body_quat.copy().astype(np.float32),
        body_ipos=mj_model.body_ipos.copy().astype(np.float32),
        body_iquat=mj_model.body_iquat.copy().astype(np.float32),
        nsite=np.int64(nsite),
        site_bodyid=mj_model.site_bodyid.copy(),
        site_pos=mj_model.site_pos.copy().astype(np.float32),
        site_quat=mj_model.site_quat.copy().astype(np.float32),
    )
    print("Done!")


if __name__ == "__main__":
    main()
