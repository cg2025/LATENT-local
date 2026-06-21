"""Collect BONES-SEED clips into concentric dataset subsets 

  Dataset 0 — "latent_only":
      The 4 original LATENT tennis clips already in storage/data/mocap/Tennis/p1/

  Dataset 1 — "tennis_extended":
      + BONES tennis clips (play_tennis) and similar sport upper-body motions

  Dataset 2 — "tennis_plus_locomotion":
      + Basic walking, running, turning from BONES locomotion category

  Dataset 3 — "full":
      + Advanced locomotion, jumping, sports, and dynamic whole-body motions

Usage:
    # List all datasets and their clips (dry run)
    python scripts/process_motion/collect_datasets.py --dry_run

    # Download and convert a specific dataset
    python scripts/process_motion/collect_datasets.py --dataset tennis_extended

    # Download and convert all datasets
    python scripts/process_motion/collect_datasets.py --dataset all

    # Just show what BONES clips would be selected without downloading
    python scripts/process_motion/collect_datasets.py --dataset full --dry_run
"""

import os
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Dataset definitions
# Each entry is (bones_filename, description)
# ─────────────────────────────────────────────────────────────────────────────

# Dataset 1 additions: tennis + relevant sport/upper-body motions
TENNIS_EXTENDED_CLIPS = [
    # Tennis
    ("play_tennis_R_002__A533",     "playing ground tennis"),
    ("play_tennis_R_002__A533_M",   "playing ground tennis mirrored"),
]

# Dataset 2 additions: basic locomotion skills
LOCOMOTION_CLIPS_KEYWORDS = [
    "walk",
    "run",
    "jog",
    "turn",
    "stand",
    "step",
]

# Dataset 3 additions: advanced and dynamic motions
ADVANCED_CLIPS_KEYWORDS = [
    "jump",
    "sprint",
    "dodge",
    "lunge",
    "squat",
    "kick",
    "swing",
    "reach",
    "crouch",
]

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "latent_only": {
        "description": "Original 4 LATENT tennis clips only",
        "latent_clips": [
            "Random_001_Tennis 001",
            "Random_002_Tennis 001",
            "Random_003_Tennis 001",
            "Random_004_Tennis 001",
        ],
        "bones_clips": [],
        "bones_keywords": [],
        "bones_categories": [],
        "bones_max_clips": 0,
    },
    "tennis_extended": {
        "description": "LATENT clips + BONES tennis and sport clips",
        "latent_clips": [
            "Random_001_Tennis 001",
            "Random_002_Tennis 001",
            "Random_003_Tennis 001",
            "Random_004_Tennis 001",
        ],
        "bones_clips": [
            "play_tennis_R_002__A533",
            "play_tennis_R_002__A533_M",
        ],
        "bones_keywords": ["tennis", "racket", "swing", "serve"],
        "bones_categories": ["Sports"],
        "bones_max_clips": 20,
    },
    "tennis_plus_locomotion": {
        "description": "tennis_extended + basic walking/running/turning",
        "latent_clips": [
            "Random_001_Tennis 001",
            "Random_002_Tennis 001",
            "Random_003_Tennis 001",
            "Random_004_Tennis 001",
        ],
        "bones_clips": [
            "play_tennis_R_002__A533",
            "play_tennis_R_002__A533_M",
        ],
        "bones_keywords": ["tennis", "racket", "walk", "run", "jog", "turn", "stand"],
        "bones_categories": ["Sports", "Basic Locomotion Neutral", "Basic Locomotion Styles"],
        "bones_max_clips": 100,
    },
    "full": {
        "description": "All above + advanced locomotion and dynamic motions",
        "latent_clips": [
            "Random_001_Tennis 001",
            "Random_002_Tennis 001",
            "Random_003_Tennis 001",
            "Random_004_Tennis 001",
        ],
        "bones_clips": [
            "play_tennis_R_002__A533",
            "play_tennis_R_002__A533_M",
        ],
        "bones_keywords": [
            "tennis", "racket", "walk", "run", "jog", "turn", "stand",
            "jump", "sprint", "dodge", "lunge", "squat", "kick", "swing",
            "reach", "crouch", "climb", "balance",
        ],
        "bones_categories": [
            "Sports", "Basic Locomotion Neutral", "Basic Locomotion Styles",
            "Advanced Locomotion", "Unusual Locomotion",
        ],
        "bones_max_clips": 500,
    },
}


def load_bones_metadata(hf_token: Optional[str] = None):
    """Download and load BONES-SEED metadata"""
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        print("Downloading BONES metadata...")
        path = hf_hub_download(
            repo_id="bones-studio/seed",
            filename="metadata/seed_metadata_v004.parquet",
            repo_type="dataset",
            token=hf_token,
        )
        df = pd.read_parquet(path)
        print(f"Loaded {len(df)} motion clips from BONES metadata.")
        return df
    except Exception as e:
        print(f"Warning: Could not load BONES metadata: {e}")
        return None


def select_bones_clips(df, dataset_cfg: dict) -> list:
    """Select BONES clips matching the dataset config."""
    if df is None:
        return dataset_cfg["bones_clips"]

    selected = set(dataset_cfg["bones_clips"])

    keywords = dataset_cfg.get("bones_keywords", [])
    categories = dataset_cfg.get("bones_categories", [])
    max_clips = dataset_cfg.get("bones_max_clips", 0)

    if max_clips == 0:
        return list(selected)

    # Filter by category
    if categories:
        cat_mask = df["category"].isin(categories)
        cat_df = df[cat_mask]
    else:
        cat_df = df

    # Filter by keywords in content descriptions
    if keywords:
        kw_mask = False
        for kw in keywords:
            kw_mask = kw_mask | (
                cat_df["content_name"].str.contains(kw, case=False, na=False) |
                cat_df["content_short_description"].str.contains(kw, case=False, na=False)
            )
        cat_df = cat_df[kw_mask]

    # Exclude mirrors to avoid duplicates (will add them back selectively)
    non_mirror = cat_df[~cat_df["is_mirror"]]

    # Sort by duration (prefer longer clips =  more motion variety)
    non_mirror = non_mirror.sort_values("move_duration_frames", ascending=False)

    # Take up to max_clips
    clips_to_add = non_mirror["filename"].tolist()
    for clip in clips_to_add:
        if len(selected) >= max_clips:
            break
        selected.add(clip)

    print(f"  Selected {len(selected)} clips total ({len(selected) - len(dataset_cfg['bones_clips'])} from BONES search)")
    return sorted(list(selected))


def download_and_convert_clip(filename: str, output_dir: Path, hf_token: Optional[str], date: str = "240327"):
    """Download a BONES G1 CSV and convert to NPZ."""
    from huggingface_hub import hf_hub_download

    csv_rel_path = f"g1/csv/{date}/{filename}.csv"
    output_path = output_dir / f"{filename}.npz"

    if output_path.exists():
        print(f"  [skip] {filename}.npz already exists")
        return True

    # Try to download individual file
    try:
        csv_path = hf_hub_download(
            repo_id="bones-studio/seed",
            filename=csv_rel_path,
            repo_type="dataset",
            token=hf_token,
        )
        print(f"  [download] {filename}.csv")
    except Exception as e:
        print(f"  [error] Could not download {filename}: {e}")
        print(f"  Tip: Extract from g1.tar.gz: tar -xzf g1.tar.gz {csv_rel_path}")
        return False

    # Convert CSV to NPZ
    convert_csv_to_npz(csv_path, str(output_path))
    return True


def convert_csv_to_npz(csv_path: str, output_path: str, fps: float = 50.0):
    """Convert a BONES G1 CSV to LATENT NPZ format."""
    import mujoco
    import pandas as pd
    import latent_mj as lmj
    from latent_mj.envs.g1_tracking import g1_tracking_constants_tennis as consts

    CSV_JOINT_COLUMNS = [
        "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
        "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
        "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
        "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
        "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
        "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof", "left_shoulder_yaw_joint_dof",
        "left_elbow_joint_dof", "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof",
        "left_wrist_yaw_joint_dof", "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof", "right_wrist_roll_joint_dof",
        "right_wrist_pitch_joint_dof", "right_wrist_yaw_joint_dof",
    ]

    def euler_to_quat(rx, ry, rz):
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
        cx, cy, cz = np.cos(rx/2), np.cos(ry/2), np.cos(rz/2)
        sx, sy, sz = np.sin(rx/2), np.sin(ry/2), np.sin(rz/2)
        w = cx*cy*cz + sx*sy*sz
        x = sx*cy*cz - cx*sy*sz
        y = cx*sy*cz + sx*cy*sz
        z = cx*cy*sz - sx*sy*cz
        return np.array([w, x, y, z])

    task_cfg = lmj.registry.get("G1TrackingTennis", "tracking_config")
    env_cfg = task_cfg.env_config
    EnvClass = lmj.registry.get("G1TrackingTennis", "tracking_train_env_class")
    env = EnvClass(config=env_cfg)
    mj_model = env._mj_model
    mj_data = mujoco.MjData(mj_model)

    joint_qpos_adrs = [mj_model.joint(jname).qposadr for jname in consts.ACTION_JOINT_NAMES]

    df = pd.read_csv(csv_path)
    step = int(120.0 / fps)
    df = df.iloc[::step].reset_index(drop=True)
    n_frames = len(df)

    nq, nv, nbody, nsite = mj_model.nq, mj_model.nv, mj_model.nbody, mj_model.nsite
    qpos_arr = np.zeros((n_frames, nq), dtype=np.float32)
    qvel_arr = np.zeros((n_frames, nv), dtype=np.float32)
    xpos_arr = np.zeros((n_frames, nbody, 3), dtype=np.float32)
    xquat_arr = np.zeros((n_frames, nbody, 4), dtype=np.float32)
    cvel_arr = np.zeros((n_frames, nbody, 6), dtype=np.float32)
    subtree_com_arr = np.zeros((n_frames, nbody, 3), dtype=np.float32)
    site_xpos_arr = np.zeros((n_frames, nsite, 3), dtype=np.float32)
    site_xmat_arr = np.zeros((n_frames, nsite, 9), dtype=np.float32)

    for i, row in enumerate(df.itertuples()):
        mj_data.qpos[0] = row.root_translateX / 100.0
        mj_data.qpos[1] = row.root_translateY / 100.0
        mj_data.qpos[2] = row.root_translateZ / 100.0
        mj_data.qpos[3:7] = euler_to_quat(row.root_rotateX, row.root_rotateY, row.root_rotateZ)
        for col, adr in zip(CSV_JOINT_COLUMNS, joint_qpos_adrs):
            mj_data.qpos[adr] = np.radians(getattr(row, col))
        if i > 0:
            dt = 1.0 / fps
            mj_data.qvel[:3] = (mj_data.qpos[:3] - qpos_arr[i-1][:3]) / dt
            mj_data.qvel[3:6] = 0.0
            mj_data.qvel[6:] = (mj_data.qpos[7:] - qpos_arr[i-1][7:]) / dt
        mujoco.mj_forward(mj_model, mj_data)
        qpos_arr[i] = mj_data.qpos.copy()
        qvel_arr[i] = mj_data.qvel.copy()
        xpos_arr[i] = mj_data.xpos.copy()
        xquat_arr[i] = mj_data.xquat.copy()
        cvel_arr[i] = mj_data.cvel.copy()
        subtree_com_arr[i] = mj_data.subtree_com.copy()
        site_xpos_arr[i] = mj_data.site_xpos.copy()
        site_xmat_arr[i] = mj_data.site_xmat.reshape(nsite, 9).copy()

    joint_names = np.array([mj_model.joint(j).name for j in range(mj_model.njnt)])
    body_names = np.array([mj_model.body(b).name for b in range(nbody)])
    site_names = np.array([mj_model.site(s).name for s in range(nsite)])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        qpos=qpos_arr, qvel=qvel_arr, xpos=xpos_arr, xquat=xquat_arr,
        cvel=cvel_arr, subtree_com=subtree_com_arr,
        site_xpos=site_xpos_arr, site_xmat=site_xmat_arr,
        split_points=np.array([0, n_frames], dtype=np.int32),
        joint_names=joint_names, frequency=np.float64(fps),
        body_names=body_names, site_names=site_names,
        metadata=np.array({"source": csv_path}, dtype=object),
        njnt=np.int64(mj_model.njnt), jnt_type=mj_model.jnt_type.copy(),
        nbody=np.int64(nbody), body_rootid=mj_model.body_rootid.copy(),
        body_weldid=mj_model.body_weldid.copy(), body_mocapid=mj_model.body_mocapid.copy(),
        body_pos=mj_model.body_pos.copy().astype(np.float32),
        body_quat=mj_model.body_quat.copy().astype(np.float32),
        body_ipos=mj_model.body_ipos.copy().astype(np.float32),
        body_iquat=mj_model.body_iquat.copy().astype(np.float32),
        nsite=np.int64(nsite), site_bodyid=mj_model.site_bodyid.copy(),
        site_pos=mj_model.site_pos.copy().astype(np.float32),
        site_quat=mj_model.site_quat.copy().astype(np.float32),
    )
    print(f"  [saved] {output_path} ({n_frames} frames)")


def save_dataset_manifest(dataset_name: str, cfg: dict, bones_clips: list, output_dir: Path):
    """Save a JSON manifest listing all clips in this dataset."""
    manifest = {
        "dataset_name": dataset_name,
        "description": cfg["description"],
        "latent_clips": cfg["latent_clips"],
        "bones_clips": bones_clips,
        "total_clips": len(cfg["latent_clips"]) + len(bones_clips),
        "mocap_dir": str(output_dir),
    }
    manifest_path = output_dir / f"manifest_{dataset_name}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [manifest] saved to {manifest_path}")
    return manifest


def print_dataset_summary(dataset_name: str, cfg: dict, bones_clips: list):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Description: {cfg['description']}")
    print(f"LATENT clips ({len(cfg['latent_clips'])}):")
    for c in cfg["latent_clips"]:
        print(f"  - {c}")
    print(f"BONES clips ({len(bones_clips)}):")
    for c in bones_clips:
        print(f"  - {c}")
    print(f"Total: {len(cfg['latent_clips']) + len(bones_clips)} clips")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Collect BONES clips into dataset subsets")
    parser.add_argument("--dataset", type=str, default="tennis_extended",
                        choices=list(DATASETS.keys()) + ["all"],
                        help="Which dataset to build")
    parser.add_argument("--output_dir", type=str,
                        default="storage/data/mocap",
                        help="Base output directory for NPZ files")
    parser.add_argument("--bones_tar", type=str, default=None,
                        help="Path to g1.tar.gz if already downloaded (faster than HF download)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for downloading BONES data")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just show what would be done without downloading")
    parser.add_argument("--date", type=str, default="240327",
                        help="BONES capture date folder")
    args = parser.parse_args()

    datasets_to_build = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    output_dir = Path(args.output_dir)

    # Load BONES metadata for clip selection
    df = None
    if not args.dry_run:
        df = load_bones_metadata(args.hf_token)

    for dataset_name in datasets_to_build:
        cfg = DATASETS[dataset_name]

        # Select BONES clips
        bones_clips = select_bones_clips(df, cfg)

        print_dataset_summary(dataset_name, cfg, bones_clips)

        if args.dry_run:
            continue

        # Save manifest
        dataset_dir = output_dir / "Tennis" / "p1"
        manifest = save_dataset_manifest(dataset_name, cfg, bones_clips, dataset_dir)

        # Convert BONES clips to NPZ
        if bones_clips:
            print(f"\nConverting {len(bones_clips)} BONES clips...")
            for filename in bones_clips:
                npz_path = dataset_dir / f"{filename}.npz"
                if npz_path.exists():
                    print(f"  [skip] {filename}.npz already exists")
                    continue

                # Try to find CSV in tar or download
                csv_found = False

                # Check if extracted already
                extracted_csv = Path(f"/data/scratch-fast/cgoyal/g1/csv/{args.date}/{filename}.csv")
                if extracted_csv.exists():
                    print(f"  [convert] {filename}")
                    convert_csv_to_npz(str(extracted_csv), str(npz_path))
                    csv_found = True

                if not csv_found and args.hf_token:
                    download_and_convert_clip(filename, dataset_dir, args.hf_token, args.date)

                if not csv_found and not args.hf_token:
                    print(f"  [skip] {filename}: no CSV found. Extract from g1.tar.gz or provide --hf_token")

        print(f"\nDataset '{dataset_name}' ready!")
        print(f"To train with this dataset, update reference_traj_config.name in the env config")
        print(f"or pass the manifest to your training script.")

    print("\nAll done!")
    print("\nDataset sizes:")
    for name, cfg in DATASETS.items():
        print(f"  {name}: {len(cfg['latent_clips'])} LATENT + ~{cfg['bones_max_clips']} BONES clips")


if __name__ == "__main__":
    main()
