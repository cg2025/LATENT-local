"""Convert BONES-SEED G1 locomotion clips into LATENT per-clip NPZ files.

Selection, header check and resampling follow preprocess_motions.py (plain-locomotion
filter, cm->m, deg->rad, 120 Hz -> 50 Hz by time-based interpolation, slerp on the root).
Output is the per-clip NPZ schema Trajectory.load / prepare_trajectory read:

    storage/data/mocap/Bones/loco/<date>__<clip>.npz
    storage/data/mocap/Bones/loco/manifest.json

Usage (compute node, repo root):
    uv run python scripts/process_motion/bones_to_latent_npz.py --dry_run
    uv run python scripts/process_motion/bones_to_latent_npz.py --limit 300
"""

import argparse
import json
import re
from itertools import zip_longest
from pathlib import Path

import mujoco
import numpy as np

from latent_mj.envs.g1_tracking import g1_tracking_constants_tennis as consts

SOURCE = Path("/data/scratch/cgoyal/bones_seed/g1/csv")
OUTPUT = Path("storage/data/mocap/Bones/loco")
NATIVE = Path("storage/data/mocap/Tennis/p1/Random_001_Tennis 001.npz")

SRC_FPS, CTL_FPS = 120.0, 50.0
CM_TO_M = 0.01

ROOT_COLUMNS = ["root_translateX", "root_translateY", "root_translateZ",
                "root_rotateX", "root_rotateY", "root_rotateZ"]

LOCO_ROOTS = (
    "walk", "jog", "run", "idle", "turn", "stand", "step", "pace", "march",
    "sprint", "stride", "strafe", "sidestep", "trot",
)

LOCO_MODIFIERS = set("""
forward backward back fwd bwd left right side sideway sideways sidewards
diagonal diag arc arced loop looping start starting stop stopping stopped around
inplace place neutral normal fast faster slow slower slowly quick quickly hurry
hurried very long short straight curve curved curving circle circular zigzag zig
zag change changing direction dir randdir random transition continuous cont mirror
mirrored deg cw ccw clockwise counterclockwise ff fb bf fwdback turning walking
jogging running standing stepping rotate rotating rotation high low mid medium
up down on off into in out to the and a of straightline
""".split())


def is_simple_loco(name: str) -> bool:
    parts = name.split("__")  # <motion>__<actor>[_M]
    toks = [t for t in re.split(r"[^a-z]+", parts[0].lower()) if t]
    if not any(t.startswith(LOCO_ROOTS) for t in toks):
        return False
    return all(t.startswith(LOCO_ROOTS) or t in LOCO_MODIFIERS or len(t) <= 2 for t in toks)


def check_header(path: Path) -> None:
    expected = ["Frame", *ROOT_COLUMNS, *(f"{n}_dof" for n in consts.ACTION_JOINT_NAMES)]
    with path.open() as f:
        header = f.readline().strip().split(",")
    if header != expected:
        bad = [(i, a, b) for i, (a, b) in enumerate(zip_longest(header, expected)) if a != b]
        raise SystemExit(f"{path}: {len(bad)} columns differ, first is index {bad[0][0]} "
                         f"{bad[0][1]!r}, expected {bad[0][2]!r}")


def euler_to_quat(deg):
    """Maya XYZ Euler angles to wxyz quaternions: q = qz (x) qy (x) qx."""
    x, y, z = (np.deg2rad(deg) / 2.0).T
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return np.stack([cz * cy * cx + sz * sy * sx,
                     cz * cy * sx - sz * sy * cx,
                     cz * sy * cx + sz * cy * sx,
                     sz * cy * cx - cz * sy * sx], axis=-1)


def unwrap(quat):
    sign = np.cumprod(np.where((quat[1:] * quat[:-1]).sum(1) < 0, -1.0, 1.0))
    return np.concatenate([quat[:1], quat[1:] * sign[:, None]])


def interp(cols, t_src, t_dst):
    return np.column_stack([np.interp(t_dst, t_src, col) for col in cols.T])


def slerp(quat, t_src, t_dst):
    hi = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    lo = hi - 1
    a = ((t_dst - t_src[lo]) / (t_src[hi] - t_src[lo]))[:, None]
    q0, q1 = quat[lo], quat[hi]
    theta = np.arccos(np.clip((q0 * q1).sum(1, keepdims=True), -1.0, 1.0))
    q = np.where(theta < 1e-6, (1 - a) * q0 + a * q1,
                 np.sin((1 - a) * theta) * q0 + np.sin(a * theta) * q1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def resample(path: Path) -> np.ndarray:
    """CSV -> 50 Hz track laid out [xyz m, wxyz, 29 joints rad in ACTION_JOINT_NAMES order]."""
    raw = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    t_src = np.arange(len(raw)) / SRC_FPS
    t_dst = np.arange(0.0, t_src[-1], 1.0 / CTL_FPS)
    if len(t_dst) < 2:
        return np.zeros((0, 36))
    return np.column_stack([
        interp(raw[:, 1:4] * CM_TO_M, t_src, t_dst),
        slerp(unwrap(euler_to_quat(raw[:, 4:7])), t_src, t_dst),
        interp(np.deg2rad(raw[:, 7:]), t_src, t_dst),
    ])


def to_model_qpos(track, adrs, nq):
    qpos = np.zeros((len(track), nq))
    qpos[:, :7] = track[:, :7]
    qpos[:, adrs] = track[:, 7:]  # joints placed by name, not by position
    return qpos


def fk_clip(model, data, qpos, dt):
    n = len(qpos)
    qvel = np.zeros((n, model.nv))
    for k in range(n - 1):
        mujoco.mj_differentiatePos(model, qvel[k], dt, qpos[k], qpos[k + 1])
    qvel[-1] = qvel[-2]
    out = {"qpos": qpos, "qvel": qvel,
           "xpos": np.zeros((n, model.nbody, 3)), "xquat": np.zeros((n, model.nbody, 4)),
           "cvel": np.zeros((n, model.nbody, 6)), "subtree_com": np.zeros((n, model.nbody, 3)),
           "site_xpos": np.zeros((n, model.nsite, 3)), "site_xmat": np.zeros((n, model.nsite, 9))}
    for k in range(n):
        data.qpos[:], data.qvel[:] = qpos[k], qvel[k]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        out["xpos"][k] = data.xpos
        out["xquat"][k] = data.xquat
        out["cvel"][k] = data.cvel
        out["subtree_com"][k] = data.subtree_com
        out["site_xpos"][k] = data.site_xpos
        out["site_xmat"][k] = data.site_xmat.reshape(model.nsite, 9)
    return {k: v.astype(np.float32) for k, v in out.items()}


def model_meta(m):
    return dict(
        joint_names=np.array([m.joint(j).name for j in range(m.njnt)]),
        body_names=np.array([m.body(b).name for b in range(m.nbody)]),
        site_names=np.array([m.site(s).name for s in range(m.nsite)]),
        njnt=np.int64(m.njnt), jnt_type=m.jnt_type.copy(),
        nbody=np.int64(m.nbody), body_rootid=m.body_rootid.copy(),
        body_weldid=m.body_weldid.copy(), body_mocapid=m.body_mocapid.copy(),
        body_pos=m.body_pos.astype(np.float32), body_quat=m.body_quat.astype(np.float32),
        body_ipos=m.body_ipos.astype(np.float32), body_iquat=m.body_iquat.astype(np.float32),
        nsite=np.int64(m.nsite), site_bodyid=m.site_bodyid.copy(),
        site_pos=m.site_pos.astype(np.float32), site_quat=m.site_quat.astype(np.float32),
    )


def check_native(meta, sample, native_path):
    """Refuse to write anything that would not concatenate with the native tennis clips."""
    native = np.load(native_path, allow_pickle=True)
    bad = []
    for k, v in meta.items():
        if k not in native.files:
            bad.append(f"{k}: missing in native file")
            continue
        a, b = np.asarray(native[k]), np.asarray(v)
        same = a.shape == b.shape and (np.allclose(a, b, atol=1e-5) if a.dtype.kind == "f"
                                       else np.array_equal(a, b))
        if not same:
            bad.append(f"{k}: differs from native")
    for k, v in sample.items():
        if native[k].shape[1:] != v.shape[1:]:
            bad.append(f"{k}: per-frame shape {v.shape[1:]} vs native {native[k].shape[1:]}")
    if float(native["frequency"]) != CTL_FPS:
        bad.append(f"native frequency is {float(native['frequency'])}, not {CTL_FPS}")
    if bad:
        raise SystemExit("Output would NOT match the native Tennis data:\n  " + "\n  ".join(bad))
    print("matches native Tennis clip format")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--native", type=Path, default=NATIVE)
    ap.add_argument("--pattern", default="**/*.csv")
    ap.add_argument("--all", action="store_true", help="keep every clip, not just plain locomotion")
    ap.add_argument("--limit", type=int, help="evenly spaced sample of this many clips")
    ap.add_argument("--min_frames", type=int, default=150, help="skip clips shorter than this at 50 Hz")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    matched = sorted(args.source.glob(args.pattern))
    if not matched:
        raise SystemExit(f"no clips matched {args.pattern} under {args.source}")
    paths = matched if args.all else [p for p in matched if is_simple_loco(p.stem)]
    if not paths:
        raise SystemExit(f"{len(matched)} clips matched but none survived the loco filter")
    print(f"{len(paths)}/{len(matched)} clips selected")
    if args.limit and args.limit < len(paths):
        idx = np.unique(np.linspace(0, len(paths) - 1, args.limit).astype(int))
        paths = [paths[i] for i in idx]
    print(f"using {len(paths)} clips; first few: {[p.stem for p in paths[:5]]}")
    check_header(paths[0])
    if args.dry_run:
        return

    model = mujoco.MjModel.from_xml_path(str(consts.FLAT_TERRAIN_WO_RACKET_XML))
    data = mujoco.MjData(model)
    names = list(consts.ACTION_JOINT_NAMES)
    adrs = np.array([np.asarray(model.joint(n).qposadr).item() for n in names])
    assert model.nq == 7 + len(names), f"nq={model.nq}, expected {7 + len(names)}"
    jids = [np.asarray(model.joint(n).id).item() for n in names]
    lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
    meta = model_meta(model)

    args.output.mkdir(parents=True, exist_ok=True)
    kept, total, checked = [], 0, False
    z_min, z_max, in_range, n_joint_vals, vel_samples = 1e9, -1e9, 0, 0, []
    for i, p in enumerate(paths, 1):
        track = resample(p)
        if len(track) < args.min_frames:
            continue
        arrays = fk_clip(model, data, to_model_qpos(track, adrs, model.nq), 1.0 / CTL_FPS)
        if not checked:
            check_native(meta, arrays, args.native)
            checked = True
        stem = f"{p.parent.name}__{p.stem}"
        n = len(arrays["qpos"])
        np.savez(args.output / f"{stem}.npz", **arrays,
                 split_points=np.array([0, n], dtype=np.int32),
                 frequency=np.float64(CTL_FPS), metadata=None, **meta)
        kept.append(stem)
        total += n
        j = arrays["qpos"][:, adrs]
        z_min, z_max = min(z_min, arrays["qpos"][:, 2].min()), max(z_max, arrays["qpos"][:, 2].max())
        in_range += ((j >= lo) & (j <= hi)).sum()
        n_joint_vals += j.size
        vel_samples.append(np.abs(arrays["qvel"][::10, 6:]).ravel())
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}", end="\r", flush=True)

    (args.output / "manifest.json").write_text(json.dumps(
        {"source": str(args.source), "fps": CTL_FPS, "frames": int(total), "clips": kept}, indent=1))
    vel = np.concatenate(vel_samples)
    print(f"\n{len(kept)} clips, {total} frames ({total / CTL_FPS / 60:.1f} min) -> {args.output}")
    print(f"pelvis height {z_min:.3f} .. {z_max:.3f} m   joints in range {100 * in_range / n_joint_vals:.2f}%")
    print(f"|joint vel| p99 {np.percentile(vel, 99):.1f}  max {vel.max():.1f} rad/s")


if __name__ == "__main__":
    main()
