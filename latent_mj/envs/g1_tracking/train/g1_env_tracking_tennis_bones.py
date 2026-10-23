"""G1TrackingTennisBones: tennis tracking task plus a subset of BONES locomotion clips.

Env vars, read at import:
    BONES_SET        folder under storage/data/mocap/Bones with the converted pool (default "loco")
    BONES_NUM_CLIPS  how many pool clips to add; unset = all, 0 = tennis only
Subsets are nested: the pool is shuffled once with a fixed seed and the first N are taken.
"""
import json
import os
import random
from pathlib import Path

import latent_mj as lmj
from latent_mj.envs.g1_tracking.train import g1_env_tracking_tennis as base

_SET = os.environ.get("BONES_SET", "loco")
_MANIFEST = Path(os.getcwd()) / "storage" / "data" / "mocap" / "Bones" / _SET / "manifest.json"


def _select_clips():
    clips = sorted(json.loads(_MANIFEST.read_text())["clips"])
    random.Random(0).shuffle(clips)
    n = os.environ.get("BONES_NUM_CLIPS")
    return clips if n is None else clips[: int(n)]


def g1_tracking_tennis_bones_task_config():
    config = base.g1_tracking_tennis_task_config()
    clips = _select_clips()
    print(f"[bones] adding {len(clips)} clips from Bones/{_SET}")
    if clips:
        config.env_config.reference_traj_config.name[f"Bones/{_SET}"] = clips
    return config


if _MANIFEST.exists():
    lmj.registry.register("G1TrackingTennisBones", "tracking_config")(g1_tracking_tennis_bones_task_config())
    lmj.registry.register("G1TrackingTennisBones", "tracking_train_env_class")(base.G1TrackingTennisEnv)
