# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for G1."""

from pathlib import Path
import numpy as np
from latent_mj.constant import PATH_STORAGE, PATH_ASSET, WANDB_PATH_LOG

ROOT_PATH = PATH_ASSET / "unitree_g1"

FLAT_TERRAIN_WO_RACKET_XML = ROOT_PATH / "scene_mjx_flat_terrain.xml"
FLAT_TERRAIN_W_RACKET_XML = ROOT_PATH / "scene_mjx_racket_wo_ball_flat_terrain.xml"
FLAT_TERRAIN_W_RACKET_W_BALL_XML = ROOT_PATH / "scene_mjx_racket_w_ball_flat_terrain.xml"

NUM_JOINT = 29

def task_to_xml(with_racket: bool = False, with_ball: bool = False) -> Path:
    if with_ball and not with_racket:
        raise ValueError("with_ball=True requires with_racket=True (no racket-less ball scene exists)")
    xml_map = {
        (False, False): FLAT_TERRAIN_WO_RACKET_XML,
        (True, False): FLAT_TERRAIN_W_RACKET_XML,
        (True, True): FLAT_TERRAIN_W_RACKET_W_BALL_XML,
    }
    return xml_map[(with_racket, with_ball)]


# Ball body/geom/site names in FLAT_TERRAIN_W_RACKET_W_BALL_XML (see
# latent_mj/learning/policy/high_level -- LAB -- and envs/g1_tracking/train/
# ball_launch.py, which this task's env composes together).
TENNIS_BALL_JOINT = "tennis_ball_freejoint"
TENNIS_BALL_GEOM = "tennis_ball_geom"
TENNIS_BALL_SITE = "tennis_ball_site"
TENNIS_RACKET_COLLISION_GEOM = "tennis_racket_collision"
TENNIS_RACKET_CENTER_SITE = "tennis_racket_center"
NET_GEOM = "net"


FEET_SITES = [
    "left_foot",
    "right_foot",
]

LEFT_FEET_SITE = "left_foot"

RIGHT_FEET_SITE = "right_foot"

FEET_ALL_SITES = [
    "left_foot",
    "right_foot",
    "left_foot_top",
    "right_foot_top",
]

LEFT_FEET_ALL_SITES = [
    "left_foot",
    "left_foot_top",
]

RIGHT_FEET_ALL_SITES = [
    "right_foot",
    "right_foot_top",
]

HAND_SITES = [
    "left_palm",
    "right_palm",
]

LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "torso_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Right leg.
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    # Waist.
    (-2.618, 2.618),
    (-0.52, 0.52),
    (-0.52, 0.52),
    # Left shoulder.
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-1.61443, 1.61443),
    (-1.61443, 1.61443),
    # Right shoulder.
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-1.61443, 1.61443),
    (-1.61443, 1.61443),
)

DOF_VEL_LIMITS = [
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 37.0, 37.0,
    37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0,
    37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0
]

ACTION_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # -------------- tracking only --------------
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

OBS_JOINT_NAMES = [
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

# fmt: off
TORQUE_LIMIT = np.array([
    88., 139., 88., 139., 50., 50.,
    88., 139., 88., 139., 50., 50.,
    88., 50., 50.,
    25., 25., 25., 25., 25., 5., 5.,
    25., 25., 25., 25., 25., 5., 5.,
])

DEFAULT_QPOS = np.float32([
    0, 0, 0.79,
    1, 0, 0, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    0, 0, 0,
    0.2, 0.3, 0, 1.28, 0, 0, 0,
    0.2, -0.3, 0, 1.28, 0, 0, 0,
])

# v2
KPs = np.float32([
    100, 100, 100, 200, 80, 20,
    100, 100, 100, 200, 80, 20,
    300, 300, 300,
    90, 60, 20, 60, 20, 20, 20,
    90, 60, 20, 60, 20, 20, 20,
])

KDs = np.float32([
    2, 2, 2, 4, 2, 1,
    2, 2, 2, 4, 2, 1,
    10, 10, 10,
    2, 2, 1, 1, 1, 1, 1,
    2, 2, 1, 1, 1, 1, 1,
])
# fmt: on

UPPER_BODY_LINKs = [
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

LOWER_BODY_LINKs = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link"
]

UPPER_BODY_JOINTs = [
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

KEY_BODY_LINKs = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link"
]

FEET_LINKs = [
    "left_ankle_roll_link",
    "right_ankle_roll_link"
]

SHOULDER_LINKs = [
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link"
]

# Joints excluded from action output (will directly track reference motion)
# These joints and their corresponding bodies are excluded from reward calculation
EXCLUDED_ACTION_JOINTs = [
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

EXCLUDED_TRACKING_LINKs = [
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link"
]

TENNIS_P1_DATASET_ALL = [
    "Random_001_Tennis 001",
    "Random_002_Tennis 001",
    "Random_003_Tennis 001",
    "Random_004_Tennis 001"
]

# Xsens S16 tennis retarget (tennis_s16_g1_v1), converted by scripts/process_motion/xsens_npz_to_latent.py
TENNIS_XSENS_S16_DATASET_ALL = [
    "12-17-27_19211_21083",
    "12-17-27_21263_23585",
    "12-17-27_23787_28582",
    "12-17-27_29342_29895",
    "12-17-27_30039_30873",
    "12-17-27_31440_32123",
    "12-17-27_33366_35530",
    "12-17-27_36552_40446",
    "12-17-27_40975_41456",
    "12-17-27_43933_45592",
    "12-17-27_46272_47193",
    "12-17-27_47705_49486",
    "12-17-27_55931_56701",
    "12-17-27_56821_57370",
    "12-17-27_59703_60406",
    "12-17-27_62889_63592",
    "12-17-27_64119_64560",
    "12-17-27_65134_65546",
    "12-17-27_66116_66827",
    "12-17-27_68891_70673",
    "12-17-27_74576_77005",
    "12-17-27_77909_79444",
    "12-17-27_79979_81200",
    "12-17-27_81893_82512",
    "12-17-27_83152_84256",
    "12-17-27_84618_85713",
    "12-51-31_5538_6322",
    "12-51-31_6700_7964",
    "12-51-31_8572_10712",
    "12-51-31_11127_14278",
    "12-51-31_14832_17532",
    "12-51-31_20694_22558",
    "12-51-31_22854_24932",
    "12-51-31_25320_28163",
    "12-51-31_29402_30002",
    "12-51-31_30488_31490",
    "12-51-31_31766_32246",
    "12-51-31_32598_37858",
    "12-51-31_38454_39932",
    "12-51-31_42324_46139",
    "12-51-31_47796_48376",
    "12-51-31_48864_49634",
    "12-51-31_49816_52690",
    "12-51-31_55490_58408",
    "12-51-31_59192_59590",
    "12-51-31_60198_62288",
    "12-51-31_63054_63706",
    "12-51-31_64436_66042",
    "12-51-31_66886_69596",
    "12-51-31_72536_73336",
    "12-51-31_74686_75650",
    "12-51-31_76135_76402",
    "12-51-31_77222_77602",
    "12-51-31_78112_81088",
    "12-51-31_85594_85910",
    "12-51-31_86304_86692",
    "12-51-31_87446_89096",
    "12-51-31_89163_89612",
    "12-51-31_90268_91472",
    "12-51-31_91950_92616",
    "12-51-31_93842_94122",
    "12-51-31_94626_94948",
    "12-51-31_95040_95408",
    "12-51-31_95822_96920",
    "12-51-31_97306_98718",
    "12-51-31_101056_101850",
    "12-51-31_102308_103536",
    "12-51-31_106948_107512",
    "12-51-31_109074_110662",
    "12-51-31_111496_112232",
    "12-51-31_112674_113390",
    "12-51-31_113936_115205",
    "12-51-31_116336_116878",
    "12-51-31_117976_120066",
    "12-51-31_120590_121090",
    "12-51-31_121502_123042",
    "12-51-31_123678_125578",
    "12-51-31_126772_127175",
    "12-51-31_135262_135988",
    "12-51-31_136976_138978",
    "12-51-31_139710_140292",
    "12-51-31_140872_142060",
]
