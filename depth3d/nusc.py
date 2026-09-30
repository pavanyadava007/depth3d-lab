"""Thin helpers around the nuScenes devkit: keyframes, calibration, poses.

All 4x4 matrices map homogeneous points from the frame in the name's first part into the
frame in the second part, for example ``T_cam_to_world @ p_cam = p_world``.
Camera convention is OpenCV (x right, y down, z forward), as in nuScenes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

DEFAULT_DATA_ROOT = "/home/ec2-user/workspace/offroad-bevfusion/data/nuscenes"
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CAMERA_GROUP = {
    "CAM_FRONT": "front",
    "CAM_FRONT_LEFT": "side",
    "CAM_FRONT_RIGHT": "side",
    "CAM_BACK_LEFT": "side",
    "CAM_BACK_RIGHT": "side",
    "CAM_BACK": "back",
}


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT))


def data_available() -> bool:
    return (data_root() / "v1.0-mini" / "sample.json").exists()


@lru_cache(maxsize=1)
def load_nusc():
    from nuscenes.nuscenes import NuScenes

    return NuScenes(version="v1.0-mini", dataroot=str(data_root()), verbose=False)


def pose_matrix(record: dict) -> np.ndarray:
    """4x4 from a nuScenes record with 'rotation' (w, x, y, z) and 'translation'."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    T[:3, 3] = np.asarray(record["translation"], dtype=np.float64)
    return T


@dataclass
class CamFrame:
    sample_token: str
    scene_name: str
    channel: str
    sd_token: str
    filename: str
    width: int
    height: int
    K: np.ndarray
    T_cam_to_world: np.ndarray
    timestamp: int

    @property
    def path(self) -> Path:
        return data_root() / self.filename

    @property
    def T_world_to_cam(self) -> np.ndarray:
        return np.linalg.inv(self.T_cam_to_world)


def sensor_to_world(nusc, sd_token: str) -> np.ndarray:
    """T_sensor_to_world for a sample_data token: sensor -> ego (at that ts) -> world."""
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return pose_matrix(ep) @ pose_matrix(cs)


def cam_frame(nusc, sample_token: str, channel: str) -> CamFrame:
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"][channel]
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    scene = nusc.get("scene", sample["scene_token"])
    return CamFrame(
        sample_token=sample_token,
        scene_name=scene["name"],
        channel=channel,
        sd_token=sd_token,
        filename=sd["filename"],
        width=sd["width"],
        height=sd["height"],
        K=np.asarray(cs["camera_intrinsic"], dtype=np.float64),
        T_cam_to_world=sensor_to_world(nusc, sd_token),
        timestamp=sd["timestamp"],
    )


def lidar_to_cam_matrix(nusc, sample_token: str, channel: str) -> np.ndarray:
    """T_lidar_to_cam: lidar -> ego(lidar ts) -> world -> ego(cam ts) -> cam."""
    sample = nusc.get("sample", sample_token)
    T_lidar_to_world = sensor_to_world(nusc, sample["data"]["LIDAR_TOP"])
    T_cam_to_world = sensor_to_world(nusc, sample["data"][channel])
    return np.linalg.inv(T_cam_to_world) @ T_lidar_to_world


def lidar_points(nusc, sample_token: str) -> np.ndarray:
    """LIDAR_TOP keyframe points in the lidar frame, shape (N, 3)."""
    from nuscenes.utils.data_classes import LidarPointCloud

    sd = nusc.get("sample_data", nusc.get("sample", sample_token)["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(str(data_root() / sd["filename"]))
    return pc.points[:3].T.astype(np.float64)


def scene_sample_tokens(nusc, scene_name: str) -> list[str]:
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    tokens = []
    tok = scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = nusc.get("sample", tok)["next"]
    return tokens


def all_keyframes(nusc) -> list[tuple[str, str]]:
    """(scene_name, sample_token) for all keyframes in scene order."""
    out = []
    for scene in nusc.scene:
        for tok in scene_sample_tokens(nusc, scene["name"]):
            out.append((scene["name"], tok))
    return out
