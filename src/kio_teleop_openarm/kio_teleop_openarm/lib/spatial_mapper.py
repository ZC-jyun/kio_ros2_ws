"""2D detection box + depth map → 3D spatial coordinates (camera → robot base)."""

import numpy as np
import yaml
from pathlib import Path


class SpatialMapper:
    def __init__(self, K: np.ndarray, T_robotbase_camera: np.ndarray):
        self.K = np.asarray(K, dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.T_base_cam = np.asarray(T_robotbase_camera, dtype=np.float64)

    def pixel_to_camera_3d(self, u: float, v: float,
                           depth_map: np.ndarray) -> np.ndarray | None:
        ui, vi = int(round(u)), int(round(v))
        h, w = depth_map.shape
        if not (0 <= vi < h and 0 <= ui < w):
            return None
        z = float(depth_map[vi, ui])
        if z <= 0:
            return None
        return self.K_inv @ np.array([u, v, 1.0]) * z

    def get_object_3d_center(self, bbox, depth_map,
                             sample_ratio=0.3) -> dict | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        roi_hw = int((x2 - x1) * sample_ratio / 2), int((y2 - y1) * sample_ratio / 2)
        r1 = max(0, int(cy) - roi_hw[1])
        r2 = min(depth_map.shape[0], int(cy) + roi_hw[1])
        c1 = max(0, int(cx) - roi_hw[0])
        c2 = min(depth_map.shape[1], int(cx) + roi_hw[0])

        region = depth_map[r1:r2, c1:c2]
        valid = region[region > 0]
        if len(valid) < 20:
            return None

        median_depth = float(np.median(valid))
        center_cam = self.K_inv @ np.array([cx, cy, 1.0]) * median_depth
        center_base = (self.T_base_cam @ np.append(center_cam, 1.0))[:3]

        fx, fy = self.K[0, 0], self.K[1, 1]
        size = {
            "width_m":  round((x2 - x1) * median_depth / fx, 3),
            "height_m": round((y2 - y1) * median_depth / fy, 3),
        }
        return {
            "center_camera":    center_cam,
            "center_robotbase": center_base,
            "size_estimate":    size,
        }


def load_hand_eye_and_build_T_base_cam(
        hand_eye_yaml_path: str,
        model=None, data=None, ee_body_id=None) -> np.ndarray:
    """Build T_robotbase_camera from hand-eye calibration + MuJoCo FK.

    T_base_cam = T_ee2base @ inv(T_cam2ee)
    """
    with open(hand_eye_yaml_path) as f:
        he = yaml.safe_load(f)
    T_cam2ee = np.array(he["T_cam2ee"], dtype=np.float64)

    if model is not None and data is not None and ee_body_id is not None:
        import mujoco
        mujoco.mj_forward(model, data)
        pos = data.xpos[ee_body_id].copy()
        quat = data.xquat[ee_body_id].copy()
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        T_ee2base = np.eye(4)
        T_ee2base[:3, :3] = rot
        T_ee2base[:3, 3] = pos
    else:
        T_ee2base = np.eye(4)

    T_ee_cam = np.linalg.inv(T_cam2ee)
    return T_ee2base @ T_ee_cam
