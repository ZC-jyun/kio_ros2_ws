"""Calibration logic extracted from kio_teleop_upoo_mujoco."""

import time
import numpy as np
from .transforms import make_transform, quat_xyzw_to_matrix


class Calibrator:
    """VR ↔ world coordinate frame calibration."""

    def __init__(
        self,
        t_world_robotbase,
        t_robotbase_world,
        robot_head_pos_robotbase,
        static_cam_lookat,
        static_cam_distance,
        static_cam_elevation,
        static_cam_azimuth,
        calibration_delay_sec=5.0,
    ):
        self.t_world_robotbase = t_world_robotbase
        self.t_robotbase_world = t_robotbase_world
        self.robot_head_pos_robotbase = robot_head_pos_robotbase
        self.static_cam_lookat = np.asarray(static_cam_lookat, dtype=np.float32)
        self.static_cam_distance = float(static_cam_distance)
        self.static_cam_elevation = float(static_cam_elevation)
        self.static_cam_azimuth = float(static_cam_azimuth)
        self.calibration_delay_sec = float(calibration_delay_sec)

        self.calibration_ready = False
        self.calibration_requested = False
        self.calibration_capture_time = None
        self.last_countdown_print = None

        # Calibration results
        self.t_world_vuer = None
        self.t_robotbase_vuer = None
        self.t_vuer_inithead = None
        self.t_robotbase_inithead = None
        self.t_world_inithead = None
        self.t_robotbase_left_hand_ref = None
        self.t_robotbase_right_hand_ref = None
        self.t_robotbase_left_eef_ref = None
        self.t_robotbase_right_eef_ref = None

    def _desired_t_robotbase_inithead(self):
        R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return make_transform(self.robot_head_pos_robotbase.astype(np.float32), R)

    def request(self):
        if not self.calibration_ready:
            self.calibration_requested = True
            self.calibration_capture_time = time.time() + self.calibration_delay_sec
            self.last_countdown_print = None
            return f"Calibration requested. Capturing in {self.calibration_delay_sec:.1f}s..."
        return "Calibration already active."

    def capture(self, t_vuer_head, t_vuer_left_hand, t_vuer_right_hand,
                get_body_pose_world_fn):
        """Capture calibration frames. get_body_pose_world_fn(body_id) → 4x4 transform."""
        # Compute world_inithead from static camera
        lookat = self.static_cam_lookat
        dist = self.static_cam_distance
        elev = np.radians(self.static_cam_elevation)
        azim = np.radians(self.static_cam_azimuth + 90.0)
        forward = np.array(
            [np.cos(elev) * np.sin(azim), np.cos(elev) * np.cos(azim), np.sin(elev)],
            dtype=np.float32)
        cam_pos = lookat - forward * dist
        left_v = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), forward)
        left_v = left_v / (np.linalg.norm(left_v) + 1e-8)
        up_v = np.cross(forward, left_v)
        r_head = np.column_stack([forward, left_v, up_v])
        self.t_world_inithead = make_transform(cam_pos, r_head)

        self.t_robotbase_inithead = self._desired_t_robotbase_inithead()
        self.t_vuer_inithead = t_vuer_head.copy()
        self.t_world_vuer = (
            self.t_world_inithead @ np.linalg.inv(self.t_vuer_inithead)).astype(np.float32)
        self.t_robotbase_vuer = (self.t_robotbase_world @ self.t_world_vuer).astype(np.float32)
        self.t_robotbase_left_hand_ref = (self.t_robotbase_vuer @ t_vuer_left_hand).astype(np.float32)
        self.t_robotbase_right_hand_ref = (self.t_robotbase_vuer @ t_vuer_right_hand).astype(np.float32)
        self.t_robotbase_left_eef_ref = (
            self.t_robotbase_world @ get_body_pose_world_fn("left")).astype(np.float32)
        self.t_robotbase_right_eef_ref = (
            self.t_robotbase_world @ get_body_pose_world_fn("right")).astype(np.float32)
        self.calibration_ready = True

    def maybe_capture(self, t_vuer_head, t_vuer_left, t_vuer_right, get_body_pose_world_fn):
        """Check if it's time to capture and do so. Returns status string or None."""
        if self.calibration_ready or not self.calibration_requested:
            return None
        remaining = self.calibration_capture_time - time.time()
        if remaining > 0:
            remaining_int = int(np.ceil(remaining))
            if remaining_int != self.last_countdown_print:
                self.last_countdown_print = remaining_int
                return f"Calibration in {remaining_int}..."
            return None
        self.capture(t_vuer_head, t_vuer_left, t_vuer_right, get_body_pose_world_fn)
        self.calibration_requested = False
        self.calibration_capture_time = None
        self.last_countdown_print = None
        return "Calibration captured. Teleoperation active."
