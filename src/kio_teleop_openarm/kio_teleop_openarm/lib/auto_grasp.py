"""Auto grasp state machine extracted from kio_teleop_upoo_mujoco."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from .transforms import make_transform


class AutoGrasp:
    """4-phase autonomous grasp: approach → descend → grasp → lift."""

    PHASES = ["approach", "descend", "grasp", "lift"]
    NEXT_PHASE = {"idle": "approach", "approach": "descend", "descend": "grasp",
                  "grasp": "lift", "lift": "done"}

    def __init__(self, gripper_open_value, gripper_close_value,
                 arrive_threshold=0.03):
        self.active = False
        self.arm = "right"
        self.phase = "idle"
        self.target_matrix = None
        self.arrive_threshold = float(arrive_threshold)
        self._smoothed_target_pos = None

        self.gripper_open_value = float(gripper_open_value)
        self.gripper_close_value = float(gripper_close_value)
        self.gripper_cmd = None  # (side, value) or None

    def advance(self, cup_body_id, ee_body_left, ee_body_right,
                data_xpos_fn, data_xquat_fn, cup_pos_override=None):
        """Advance one phase. Returns log message string."""
        if cup_body_id < 0:
            return "[auto] No cup in scene"

        cup_pos = (cup_pos_override if cup_pos_override is not None
                   else data_xpos_fn(cup_body_id).copy())
        ee_body = ee_body_left if self.arm == "left" else ee_body_right
        ee_pos = data_xpos_fn(ee_body).copy()
        ee_quat = data_xquat_fn(ee_body)

        r_ee = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
        rot_mat = r_ee.as_matrix()
        fingertip_offset_local = np.array([0.0, 0.0, -0.18])
        fingertip_offset_world = rot_mat @ fingertip_offset_local

        if self.phase not in self.NEXT_PHASE:
            return "[auto] Grasp sequence finished"

        new_phase = self.NEXT_PHASE[self.phase]
        msg = ""

        if new_phase == "approach":
            self._smoothed_target_pos = None
            target_pos = cup_pos + np.array([0.0, 0.0, 0.30], dtype=np.float32) - fingertip_offset_world
            self.target_matrix = make_transform(target_pos, rot_mat)
            self.active = True

        elif new_phase == "descend":
            self._smoothed_target_pos = None
            target_pos = cup_pos + np.array([0.0, 0.0, 0.0], dtype=np.float32) - fingertip_offset_world
            self.target_matrix = make_transform(target_pos, rot_mat)

        elif new_phase == "grasp":
            self.gripper_cmd = (self.arm, self.gripper_close_value)
            msg = "[auto] Close gripper..."

        elif new_phase == "lift":
            self._smoothed_target_pos = None
            target_pos = cup_pos + np.array([0.0, 0.0, 0.10], dtype=np.float32) - fingertip_offset_world
            self.target_matrix = make_transform(target_pos, rot_mat)

        elif new_phase == "done":
            self.active = False
            self.target_matrix = None
            msg = "[auto] Grasp sequence complete"

        self.phase = new_phase
        if new_phase in ("approach", "descend", "lift"):
            msg = f"[auto] {new_phase}: target=({self.target_matrix[0,3]:.3f},{self.target_matrix[1,3]:.3f},{self.target_matrix[2,3]:.3f})"
        return msg

    def get_smoothed_target(self, ee_pos):
        """Get exponentially smoothed IK target matrix for the current phase."""
        if self.target_matrix is None:
            return None
        target_pos = self.target_matrix[:3, 3].copy()
        if self._smoothed_target_pos is None:
            self._smoothed_target_pos = ee_pos
        alpha = 0.08
        self._smoothed_target_pos = (
            (1.0 - alpha) * self._smoothed_target_pos + alpha * target_pos)
        smoothed = self.target_matrix.copy()
        smoothed[:3, 3] = self._smoothed_target_pos
        return smoothed
