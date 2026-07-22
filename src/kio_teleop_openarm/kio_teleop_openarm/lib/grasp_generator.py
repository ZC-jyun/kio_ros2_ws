"""Heuristic grasp pose generation — per-category strategies + IK reachability."""

import numpy as np
from scipy.spatial.transform import Rotation as R


class GraspGenerator:
    STRATEGIES = {
        "medicine box": {
            "approach": np.array([0, 0, -1]),
            "grasp_width": 0.06,
            "pre_offset": 0.10,
        },
        "towel": {
            "approach": np.array([0, 0, -1]),
            "grasp_width": 0.02,
            "pre_offset": 0.08,
        },
        "door handle": {
            "approach": np.array([0, 0, -1]),
            "grasp_width": 0.04,
            "pre_offset": 0.10,
        },
        "takeout bag": {
            "approach": np.array([0, 0, -1]),
            "grasp_width": 0.03,
            "pre_offset": 0.12,
        },
        "cup": {
            "approach": np.array([0, 0, -1]),
            "grasp_width": 0.05,
            "pre_offset": 0.10,
        },
        "bottle": {
            "approach": np.array([0, 0, -1]),
            "grasp_width": 0.04,
            "pre_offset": 0.10,
        },
    }

    DEFAULT = {
        "approach": np.array([0, 0, -1]),
        "grasp_width": 0.05,
        "pre_offset": 0.10,
    }

    def __init__(self, ik_solver_left, ik_solver_right,
                 model, data, left_body_id, right_body_id,
                 table_height=0.0):
        self.ik_left = ik_solver_left
        self.ik_right = ik_solver_right
        self.model = model
        self.data = data
        self.left_body_id = left_body_id
        self.right_body_id = right_body_id
        self.table_height = table_height

    def generate_candidates(self, detection: dict,
                            object_3d: dict,
                            arm: str = "left") -> list[dict]:
        strategy = self.STRATEGIES.get(detection["class_name"], self.DEFAULT)
        center = object_3d["center_robotbase"]
        candidates = []

        top = self._make_one(center, strategy, roll_deg=0.0, arm=arm)
        if top:
            top["description"] = f'{detection["class_name"]} top grasp'
            top["score"] = 0.9
            candidates.append(top)

        side = self._make_one(center, strategy, roll_deg=90.0, arm=arm)
        if side:
            side["description"] = f'{detection["class_name"]} side grasp'
            side["score"] = 0.7
            candidates.append(side)

        return candidates

    def _make_one(self, center, strategy, roll_deg=0.0, arm="left") -> dict | None:
        approach = strategy["approach"].astype(np.float64)
        approach = approach / np.linalg.norm(approach)

        pre_pos = center + approach * strategy["pre_offset"]
        grasp_pos = center

        if pre_pos[2] < self.table_height + 0.02:
            return None
        if grasp_pos[2] < self.table_height:
            return None

        z_ee = -approach
        if abs(z_ee[2]) > 0.9:
            x_ee = np.array([1.0, 0.0, 0.0])
        else:
            x_ee = np.array([0.0, 0.0, 1.0])
        x_ee = x_ee - np.dot(x_ee, z_ee) * z_ee
        x_ee = x_ee / np.linalg.norm(x_ee)

        if abs(roll_deg) > 1e-6:
            rot = R.from_rotvec(z_ee * np.deg2rad(roll_deg))
            x_ee = rot.apply(x_ee)

        y_ee = np.cross(z_ee, x_ee)
        y_ee = y_ee / np.linalg.norm(y_ee)
        x_ee = np.cross(y_ee, z_ee)

        pre_grasp = np.eye(4)
        pre_grasp[:3, :3] = np.column_stack([x_ee, y_ee, z_ee])
        pre_grasp[:3, 3] = pre_pos

        grasp = pre_grasp.copy()
        grasp[:3, 3] = grasp_pos

        # Save current qpos
        saved_qpos = self.data.qpos.copy()
        try:
            reachable = self._reachable(grasp, arm)
        finally:
            self.data.qpos[:] = saved_qpos

        if not reachable:
            return None

        return {
            "grasp_id": 0,
            "pre_grasp_pose": pre_grasp,
            "grasp_pose": grasp,
        }

    def _reachable(self, T_target: np.ndarray, arm: str = "left") -> bool:
        try:
            body_id = self.left_body_id if arm == "left" else self.right_body_id
            cur_pos = self.data.xpos[body_id].copy()
            target_pos = T_target[:3, 3]
            if np.linalg.norm(target_pos - cur_pos) > 1.0:
                return False

            ik = self.ik_left if arm == "left" else self.ik_right
            dq_norm = ik.solve(T_target.astype(np.float32))
            body_pos = self.data.xpos[body_id]
            residual = np.linalg.norm(target_pos - body_pos)
            return residual < 0.05
        except Exception:
            return True
