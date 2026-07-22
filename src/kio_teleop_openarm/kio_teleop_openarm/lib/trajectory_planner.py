"""Cartesian path → DLS IK → cubic spline → collision check → joint trajectory."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from .transforms import make_transform, quat_xyzw_from_matrix


class TrajectoryPlanner:
    def __init__(self, ik_solver_left, ik_solver_right,
                 model, data, left_body_id, right_body_id,
                 arm_qpos_left, arm_qpos_right,
                 table_height=0.0,
                 joint_velocity_limit=3.0,
                 default_speed=0.3):
        self.ik_left = ik_solver_left
        self.ik_right = ik_solver_right
        self.model = model
        self.data = data
        self.left_body_id = left_body_id
        self.right_body_id = right_body_id
        self.arm_qpos_left = np.asarray(arm_qpos_left, dtype=int)
        self.arm_qpos_right = np.asarray(arm_qpos_right, dtype=int)
        self.table_height = table_height
        self.v_limit = joint_velocity_limit
        self.default_speed = default_speed
        self.freq = 100

    def plan_pick(self, q_curr: np.ndarray,
                  T_pre_grasp: np.ndarray,
                  T_grasp: np.ndarray,
                  arm: str = "left") -> list | None:
        T_pre_grasp = np.asarray(T_pre_grasp, dtype=np.float64)
        T_grasp = np.asarray(T_grasp, dtype=np.float64)

        # 1. Cartesian waypoints
        T_curr = self._get_current_ee_pose(arm)
        T_lift_up = T_curr.copy()
        T_lift_up[2, 3] += 0.15

        cart_waypoints = [T_curr, T_lift_up, T_pre_grasp, T_grasp]

        # 2. IK for each waypoint
        q_waypoints = [q_curr.copy()]
        saved_qpos = self.data.qpos.copy()
        try:
            self.data.qpos[:] = q_curr
            for i in range(1, len(cart_waypoints)):
                q_wp = self._solve_ik(cart_waypoints[i], q_waypoints[-1], arm)
                if q_wp is None:
                    return None
                q_waypoints.append(q_wp)
        finally:
            self.data.qpos[:] = saved_qpos

        q_waypoints = np.array(q_waypoints)

        # 3. Estimate segment durations
        times = [0.0]
        for i in range(len(cart_waypoints) - 1):
            d = np.linalg.norm(
                cart_waypoints[i + 1][:3, 3] - cart_waypoints[i][:3, 3])
            dt = max(d / self.default_speed, 0.5)
            times.append(times[-1] + dt)

        total_t = times[-1]
        n_samples = max(int(total_t * self.freq), 20)
        t_samples = np.linspace(0, total_t, n_samples)

        # 4. Cubic spline interpolation
        q_samples = np.zeros((n_samples, q_curr.shape[0]))
        for dof in range(q_curr.shape[0]):
            cs = CubicSpline(times, q_waypoints[:, dof], bc_type='natural')
            q_samples[:, dof] = cs(t_samples)

        # 5. Velocity check
        dq = np.diff(q_samples, axis=0) * self.freq
        max_dq = np.max(np.abs(dq))
        if max_dq > self.v_limit:
            scale = max_dq / self.v_limit
            long_times = [t * scale for t in times]
            total_t = long_times[-1]
            n_samples = max(int(total_t * self.freq), 20)
            t_samples = np.linspace(0, total_t, n_samples)
            for dof in range(q_curr.shape[0]):
                cs = CubicSpline(long_times, q_waypoints[:, dof], bc_type='natural')
                q_samples[:, dof] = cs(t_samples)

        # 6. Collision check (sample every 5th frame)
        saved_qpos = self.data.qpos.copy()
        try:
            for i in range(0, n_samples, 5):
                if self._check_collision(q_samples[i], arm):
                    return None
        finally:
            self.data.qpos[:] = saved_qpos

        trajectory = list(zip(t_samples.tolist(),
                              [q.copy() for q in q_samples]))
        return trajectory

    def plan_pick_simple(self, q_curr, T_target, steps=30, arm="left"):
        """Cartesian linear interpolation + SLERP + per-point IK. Fallback."""
        T_curr = self._get_current_ee_pose(arm)
        trajectory = []
        estimated_t = 3.0
        saved_qpos = self.data.qpos.copy()
        try:
            for i in range(1, steps + 1):
                alpha = i / steps
                T_interp = self._interp_pose(T_curr, T_target, alpha)
                q = self._solve_ik(T_interp, q_curr if i == 1 else trajectory[-1][1], arm)
                if q is None:
                    return None
                trajectory.append((alpha * estimated_t, q))
        finally:
            self.data.qpos[:] = saved_qpos
        return trajectory

    def _interp_pose(self, T1, T2, alpha):
        T = np.eye(4)
        T[:3, 3] = T1[:3, 3] + alpha * (T2[:3, 3] - T1[:3, 3])
        rot = R.from_matrix([T1[:3, :3], T2[:3, :3]])
        slerped = rot[0].slerp([0, 1], rot)(alpha)
        T[:3, :3] = slerped.as_matrix()
        return T

    def _get_current_ee_pose(self, arm="left"):
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        body_id = self.left_body_id if arm == "left" else self.right_body_id
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        return make_transform(pos, rot)

    def _solve_ik(self, T_target, q_seed, arm="left"):
        self.data.qpos[:] = q_seed
        ik = self.ik_left if arm == "left" else self.ik_right
        ik.solve(T_target.astype(np.float32))
        body_id = self.left_body_id if arm == "left" else self.right_body_id
        pos = self.data.xpos[body_id]
        residual = np.linalg.norm(T_target[:3, 3] - pos)
        if residual > 0.05:
            return None
        return self.data.qpos.copy()

    def _check_collision(self, q, arm="left"):
        import mujoco
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

        # Self-collision via contact pairs
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom_bodyid[contact.geom1]
            geom2 = self.model.geom_bodyid[contact.geom2]
            if geom1 != geom2 and contact.dist < 0.0:
                return True

        # Table collision: EE must stay above table
        body_id = self.left_body_id if arm == "left" else self.right_body_id
        ee_z = self.data.xpos[body_id][2]
        if ee_z < self.table_height - 0.02:
            return True

        return False
