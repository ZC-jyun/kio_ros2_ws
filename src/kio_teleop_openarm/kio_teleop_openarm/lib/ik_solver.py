"""DLS IK solver extracted from kio_teleop_upoo_mujoco."""

import numpy as np
import mujoco
from .transforms import quat_error, quat_xyzw_from_matrix, project_to_rotation_matrix, make_transform


class IKSolver:
    """Weighted damped least-squares IK for a single 6-DOF arm."""

    def __init__(
        self,
        model,
        data,
        body_id,
        arm_dof_indices,
        arm_qpos_indices,
        joint_weights=None,
        position_gain=1.0,
        orientation_gain=0.8,
        orientation_weight=1.0,
        damping=0.1,
        max_dq=0.05,
        ik_max_iters=3,
        ik_tolerance=0.001,
    ):
        self.model = model
        self.data = data
        self.body_id = body_id
        self.arm_dof_indices = np.asarray(arm_dof_indices, dtype=int)
        self.arm_qpos_indices = np.asarray(arm_qpos_indices, dtype=int)
        self.nv = model.nv

        if joint_weights is None:
            self.joint_weights = np.ones(6, dtype=np.float32)
        else:
            self.joint_weights = np.array(joint_weights, dtype=np.float32)

        self.position_gain = float(position_gain)
        self.orientation_gain = float(orientation_gain)
        self.orientation_weight = float(orientation_weight)
        self.damping = float(damping)
        self.max_dq = float(max_dq)
        self.ik_max_iters = int(ik_max_iters)
        self.ik_tolerance = float(ik_tolerance)

        # Build qpos_adr → joint_id lookup for joint limit clamping
        self._jnt_qposadr2id = {}
        for jid in range(model.njnt):
            adr = model.jnt_qposadr[jid]
            if adr >= 0:
                self._jnt_qposadr2id[adr] = jid

    def _ik_step(self, target_pos, target_quat_xyzw):
        """Single DLS IK step, returns dq (6,)."""
        mujoco.mj_forward(self.model, self.data)

        cur_quat = self.data.xquat[self.body_id]
        cur_quat_xyzw = np.array(
            [cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0]], dtype=np.float32)
        pos_err = target_pos - self.data.xpos[self.body_id]
        ori_err = quat_error(cur_quat_xyzw, target_quat_xyzw)

        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.body_id)

        dofs = self.arm_dof_indices
        if self.orientation_weight > 0.0:
            J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            error = np.concatenate([
                pos_err * self.position_gain,
                ori_err * self.orientation_gain * self.orientation_weight,
            ])
        else:
            J = jacp[:, dofs]
            error = pos_err * self.position_gain

        jTj = J.T @ J
        W2 = np.diag(self.joint_weights.astype(np.float64) ** 2)
        A = jTj + W2 * (self.damping ** 2)
        rhs = J.T @ error

        try:
            dq = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return np.zeros(len(dofs))

        if self.max_dq > 0:
            dq = np.clip(dq, -self.max_dq, self.max_dq)
        return dq

    def solve(self, t_world_target):
        """Iterative IK, modifies self.data.qpos in place. Returns dq change norm."""
        target_pos = t_world_target[:3, 3].astype(np.float32)
        target_quat_xyzw = quat_xyzw_from_matrix(t_world_target[:3, :3])
        q_initial = self.data.qpos[self.arm_qpos_indices].copy()

        for _ in range(self.ik_max_iters):
            dq = self._ik_step(target_pos, target_quat_xyzw)

            for i, qpos_adr in enumerate(self.arm_qpos_indices):
                self.data.qpos[qpos_adr] += dq[i]
                jid = self._jnt_qposadr2id.get(qpos_adr, -1)
                if jid >= 0:
                    jnt_range = self.model.jnt_range[jid]
                    if jnt_range[0] < jnt_range[1]:
                        self.data.qpos[qpos_adr] = np.clip(
                            self.data.qpos[qpos_adr], jnt_range[0], jnt_range[1])

            if np.linalg.norm(dq) < self.ik_tolerance:
                break

        q_final = self.data.qpos[self.arm_qpos_indices]
        return float(np.linalg.norm(q_final - q_initial))


def target_pose_from_hand(t_robotbase_hand_current, t_robotbase_hand_ref, t_robotbase_eef_ref, position_scale=1.0):
    """Compute EE target pose from VR hand delta."""
    p_delta = (t_robotbase_hand_current[:3, 3] - t_robotbase_hand_ref[:3, 3]) * position_scale
    r_delta = project_to_rotation_matrix(
        t_robotbase_hand_current[:3, :3] @ t_robotbase_hand_ref[:3, :3].T)
    p_target = t_robotbase_eef_ref[:3, 3] + p_delta
    r_target = project_to_rotation_matrix(r_delta @ t_robotbase_eef_ref[:3, :3])
    return make_transform(p_target, r_target)
