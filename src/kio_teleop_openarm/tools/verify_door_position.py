#!/usr/bin/env python3
"""Verify door handle position is within robot workspace and IK-reachable.

Usage:
  python tools/verify_door_position.py --x 0.5 --y 0.0 --z 0.9 --arm left
"""

import sys
import numpy as np
from pathlib import Path

_DEPLOY_DIR = Path("/home/kiorobot/kio_robot_zzc/openarm-main/teleop_deploy")
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

import mujoco
import openarm_mujoco.v2 as openarm_mujoco

LEFT_ARM_JOINTS = [
    "upoo_left_Base_J01", "upoo_left_J02", "upoo_left_J03",
    "upoo_left_J04", "upoo_left_J05", "upoo_left_J06",
]
RIGHT_ARM_JOINTS = [
    "upoo_right_Base_J01", "upoo_right_J02", "upoo_right_J03",
    "upoo_right_J04", "upoo_right_J05", "upoo_right_J06",
]
LEFT_EE_BODY = "upoo_left_Link_06"
RIGHT_EE_BODY = "upoo_right_Link_06"


def verify_door_handle(handle_pos, arm="left"):
    """Verify handle position is reachable via MuJoCo IK."""
    p = np.array(handle_pos, dtype=np.float64)

    xml_path = openarm_mujoco.openarm_upoo_bimanual_xml()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    def _qpos_idx(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid]

    def _dof_idx(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_dofadr[jid]

    arm_joints = LEFT_ARM_JOINTS if arm == "left" else RIGHT_ARM_JOINTS
    ee_body_name = LEFT_EE_BODY if arm == "left" else RIGHT_EE_BODY
    qpos_indices = np.array([_qpos_idx(n) for n in arm_joints], dtype=int)
    dof_indices = np.array([_dof_idx(n) for n in arm_joints], dtype=int)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

    # Set home position
    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "upoo_home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    except Exception:
        pass
    data.qpos[qpos_indices[0]] = -1.57

    # Range check
    shoulder_offset = np.array([0.0, 0.18, 0.35])
    dist = np.linalg.norm(p - shoulder_offset)
    if dist > 0.75:
        print(f"[verify] WARNING: handle {dist:.3f}m from shoulder, near limit")
    else:
        print(f"[verify] Distance check: {dist:.3f}m from shoulder — OK")

    # IK reachability: try multiple approach directions
    from kio_teleop_openarm.lib.ik_solver import IKSolver
    ik = IKSolver(model, data, body_id, dof_indices, qpos_indices, ik_max_iters=5)

    approaches = [
        np.array([0, 0, -1]),   # top-down
        np.array([0, -1, 0]),   # from front
        np.array([-1, 0, 0]),   # from side
    ]

    reachable = False
    for approach in approaches:
        T = np.eye(4)
        T[:3, 3] = p
        z_ee = -approach / np.linalg.norm(approach)
        if abs(z_ee[2]) > 0.9:
            x_ee = np.array([1.0, 0.0, 0.0])
        else:
            x_ee = np.array([0.0, 0.0, 1.0])
        x_ee = x_ee - np.dot(x_ee, z_ee) * z_ee
        x_ee = x_ee / np.linalg.norm(x_ee)
        y_ee = np.cross(z_ee, x_ee)
        y_ee = y_ee / np.linalg.norm(y_ee)
        x_ee = np.cross(y_ee, z_ee)
        T[:3, :3] = np.column_stack([x_ee, y_ee, z_ee])

        dq_norm = ik.solve(T.astype(np.float32))
        body_pos = data.xpos[body_id]
        residual = np.linalg.norm(p - body_pos)
        print(f"  approach {approach}: residual={residual:.4f}m, dq={dq_norm:.4f}")
        if residual < 0.05:
            reachable = True

    if reachable:
        print(f"[verify] PASS: door handle at {p} is reachable")
    else:
        print(f"[verify] FAIL: door handle at {p} NOT reachable")
    return reachable


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify door handle reachability")
    parser.add_argument("--x", type=float, default=0.5, help="Handle X in robot base frame")
    parser.add_argument("--y", type=float, default=0.0, help="Handle Y in robot base frame")
    parser.add_argument("--z", type=float, default=0.9, help="Handle Z in robot base frame")
    parser.add_argument("--arm", default="left", choices=["left", "right"])
    args = parser.parse_args()
    verify_door_handle([args.x, args.y, args.z], args.arm)
