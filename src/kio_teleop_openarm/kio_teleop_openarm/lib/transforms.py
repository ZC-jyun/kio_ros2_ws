"""Coordinate transform & math utilities extracted from kio_teleop_upoo_mujoco."""

import numpy as np
from pytransform3d import rotations


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def project_to_rotation_matrix(mat3):
    u, _, vh = np.linalg.svd(mat3.astype(np.float64))
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vh
    return r.astype(np.float32)


def make_transform(pos, rmat):
    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = project_to_rotation_matrix(rmat)
    t[:3, 3] = np.asarray(pos, dtype=np.float32)
    return t


def quat_xyzw_from_matrix(mat3):
    mat3 = project_to_rotation_matrix(mat3)
    return rotations.quaternion_from_matrix(mat3)[[1, 2, 3, 0]].astype(np.float32)


def quat_xyzw_to_matrix(q_xyzw):
    q_xyzw = np.asarray(q_xyzw, dtype=np.float32)
    n = np.linalg.norm(q_xyzw)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    q_xyzw = q_xyzw / n
    return rotations.matrix_from_quaternion(
        np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    ).astype(np.float32)


def euler_xyz_deg_to_quat_xyzw(roll_deg, pitch_deg, yaw_deg):
    rpy = np.deg2rad([roll_deg, pitch_deg, yaw_deg]).astype(np.float32)
    quat_wxyz = rotations.quaternion_from_euler(rpy, 0, 1, 2, extrinsic=False)
    return quat_wxyz[[1, 2, 3, 0]].astype(np.float32)


def quat_error(q_current_xyzw, q_target_xyzw):
    """Quaternion orientation error → 3D angular velocity vector (MuJoCo convention)."""
    qc_w, qc_x, qc_y, qc_z = (
        q_current_xyzw[3], q_current_xyzw[0], q_current_xyzw[1], q_current_xyzw[2])
    qt_w, qt_x, qt_y, qt_z = (
        q_target_xyzw[3], q_target_xyzw[0], q_target_xyzw[1], q_target_xyzw[2])
    q_rel_w = qt_w * qc_w + qt_x * qc_x + qt_y * qc_y + qt_z * qc_z
    q_rel_x = -qt_w * qc_x + qt_x * qc_w - qt_y * qc_z + qt_z * qc_y
    q_rel_y = -qt_w * qc_y + qt_x * qc_z + qt_y * qc_w - qt_z * qc_x
    q_rel_z = -qt_w * qc_z - qt_x * qc_y + qt_y * qc_x + qt_z * qc_w
    q_rel = np.array([q_rel_w, q_rel_x, q_rel_y, q_rel_z])
    if q_rel[0] < 0:
        q_rel = -q_rel
    return 2.0 * q_rel[1:]
