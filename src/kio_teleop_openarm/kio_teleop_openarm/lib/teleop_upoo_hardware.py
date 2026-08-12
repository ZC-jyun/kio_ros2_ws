#!/usr/bin/env python3
"""
UPOO Arm VR Teleop — Hardware Edition v2

Servo-mode MuJoCo simulation + optional DM motor control via USB2CANFD.
Incorporates improvements from kio_teleop_upoo_mujoco.py (bimanual version):
  - Servo mode: ctrl-driven PD actuators instead of direct qpos write
  - Real-time multi-step physics: wall-clock-aware mj_step count
  - GLContext for offscreen rendering
  - Pre-built joint index maps

Thread model:
  Main thread  (~30-60 Hz): VR → IK → target publish → MuJoCo render
  Motor thread (~1 kHz):    Read target → smooth → clip → CAN MIT send

Safety: E=estop, P=calibrate, R=reset cup
"""

import argparse
import atexit
import os
import select
import signal
import sys
import tempfile
import threading
import time
from multiprocessing import Event, Queue, shared_memory
from pathlib import Path

import numpy as np

# ── Paths ──
_LOCAL_LIB_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
_ROBOT_ROOT = Path(__file__).resolve().parents[5]
_TOOLS_DIR = _WORKSPACE_ROOT / "tools"
_UPOO_SOURCE = _ROBOT_ROOT / "kio_upoo-main"
_DEPLOY_DIR = Path(os.environ.get(
    "KIO_TELEOP_DEPLOY_DIR",
    _ROBOT_ROOT / "openarm-main" / "teleop_deploy",
)).expanduser()
_TELEVISION_DIR = _DEPLOY_DIR / "television"
_DEFAULT_CERT_FILE = _ROBOT_ROOT / "openarm-main" / "teleop" / "cert.pem"
_DEFAULT_KEY_FILE = _ROBOT_ROOT / "openarm-main" / "teleop" / "key.pem"
for _path in (_LOCAL_LIB_DIR, _TOOLS_DIR, _UPOO_SOURCE, _DEPLOY_DIR, _TELEVISION_DIR):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

import upoo_motor_constants as umc
from pytransform3d import rotations

_simulation_import_error = None
try:
    import mujoco
    import mujoco.viewer
    from TeleVision import OpenTeleVision
    from constants_vuer import grd_yup2grd_zup
    from motion_utils import mat_update, fast_mat_inv
except ImportError as exc:
    _simulation_import_error = exc

# ── Damiao imports ────────────────────────────────────────────
_damiao_available = False
_dmcan_available = False
try:
    from dmcan import dmcan_device_type
    _dmcan_available = True
except ImportError:
    pass

if _dmcan_available:
    try:
        from damiao import (
            Control_Mode, Control_Mode_Code, DM_Motor_Type, DM_REG,
            DmActData, Motor_Control,
        )
        _damiao_available = True
    except ImportError:
        pass

# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

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


def quat_error(q_current_xyzw, q_target_xyzw):
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


def safe_get_landmarks(tv, side: str):
    candidates = [f"{side}_landmarks", f"{side}_hand_landmarks", f"{side}HandLandmarks"]
    for name in candidates:
        if hasattr(tv, name):
            try:
                arr = np.asarray(getattr(tv, name), dtype=np.float32).reshape(-1, 3)
                if arr.shape[0] >= 10 and np.isfinite(arr).all():
                    return arr
            except Exception:
                pass
    return None


def normalized_pinch_metric(landmarks, thumb_tip_index=4, index_tip_index=9):
    if landmarks is None:
        return np.nan
    lm = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)
    n = lm.shape[0]
    if n <= max(thumb_tip_index, index_tip_index):
        return np.nan
    thumb, index = lm[thumb_tip_index], lm[index_tip_index]
    pinch_dist = float(np.linalg.norm(thumb - index))
    palm_candidates = []
    for a, b in [(0, 10), (5, 20), (0, 5), (0, 17), (5, 17)]:
        if n > max(a, b):
            d = float(np.linalg.norm(lm[a] - lm[b]))
            if np.isfinite(d) and d > 1e-5:
                palm_candidates.append(d)
    palm = max(palm_candidates) if palm_candidates else 1.0
    return pinch_dist / max(palm, 1e-5)


def materialize_collector_scene(cup_x=None, cup_y=None):
    from vr_left_grasp_scene import cup_grid_xy, scene_xml

    default_x, default_y = cup_grid_xy(0)
    x = default_x if cup_x is None else float(cup_x)
    y = default_y if cup_y is None else float(cup_y)
    if not np.isfinite([x, y]).all():
        raise ValueError("cup_x and cup_y must be finite")

    scene_dir = tempfile.TemporaryDirectory(prefix="upoo_vr_hardware_")
    scene_path = Path(scene_dir.name) / "scene.xml"
    scene_path.write_text(scene_xml(x, y), encoding="utf-8")

    assets_source = _UPOO_SOURCE / "openarm_mujoco-master" / "v2" / "assets"
    if not assets_source.is_dir():
        scene_dir.cleanup()
        raise FileNotFoundError(f"Collector model assets not found: {assets_source}")
    os.symlink(assets_source, Path(scene_dir.name) / "assets", target_is_directory=True)
    return scene_dir, scene_path.resolve(), x, y


# ═══════════════════════════════════════════════════════════════
# VR 预处理
# ═══════════════════════════════════════════════════════════════

class AbsoluteVuerPreprocessor:
    def __init__(self):
        self.vuer_head_mat = np.eye(4, dtype=np.float32)
        self.vuer_left_wrist_mat = np.eye(4, dtype=np.float32)

    def process(self, tv):
        self.vuer_head_mat = mat_update(
            self.vuer_head_mat, tv.head_matrix.copy())
        self.vuer_left_wrist_mat = mat_update(
            self.vuer_left_wrist_mat, tv.left_hand.copy())
        t_vuer_head = (
            grd_yup2grd_zup @ self.vuer_head_mat
            @ fast_mat_inv(grd_yup2grd_zup)
        )
        t_vuer_left = (
            grd_yup2grd_zup @ self.vuer_left_wrist_mat
            @ fast_mat_inv(grd_yup2grd_zup)
        )
        return t_vuer_head.astype(np.float32), t_vuer_left.astype(np.float32)


class VuerTeleop:
    def __init__(self, resolution=(480, 640), ngrok=True, cert_file="./cert.pem", key_file="./key.pem"):
        self.resolution = resolution
        self.resolution_cropped = resolution
        self.img_shape = (resolution[0], 2 * resolution[1], 3)
        self.shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.img_shape)) * np.uint8().itemsize)
        self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.shm.buf)
        self.image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(
            self.resolution_cropped, self.shm.name, self.image_queue,
            toggle_streaming, ngrok=ngrok, cert_file=cert_file, key_file=key_file,
        )
        self.processor = AbsoluteVuerPreprocessor()

    def step(self):
        t_vuer_head, t_vuer_left = tuple(
            value.copy() for value in self.processor.process(self.tv))
        left_landmarks = safe_get_landmarks(self.tv, "left")
        return t_vuer_head, t_vuer_left, left_landmarks


# ═══════════════════════════════════════════════════════════════
# MuJoCo 立体渲染
# ═══════════════════════════════════════════════════════════════

def make_stereo_cameras(model, scene, cam_lookat, cam_distance, cam_azimuth, cam_elevation,
                         width=640, height=480, ipd=0.064):
    cam_left  = mujoco.MjvCamera()
    cam_right = mujoco.MjvCamera()
    for cam in (cam_left, cam_right):
        cam.lookat[:]  = cam_lookat
        cam.distance   = cam_distance
        cam.azimuth    = cam_azimuth
        cam.elevation  = cam_elevation
        cam.type       = mujoco.mjtCamera.mjCAMERA_FREE

    forward = np.array([
        np.cos(np.radians(cam_elevation)) * np.sin(np.radians(cam_azimuth)),
        np.cos(np.radians(cam_elevation)) * np.cos(np.radians(cam_azimuth)),
        np.sin(np.radians(cam_elevation)),
    ])
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right_vec = np.cross(forward, np.array([0, 0, 1]))
    right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-8)
    cam_left.lookat[:]  = cam_lookat - right_vec * (ipd / 2)
    cam_right.lookat[:] = cam_lookat + right_vec * (ipd / 2)

    r_left  = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    r_right = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    vp = mujoco.MjrRect(0, 0, width, height)
    return cam_left, cam_right, r_left, r_right, vp


def render_stereo(model, data, scene, cam_left, cam_right, r_left, r_right, vp):
    opt = mujoco.MjvOption()
    mujoco.mjv_updateScene(model, data, opt, None, cam_left,  mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(vp, scene, r_left)
    mujoco.mjv_updateScene(model, data, opt, None, cam_right, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(vp, scene, r_right)
    left_rgb  = np.empty((vp.height, vp.width, 3), dtype=np.uint8)
    right_rgb = np.empty((vp.height, vp.width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(left_rgb,  None, vp, r_left)
    mujoco.mjr_readPixels(right_rgb, None, vp, r_right)
    return left_rgb[::-1, :], right_rgb[::-1, :]


def set_camera_free_pose(cam, position, lookat):
    dir_vec = lookat - position
    dist = float(np.linalg.norm(dir_vec))
    if dist < 1e-6:
        return
    d = dir_vec / dist
    cam.lookat[:]  = lookat
    cam.distance   = dist
    cam.elevation  = float(np.degrees(np.arcsin(d[2])))
    cam.azimuth    = float(np.degrees(np.arctan2(d[1], d[0])))


# ═══════════════════════════════════════════════════════════════
# UPOO Arm Sim v2 — Servo Mode + Real-Time Physics
# ═══════════════════════════════════════════════════════════════

class UPOOArmSimV2:
    """Single-arm UPOO simulation with servo-mode control.

    Key improvements over v1 (teleop_upoo.py):
      - Servo mode: sets data.ctrl for PD actuators instead of writing qpos
      - Real-time multi-step physics: n_steps based on wall-clock elapsed time
      - GLContext for offscreen rendering (cleaner than raw glfw)
      - Pre-built _jnt_qposadr2id map for joint limit clipping
    """

    ARM_JOINT_NAMES = [
        "upoo_left_J01", "upoo_left_J02", "upoo_left_J03",
        "upoo_left_J04", "upoo_left_J05", "upoo_left_J06",
    ]
    FINGER_JOINT_NAMES = [
        "upoo_left_openarm_v1_finger_joint1",
        "upoo_left_openarm_v1_finger_joint2",
    ]
    EE_SITE_NAME = "upoo_left_tcp"
    BODY_LINK_NAME = "upoo_left_base_link"
    GRIPPER_ACTUATOR_NAME = "upoo_left_gripper_ctrl"

    def __init__(
        self,
        print_freq=False,
        orientation_weight=1.0, position_gain=1.0, orientation_gain=0.8,
        damping=0.1, max_dq=0.05, position_scale=1.0,
        robot_base_xyz=(0.0, 0.0, 0.0),
        base_roll_deg=0.0, base_pitch_deg=0.0, base_yaw_deg=0.0,
        calibration_delay_sec=5.0,
        cup_x=None,
        cup_y=None,
        enable_gripper=True,
        gripper_open_value=0.044,
        gripper_close_value=0.005,
        gripper_close_threshold=0.25,
        gripper_open_threshold=0.75,
        gripper_smoothing=0.35,
        arm_smoothing=0.3,
        ik_max_iters=3,
        ik_tolerance=0.001,
        thumb_tip_index=4,
        index_tip_index=9,
        stereo_res=(640, 480),
        joint_weights=None,
    ):
        # ── 参数 ──
        self.print_freq = print_freq
        self.position_gain      = float(position_gain)
        self.orientation_gain   = float(orientation_gain)
        self.orientation_weight = float(orientation_weight)
        self.damping            = float(damping)
        self.max_dq             = float(max_dq)
        if joint_weights is None:
            self.joint_weights = np.ones(6, dtype=np.float32)
        else:
            self.joint_weights = np.array(joint_weights, dtype=np.float32)
            if self.joint_weights.shape != (6,):
                raise ValueError(f"joint_weights must be 6 values, got {self.joint_weights.shape}")
        self.position_scale    = float(position_scale)
        self.arm_smoothing     = float(arm_smoothing)
        self.ik_max_iters      = int(ik_max_iters)
        self.ik_tolerance      = float(ik_tolerance)
        self.robot_base_xyz    = np.array(robot_base_xyz, dtype=np.float32)
        base_quat_xyzw = self._euler_xyz_deg_to_quat_xyzw(base_roll_deg, base_pitch_deg, base_yaw_deg)
        self.calibration_delay_sec = float(calibration_delay_sec)

        self.enable_gripper          = bool(enable_gripper)
        self.gripper_open_value      = float(gripper_open_value)
        self.gripper_close_value     = float(gripper_close_value)
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.gripper_open_threshold  = float(gripper_open_threshold)
        self.gripper_smoothing       = float(gripper_smoothing)
        self.thumb_tip_index  = int(thumb_tip_index)
        self.index_tip_index  = int(index_tip_index)

        self.gripper_cmd  = self.gripper_open_value
        self._gripper_landmarks_ready = False
        self.gripper_fixed_value = self.gripper_open_value

        self.calibration_ready     = False
        self.calibration_requested = False
        self.calibration_capture_time = None
        self.last_countdown_print  = None

        self.t_world_vuer = None
        self.t_robotbase_vuer = None
        self.t_vuer_inithead = None
        self.t_robotbase_inithead = None
        self.t_world_inithead = None
        self.t_robotbase_left_hand_ref = None
        self.t_robotbase_left_eef_ref  = None

        # Use the exact scene generator used by vr_collect_act_grasp_data.py.
        (self._scene_tempdir, xml_path,
         self.cup_x, self.cup_y) = materialize_collector_scene(cup_x, cup_y)
        print(
            f"[mujoco] Collector scene: {xml_path} "
            f"cup=({self.cup_x:+.3f}, {self.cup_y:+.3f})"
        )
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data  = mujoco.MjData(self.model)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        # Reset to home keyframe
        try:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        except Exception:
            pass

        # ── 预建 joint 索引映射 (from bimanual pattern) ──
        self._jnt_qposadr2id = {}
        for jid in range(self.model.njnt):
            adr = self.model.jnt_qposadr[jid]
            if adr >= 0:
                self._jnt_qposadr2id[adr] = jid

        # Cup
        self._cup_qpos_adr = -1
        self._cup_init_qpos = None
        try:
            cup_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
            if cup_body >= 0:
                cup_jnt = self.model.body_jntadr[cup_body]
                self._cup_qpos_adr = self.model.jnt_qposadr[cup_jnt]
                self._cup_init_qpos = self.data.qpos[self._cup_qpos_adr:self._cup_qpos_adr+7].copy()
                print(f"[init] Cup qpos stored: adr={self._cup_qpos_adr}, pos={self._cup_init_qpos[:3]}")
        except Exception:
            pass

        # The collector controls the left-arm TCP site.
        self.ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE_NAME)

        # Arm joint qpos/dof indices
        self.arm_qpos_indices = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.ARM_JOINT_NAMES
        ], dtype=int)
        self.arm_dof_indices = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.ARM_JOINT_NAMES
        ], dtype=int)

        # Base link (camera reference)
        self.body_link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.BODY_LINK_NAME)

        # Left gripper joints and actuator use the collector model names.
        finger_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.FINGER_JOINT_NAMES
        ]
        self.finger_left_qpos = self.model.jnt_qposadr[finger_joint_ids[0]]
        self.finger_right_qpos = self.model.jnt_qposadr[finger_joint_ids[1]]
        self.finger_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
            self.GRIPPER_ACTUATOR_NAME)
        self._finger_act_set = {self.finger_act}

        required_ids = [
            self.ee_site, self.body_link_id, self.finger_act, *finger_joint_ids,
        ]
        if min(required_ids) < 0:
            raise RuntimeError(
                "Collector scene is missing a required left-arm TCP, base, "
                "joint, or gripper actuator"
            )

        print(f"[init] nq={self.model.nq}, nu={self.model.nu}")
        print(f"[init] arm qpos indices: {self.arm_qpos_indices}")
        for name in self.ARM_JOINT_NAMES + self.FINGER_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            print(f"[init] {name}: qpos_adr={self.model.jnt_qposadr[jid]} "
                  f"dof_adr={self.model.jnt_dofadr[jid]} range={self.model.jnt_range[jid]}")

        self._sync_ctrl_from_qpos()
        self._apply_gripper_ctrl()

        # World ↔ robotbase transform
        self.t_world_robotbase = make_transform(self.robot_base_xyz,
                                                 quat_xyzw_to_matrix(base_quat_xyzw))
        self.t_robotbase_world = np.linalg.inv(self.t_world_robotbase).astype(np.float32)

        # Estimate camera reference height
        self._estimate_head_height()
        self.t_robotbase_inithead = self._desired_t_robotbase_inithead()
        self.t_world_inithead = self.t_world_robotbase @ self.t_robotbase_inithead

        # ── 实时物理步进 ──
        self._last_real_time = None

        # ── 立体渲染 (GLContext) ──
        self.sw, self.sh = stereo_res
        self._stereo_ready = False
        self._gl_context = None
        self._cam_left  = None
        self._cam_right = None
        self._r_left    = None
        self._r_right   = None
        self._vp        = None

    @staticmethod
    def _euler_xyz_deg_to_quat_xyzw(roll_deg, pitch_deg, yaw_deg):
        rpy = np.deg2rad([roll_deg, pitch_deg, yaw_deg]).astype(np.float32)
        quat_wxyz = rotations.quaternion_from_euler(rpy, 0, 1, 2, extrinsic=False)
        return quat_wxyz[[1, 2, 3, 0]].astype(np.float32)

    # ── Ctrl helpers (servo mode) ─────────────────────────────

    def _sync_ctrl_from_qpos(self):
        """Sync ctrl ← qpos, skipping finger actuators."""
        for i in range(self.model.nu):
            if i in self._finger_act_set:
                continue
            jid = self.model.actuator_trnid[i, 0]
            if jid >= 0:
                self.data.ctrl[i] = self.data.qpos[self.model.jnt_qposadr[jid]]

    def sync_arm_from_hardware(self, arm_positions):
        """Seed the MuJoCo arm from fresh physical joint feedback."""
        arm_positions = np.asarray(arm_positions, dtype=np.float64).ravel()
        if arm_positions.shape != (len(self.arm_qpos_indices),):
            raise ValueError(
                f"Expected {len(self.arm_qpos_indices)} arm positions, "
                f"got {arm_positions.shape}"
            )
        if not np.isfinite(arm_positions).all():
            raise ValueError("Hardware arm positions contain non-finite values")

        for index, (qpos_adr, position) in enumerate(
                zip(self.arm_qpos_indices, arm_positions)):
            jid = self._jnt_qposadr2id.get(qpos_adr, -1)
            model_position = float(position)
            if jid >= 0:
                lo, hi = self.model.jnt_range[jid]
                if lo < hi and not lo <= position <= hi:
                    joint_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    boundary_error = max(lo - position, position - hi)
                    if boundary_error > umc.ZERO_VERIFY_TOLERANCE_RAD:
                        raise RuntimeError(
                            f"Hardware {joint_name}={position:+.4f} is outside "
                            f"MuJoCo range [{lo:+.4f}, {hi:+.4f}] by more than "
                            f"the {umc.ZERO_VERIFY_TOLERANCE_RAD:.4f} rad "
                            "zero tolerance"
                        )
                    model_position = float(np.clip(position, lo, hi))
                    hardware_name = umc.ARM_MOTOR_CONFIG[index][0]
                    print(
                        f"[mujoco] {hardware_name} feedback {position:+.4f} is "
                        f"within zero tolerance; seeding model at boundary "
                        f"{model_position:+.4f}"
                    )
            self.data.qpos[qpos_adr] = model_position

        self.data.qvel[self.arm_dof_indices] = 0.0
        self._sync_ctrl_from_qpos()
        mujoco.mj_forward(self.model, self.data)
        self.calibration_ready = False
        self.calibration_requested = False
        self.calibration_capture_time = None
        self.last_countdown_print = None

    def _apply_arm_ctrl(self, q_target):
        """Servo mode: set arm actuator ctrl targets from q_target.

        Does NOT write qpos — lets MuJoCo PD controllers execute the motion.
        This mirrors how real DM motors work: we send position targets,
        the motor's internal PD controller executes them.
        """
        for i in range(self.model.nu):
            if i in self._finger_act_set:
                continue
            jid = self.model.actuator_trnid[i, 0]
            if jid >= 0:
                self.data.ctrl[i] = q_target[self.model.jnt_qposadr[jid]]

    def _apply_gripper_ctrl(self):
        """Set gripper actuator ctrl from gripper_cmd."""
        for act in self._finger_act_set:
            self.data.ctrl[act] = self.gripper_cmd

    # ── Camera / pose helpers ─────────────────────────────────

    def _estimate_head_height(self):
        mujoco.mj_forward(self.model, self.data)
        body_pos_world = self.data.xpos[self.body_link_id]
        body_pos_robotbase = (
            self.t_robotbase_world @ np.r_[body_pos_world, 1.0].astype(np.float32))[:3]
        self.robot_head_pos_robotbase = body_pos_robotbase.copy()
        self.robot_head_height = float(body_pos_robotbase[2])

    def _desired_t_robotbase_inithead(self):
        R = np.eye(3, dtype=np.float32)
        return make_transform(self.robot_head_pos_robotbase.astype(np.float32), R)

    def _init_stereo(self):
        self._gl_context = mujoco.glfw.GLContext(self.sw, self.sh)
        self._gl_context.make_current()

        self.static_cam_lookat    = np.array([0.0, -0.5, 0.5], dtype=np.float32)
        self.static_cam_distance  = 0.8
        self.static_cam_azimuth   = 90.0
        self.static_cam_elevation = -35.0
        (self._cam_left, self._cam_right, self._r_left,
         self._r_right, self._vp) = make_stereo_cameras(
            self.model, self.scene, cam_lookat=self.static_cam_lookat,
            cam_distance=self.static_cam_distance,
            cam_azimuth=self.static_cam_azimuth,
            cam_elevation=self.static_cam_elevation,
            width=self.sw, height=self.sh)
        self._static_cam_left_lookat   = self._cam_left.lookat.copy()
        self._static_cam_left_dist     = self._cam_left.distance
        self._static_cam_left_azimuth  = self._cam_left.azimuth
        self._static_cam_left_elev     = self._cam_left.elevation
        self._static_cam_right_lookat  = self._cam_right.lookat.copy()
        self._static_cam_right_dist    = self._cam_right.distance
        self._static_cam_right_azimuth = self._cam_right.azimuth
        self._static_cam_right_elev    = self._cam_right.elevation
        self._stereo_ready = True

    def _set_head_tracked_cameras(self, t_vuer_currenthead=None):
        if not self._stereo_ready:
            return
        if not self.calibration_ready or t_vuer_currenthead is None:
            self._cam_left.lookat[:]   = self._static_cam_left_lookat
            self._cam_left.distance    = self._static_cam_left_dist
            self._cam_left.azimuth     = self._static_cam_left_azimuth
            self._cam_left.elevation   = self._static_cam_left_elev
            self._cam_right.lookat[:]  = self._static_cam_right_lookat
            self._cam_right.distance   = self._static_cam_right_dist
            self._cam_right.azimuth    = self._static_cam_right_azimuth
            self._cam_right.elevation  = self._static_cam_right_elev
            return

        t_world_currenthead = self.t_world_vuer @ t_vuer_currenthead
        r_world_head = project_to_rotation_matrix(t_world_currenthead[:3, :3])
        p_world_head = t_world_currenthead[:3, 3]
        forward = r_world_head @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        left_v  = r_world_head @ np.array([0.0, 1.0, 0.0], dtype=np.float32)

        ipd = 0.065
        look_dist = 1.0
        lookat = p_world_head + look_dist * forward
        left_eye  = p_world_head + 0.5 * ipd * left_v
        right_eye = p_world_head - 0.5 * ipd * left_v

        self._gl_context.make_current()
        set_camera_free_pose(self._cam_left,  left_eye,  lookat)
        set_camera_free_pose(self._cam_right, right_eye, lookat)

    def _get_site_pose_world(self, site_id):
        pos = self.data.site_xpos[site_id].copy()
        rotation = self.data.site_xmat[site_id].reshape(3, 3).copy()
        return make_transform(pos, rotation)

    # ── 标定 ──────────────────────────────────────────────────

    def _capture_calibration(self, t_vuer_inithead, t_vuer_lefthand_ref):
        mujoco.mj_forward(self.model, self.data)
        lookat = self.static_cam_lookat
        dist = self.static_cam_distance
        elev = np.radians(self.static_cam_elevation)
        azim = np.radians(self.static_cam_azimuth + 90.0)
        forward = np.array([np.cos(elev)*np.sin(azim),
                            np.cos(elev)*np.cos(azim),
                            np.sin(elev)], dtype=np.float32)
        cam_pos = lookat - forward * dist
        left_v = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), forward)
        left_v = left_v / (np.linalg.norm(left_v) + 1e-8)
        up_v = np.cross(forward, left_v)
        r_head = np.column_stack([forward, left_v, up_v])
        head_pos = cam_pos
        self.t_world_inithead = make_transform(head_pos, r_head)

        self.t_robotbase_inithead = self._desired_t_robotbase_inithead()
        self.t_vuer_inithead = t_vuer_inithead.copy()
        self.t_world_vuer = (self.t_world_inithead @ np.linalg.inv(self.t_vuer_inithead)).astype(np.float32)
        self.t_robotbase_vuer = (self.t_robotbase_world @ self.t_world_vuer).astype(np.float32)
        self.t_robotbase_left_hand_ref = (
            self.t_robotbase_vuer @ t_vuer_lefthand_ref
        ).astype(np.float32)
        self.t_robotbase_left_eef_ref = (
            self.t_robotbase_world @ self._get_site_pose_world(self.ee_site)
        ).astype(np.float32)
        self.calibration_ready = True
        print("[teleop] Calibration captured. Teleoperation active.")

    def _request_calibration(self):
        if not self.calibration_ready:
            self.calibration_requested = True
            self.calibration_capture_time = time.time() + self.calibration_delay_sec
            self.last_countdown_print = None
            print(f"[teleop] Calibration requested. Capturing in {self.calibration_delay_sec:.1f}s...")

    def _reset_cup(self):
        if self._cup_qpos_adr >= 0 and self._cup_init_qpos is not None:
            self.data.qpos[self._cup_qpos_adr:self._cup_qpos_adr+7] = self._cup_init_qpos.copy()
            print("[teleop] Cup reset")

    def _maybe_capture_calibration(self, t_vuer_head, t_vuer_left):
        if self.calibration_ready or not self.calibration_requested:
            return
        remaining = self.calibration_capture_time - time.time()
        if remaining > 0:
            remaining_int = int(np.ceil(remaining))
            if remaining_int != self.last_countdown_print:
                print(f"[teleop] Calibration in {remaining_int}...")
                self.last_countdown_print = remaining_int
            return
        self._capture_calibration(t_vuer_head, t_vuer_left)
        self.calibration_requested = False
        self.calibration_capture_time = None
        self.last_countdown_print = None

    # ── 目标位姿 ──────────────────────────────────────────────

    def _target_pose_from_hand(self, t_robotbase_hand_current, t_robotbase_hand_ref, t_robotbase_eef_ref):
        p_delta = (t_robotbase_hand_current[:3, 3] - t_robotbase_hand_ref[:3, 3]) * self.position_scale
        r_delta = project_to_rotation_matrix(
            t_robotbase_hand_current[:3, :3] @ t_robotbase_hand_ref[:3, :3].T)
        p_target = t_robotbase_eef_ref[:3, 3] + p_delta
        r_target = project_to_rotation_matrix(r_delta @ t_robotbase_eef_ref[:3, :3])
        return make_transform(p_target, r_target)

    # ── DLS IK (MuJoCo Jacobian) ──────────────────────────────

    def _ik_step_arm(self, target_pos, target_quat_xyzw):
        mujoco.mj_forward(self.model, self.data)

        current_rotation = self.data.site_xmat[self.ee_site].reshape(3, 3)
        current_quat_xyzw = quat_xyzw_from_matrix(current_rotation)
        pos_err = target_pos - self.data.site_xpos[self.ee_site]
        ori_err = quat_error(current_quat_xyzw, target_quat_xyzw)

        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)

        if self.orientation_weight > 0.0:
            J = np.vstack([jacp[:, self.arm_dof_indices], jacr[:, self.arm_dof_indices]])
            error = np.concatenate([
                pos_err * self.position_gain,
                ori_err * self.orientation_gain * self.orientation_weight,
            ])
        else:
            J = jacp[:, self.arm_dof_indices]
            error = pos_err * self.position_gain

        jTj = J.T @ J
        W2 = np.diag(self.joint_weights.astype(np.float64) ** 2)
        A = jTj + W2 * (self.damping ** 2)
        rhs = J.T @ error

        try:
            dq = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return np.zeros(6)

        if self.max_dq > 0:
            dq = np.clip(dq, -self.max_dq, self.max_dq)
        return dq

    def _ik_solve_arm(self, t_world_target):
        target_pos = t_world_target[:3, 3].astype(np.float32)
        target_quat_xyzw = quat_xyzw_from_matrix(t_world_target[:3, :3])

        q_initial = self.data.qpos[self.arm_qpos_indices].copy()

        for _ in range(self.ik_max_iters):
            dq = self._ik_step_arm(target_pos, target_quat_xyzw)

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

        mujoco.mj_forward(self.model, self.data)
        q_final = self.data.qpos[self.arm_qpos_indices]
        return float(np.linalg.norm(q_final - q_initial))

    def compute_ik(self, t_vuer_head, t_vuer_left):
        if self.print_freq:
            tic = time.time()

        self._maybe_capture_calibration(t_vuer_head, t_vuer_left)

        if not self.calibration_ready:
            return None, 0.0

        t_robotbase_left_current = (
            self.t_robotbase_vuer @ t_vuer_left
        ).astype(np.float32)

        t_robotbase_left_target = self._target_pose_from_hand(
            t_robotbase_left_current,
            self.t_robotbase_left_hand_ref,
            self.t_robotbase_left_eef_ref,
        )
        t_world_left_target = (
            self.t_world_robotbase @ t_robotbase_left_target
        ).astype(np.float32)

        dq_norm = self._ik_solve_arm(t_world_left_target)

        if self.print_freq:
            dt = time.time() - tic
            if dt > 0:
                print(f"[ik] {1.0 / dt:.1f} Hz")

        return self.data.qpos.copy(), dq_norm

    # ── 夹爪 ──────────────────────────────────────────────────

    def _gripper_command_from_landmarks(self, landmarks):
        if not self.enable_gripper:
            return self.gripper_fixed_value
        metric = normalized_pinch_metric(landmarks,
                                         thumb_tip_index=self.thumb_tip_index,
                                         index_tip_index=self.index_tip_index)
        prev_metric = getattr(self, '_gripper_metric', None)
        if prev_metric is None:
            prev_metric = metric
        if not np.isfinite(metric):
            metric = prev_metric
        smoothed = (1.0 - self.gripper_smoothing) * prev_metric + self.gripper_smoothing * metric
        self._gripper_metric = smoothed
        denom = max(self.gripper_open_threshold - self.gripper_close_threshold, 1e-6)
        alpha_open = np.clip((smoothed - self.gripper_close_threshold) / denom, 0.0, 1.0)
        raw = alpha_open * self.gripper_close_value + (1.0 - alpha_open) * self.gripper_open_value
        return float(raw)

    # ── 仿真 & 渲染 (servo mode + real-time steps) ────────────

    def apply_and_render(self, q_actual, t_vuer_head):
        """Servo-mode step + stereo render.

        1. Set arm ctrl targets (not qpos) — PD actuators execute the motion
        2. Set gripper ctrl
        3. Multi-step physics based on wall-clock elapsed time
        4. Render stereo pair
        """
        self._apply_arm_ctrl(q_actual)
        self._apply_gripper_ctrl()

        # Real-time step count (from bimanual version)
        now = time.time()
        sim_timestep = self.model.opt.timestep
        if self._last_real_time is not None:
            real_elapsed = now - self._last_real_time
            n_steps = max(1, int(real_elapsed / sim_timestep))
            n_steps = min(n_steps, 50)
        else:
            n_steps = 1
        self._last_real_time = now

        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        if self._stereo_ready:
            self._set_head_tracked_cameras(t_vuer_head)
            self._gl_context.make_current()
            left_img, right_img = render_stereo(
                self.model, self.data, self.scene,
                self._cam_left, self._cam_right,
                self._r_left, self._r_right, self._vp)
            return left_img, right_img
        return None, None

    def step_simulation_free(self):
        """Pre-calibration free-running simulation."""
        now = time.time()
        sim_timestep = self.model.opt.timestep
        if self._last_real_time is not None:
            real_elapsed = now - self._last_real_time
            n_steps = max(1, int(real_elapsed / sim_timestep))
            n_steps = min(n_steps, 50)
        else:
            n_steps = 1
        self._last_real_time = now
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def render_only(self, t_vuer_head):
        """Render without stepping physics."""
        if not self._stereo_ready:
            return None, None
        self._set_head_tracked_cameras(t_vuer_head)
        self._gl_context.make_current()
        left_img, right_img = render_stereo(
            self.model, self.data, self.scene,
            self._cam_left, self._cam_right,
            self._r_left, self._r_right, self._vp)
        return left_img, right_img

    def close(self):
        if self._gl_context is not None:
            self._gl_context.free()
            self._gl_context = None
        if getattr(self, "_scene_tempdir", None) is not None:
            self._scene_tempdir.cleanup()
            self._scene_tempdir = None


# ═══════════════════════════════════════════════════════════════
# Hardware Motor Bridge
# ═══════════════════════════════════════════════════════════════

class HardwareMotorBridge:
    """Thread-safe bridge between IK joint angles and DM motor CAN control.

    Runs a background thread at ~1 kHz that reads target joint angles,
    applies time-based slew limits and bounded torque feedforward, and sends
    MIT commands.
    """

    def __init__(
        self,
        kp=None,
        kd=None,
        motor_smoothing=umc.MOTOR_SMOOTHING,
        max_step=None,
        device_sn=None,
        enable_gripper=True,
        calibration_record=None,
        require_control_tests=True,
    ):
        if not _damiao_available:
            raise RuntimeError(
                "damiao.py or dmcan not importable. "
                "Ensure the damiao directory is on sys.path "
                "and dmcan is installed."
            )

        self._kp = np.array(kp if kp is not None else umc.DEFAULT_KP, dtype=np.float64)
        self._kd = np.array(kd if kd is not None else umc.DEFAULT_KD, dtype=np.float64)
        self._motor_smoothing = float(motor_smoothing)
        self._device_sn = device_sn or umc.USB2CANFD_SN
        self._enable_gripper = bool(enable_gripper)
        self._calibration_record = Path(calibration_record or umc.CALIBRATION_RECORD)
        self._require_control_tests = bool(require_control_tests)
        umc.validate_calibration_record(
            self._calibration_record,
            require_control_tests=self._require_control_tests,
            expected_kp=self._kp,
            expected_kd=self._kd,
        )
        if not self._require_control_tests:
            print(
                "[safety] WARNING: mapped-control evidence is not required for this run; "
                "zero and direction checks still passed."
            )
        self._direction = np.asarray(
            umc.arm_direction_vector() + [1.0], dtype=np.float64)

        if len(self._kp) != umc.NUM_MOTORS or len(self._kd) != umc.NUM_MOTORS:
            raise ValueError(
                f"kp/kd must have {umc.NUM_MOTORS} elements "
                f"(6 arm + 1 gripper), got kp={len(self._kp)} kd={len(self._kd)}"
            )
        if not np.isfinite(self._kp).all() or not np.isfinite(self._kd).all():
            raise ValueError("kp/kd must contain only finite values")
        if (self._kp < 0).any() or (self._kd < 0).any():
            raise ValueError("kp/kd must be non-negative")
        if (self._kp > umc.MAX_RUNTIME_KP).any():
            raise ValueError(
                f"kp must not exceed {umc.MAX_RUNTIME_KP:g}"
            )
        if (self._kd > umc.MAX_RUNTIME_KD).any():
            raise ValueError(f"kd must not exceed {umc.MAX_RUNTIME_KD:g}")
        if not 0.0 <= self._motor_smoothing <= 1.0:
            raise ValueError("motor_smoothing must be in [0, 1]")
        if umc.MOTOR_CTRL_FREQ <= 0:
            raise ValueError("MOTOR_CTRL_FREQ must be positive")

        # Shared state (protected by _lock)
        self._lock = threading.Lock()
        self._target_q = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self._target_feedforward_tau = np.zeros(
            umc.NUM_MOTORS, dtype=np.float64
        )
        self._emergency_stop = False
        self._last_sent_q = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self._last_sent_feedforward_tau = np.zeros(
            umc.NUM_MOTORS, dtype=np.float64
        )
        self._last_read_q = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self._last_read_dq = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self._last_read_torque = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self._motor_err = np.zeros(umc.NUM_MOTORS, dtype=np.int32)
        self._loop_frequency_hz = 0.0
        self._loop_mean_dt = 0.0
        self._loop_max_dt = 0.0

        # Thread control
        self._running = threading.Event()
        self._thread = None

        # Build motor init data
        init_data = []
        for joint_name, can_id, mst_id in umc.ARM_MOTOR_CONFIG:
            init_data.append(DmActData(
                motorType=getattr(DM_Motor_Type, umc.ARM_MOTOR_TYPES[joint_name]),
                mode=Control_Mode.MIT_MODE,
                can_id=can_id,
                mst_id=mst_id,
            ))
        if self._enable_gripper:
            init_data.append(DmActData(
                motorType=getattr(DM_Motor_Type, umc.GRIPPER_MOTOR_TYPE),
                mode=Control_Mode.MIT_MODE,
                can_id=umc.GRIPPER_CAN_ID,
                mst_id=umc.GRIPPER_MST_ID,
            ))

        # Create Motor_Control (auto_enable=False avoids libusb threading crash)
        self._control = Motor_Control(
            umc.NOM_BAUD, umc.DAT_BAUD,
            sn=self._device_sn,
            data_ptr=init_data,
            device_type=dmcan_device_type.USB2CANFD,
            auto_enable=False,
        )

        # CAN ID lookup: motor index → can_id (needed for error clearing below)
        self._can_ids = [cid for _, cid, _ in umc.ARM_MOTOR_CONFIG] + [umc.GRIPPER_CAN_ID]
        self._active_indices = list(range(umc.ARM_DOF))
        if self._enable_gripper:
            self._active_indices.append(umc.ARM_DOF)

        # Clear motor errors (motors are NOT enabled here — that happens in start())
        for _ in range(5):
            for i in self._active_indices:
                can_id = self._can_ids[i]
                self._control.control_cmd(can_id, 0xFB, 0)
            time.sleep(0.005)

        # Build soft limit arrays for fast clipping
        self._limit_lo = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self._limit_hi = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        for i, (joint_name, _, _) in enumerate(umc.ARM_MOTOR_CONFIG):
            lo, hi = umc.SOFT_POSITION_LIMITS[joint_name]
            hard_lo, hard_hi = umc.HARD_POSITION_LIMITS[joint_name]
            if not hard_lo <= lo < hi <= hard_hi:
                raise ValueError(
                    f"Invalid {joint_name} soft limits [{lo}, {hi}]; "
                    f"hard limits are [{hard_lo}, {hard_hi}]"
                )
            self._limit_lo[i] = lo
            self._limit_hi[i] = hi
        lo, hi = umc.SOFT_POSITION_LIMITS["gripper"]
        self._limit_lo[umc.ARM_DOF] = lo
        self._limit_hi[umc.ARM_DOF] = hi
        self._startup_limit_lo = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self._startup_limit_hi = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        for i, (joint_name, _, _) in enumerate(umc.ARM_MOTOR_CONFIG):
            lo, hi = umc.startup_position_limits(joint_name)
            self._startup_limit_lo[i] = lo
            self._startup_limit_hi[i] = hi
        self._startup_limit_lo[umc.ARM_DOF] = self._limit_lo[umc.ARM_DOF]
        self._startup_limit_hi[umc.ARM_DOF] = self._limit_hi[umc.ARM_DOF]

        # Time-based slew limits remain correct when the CAN loop misses its
        # nominal frequency.
        if max_step is not None:
            legacy_step = np.broadcast_to(
                np.asarray(max_step, dtype=np.float64),
                umc.NUM_MOTORS,
            ).copy()
            self._max_speed = legacy_step * float(umc.MOTOR_CTRL_FREQ)
        else:
            self._max_speed = np.asarray(
                umc.MAX_COMMAND_SPEED, dtype=np.float64
            )
            if (
                self._max_speed.shape != (umc.NUM_MOTORS,)
                or (self._max_speed <= 0).any()
            ):
                raise ValueError(
                    f"MAX_COMMAND_SPEED must contain {umc.NUM_MOTORS} positive values"
                )
        self._feedforward_torque_limit = np.asarray(
            umc.MAX_FEEDFORWARD_TORQUE, dtype=np.float64
        )
        self._feedforward_torque_slew = np.asarray(
            umc.MAX_FEEDFORWARD_TORQUE_SLEW, dtype=np.float64
        )
        for name, values in (
            ("MAX_FEEDFORWARD_TORQUE", self._feedforward_torque_limit),
            (
                "MAX_FEEDFORWARD_TORQUE_SLEW",
                self._feedforward_torque_slew,
            ),
        ):
            if values.shape != (umc.NUM_MOTORS,) or (values < 0).any():
                raise ValueError(
                    f"{name} must contain {umc.NUM_MOTORS} non-negative values"
                )

        # Track which motors are actually present on the bus
        self._motor_connected = [False] * umc.NUM_MOTORS

        # Safety net: disable motors on exit, even if caller forgets stop()
        self._stopped = False
        atexit.register(self._disable_motors)

    # ── Public API ────────────────────────────────────────────

    def set_zero_all(self):
        """Set current position as zero for all connected motors.

        Pauses the motor thread, sends 0xFE zero command to each connected
        motor, resets tracked positions to 0, then resumes.
        """
        # Pause the motor thread so we don't race with control_mit
        was_running = self._running.is_set()
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        time.sleep(0.01)

        for i in range(umc.NUM_MOTORS):
            if not self._motor_connected[i]:
                continue
            motor = self._control.getMotor(self._can_ids[i])
            if motor is None:
                continue
            try:
                self._control.set_zero_position(motor)
                time.sleep(0.003)
            except Exception as exc:
                print(f"[motor] set_zero motor {i} failed: {exc}", file=sys.stderr)

        # Reset tracked positions to zero
        self._last_sent_q[:] = 0.0
        self._last_sent_feedforward_tau[:] = 0.0
        with self._lock:
            self._target_q[:] = 0.0
            self._target_feedforward_tau[:] = 0.0

        # Resume motor thread
        if was_running:
            self._running.set()
            self._thread = threading.Thread(
                target=self._motor_thread, name="motor-ctrl", daemon=True)
            self._thread.start()

        connected = [f"{umc.ARM_MOTOR_CONFIG[i][0] if i < umc.ARM_DOF else 'gripper'}"
                      for i in range(umc.NUM_MOTORS) if self._motor_connected[i]]
        print(f"[motor] Zero set for: {connected}")

    def start(self):
        if self._thread is not None:
            return

        umc.validate_calibration_record(
            self._calibration_record,
            require_control_tests=self._require_control_tests,
            expected_kp=self._kp,
            expected_kd=self._kd,
        )
        self._validate_motor_parameters()

        # A real zero angle is valid. Readiness is based on a newly received
        # CAN frame, not on the numeric position value.
        positions = self._read_fresh_positions(timeout_sec=0.5)
        for i in self._active_indices:
            name = (umc.ARM_MOTOR_CONFIG[i][0]
                    if i < umc.ARM_DOF else "gripper")
            allowed_lo, allowed_hi = umc.startup_position_limits(name)
            if not allowed_lo <= positions[i] <= allowed_hi:
                raise RuntimeError(
                    f"Refusing to enable: {name}={positions[i]:+.4f} is outside "
                    f"startup limits [{allowed_lo:+.4f}, {allowed_hi:+.4f}]"
                )
        self._last_sent_q[:] = positions
        self._last_sent_feedforward_tau[:] = 0.0
        self._last_read_q[:] = positions
        with self._lock:
            self._target_q[:] = positions
            self._target_feedforward_tau[:] = 0.0
        print(f"[motor] Pre-enable positions: "
              f"{[round(self._target_q[i], 4) for i in self._active_indices]}")

        # Start the control thread with its send gate closed, then enable the
        # registered motors and open only their gates.
        self._motor_connected = [False] * umc.NUM_MOTORS
        self._running.set()
        self._emergency_stop = False
        self._thread = threading.Thread(
            target=self._motor_thread, name="motor-ctrl", daemon=True)
        self._thread.start()
        time.sleep(0.005)

        self._motor_connected = [False] * umc.NUM_MOTORS
        self._control.enable_all()
        for i in self._active_indices:
            self._motor_connected[i] = True

        print(f"[motor] Control thread started at {int(umc.MOTOR_CTRL_FREQ)} Hz "
              f"targets={[round(self._target_q[i], 4) for i in self._active_indices]}")

    def _validate_motor_parameters(self):
        """Refuse arming when motor mode or MIT ranges differ from config."""
        registers = (("PMAX", DM_REG.PMAX), ("VMAX", DM_REG.VMAX),
                     ("TMAX", DM_REG.TMAX))
        for i in self._active_indices:
            motor = self._control.getMotor(self._can_ids[i])
            if motor is None:
                raise RuntimeError(f"Motor index {i} is not registered")
            name = (umc.ARM_MOTOR_CONFIG[i][0]
                    if i < umc.ARM_DOF else "gripper")
            type_name = (umc.ARM_MOTOR_TYPES[name]
                         if i < umc.ARM_DOF else umc.GRIPPER_MOTOR_TYPE)
            mode = self._control.read_motor_param(motor, DM_REG.CTRL_MODE, timeout=0.5)
            if mode is None or int(mode) != int(Control_Mode_Code.MIT):
                raise RuntimeError(
                    f"Refusing to enable {name}: CTRL_MODE={mode!r}, expected MIT=1")
            actual = []
            for register_name, register in registers:
                value = self._control.read_motor_param(motor, register, timeout=0.5)
                if value is None:
                    raise RuntimeError(f"No {register_name} response from {name}")
                actual.append(float(value))
            expected = umc.EXPECTED_MIT_LIMITS[type_name]
            if not np.allclose(actual, expected, rtol=0.0, atol=1e-3):
                raise RuntimeError(
                    f"Refusing to enable {name}: MIT limits {tuple(actual)} do not "
                    f"match {type_name} expected {expected}")
            motor.limit_param = actual
            print(f"[motor] {name}: {type_name}, direction="
                  f"{int(self._direction[i]):+d}, MIT limits={tuple(actual)}")

    def _read_fresh_positions(self, timeout_sec):
        motors = {}
        timestamps = {}
        for i in self._active_indices:
            motor = self._control.getMotor(self._can_ids[i])
            if motor is None:
                raise RuntimeError(f"Motor index {i} is not registered")
            motors[i] = motor
            timestamps[i] = float(motor.last_time_)
            self._control.refresh_motor_status(motor)

        pending = set(self._active_indices)
        deadline = time.monotonic() + float(timeout_sec)
        while pending and time.monotonic() < deadline:
            pending = {
                i for i in pending
                if float(motors[i].last_time_) <= timestamps[i]
            }
            if pending:
                time.sleep(0.005)
        if pending:
            names = [
                umc.ARM_MOTOR_CONFIG[i][0] if i < umc.ARM_DOF else "gripper"
                for i in sorted(pending)
            ]
            raise TimeoutError(f"No fresh CAN feedback from: {', '.join(names)}")

        positions = self._last_sent_q.copy()
        for i, motor in motors.items():
            positions[i] = self._direction[i] * float(motor.Get_Position())
            status = int(motor.Get_err())
            if umc.is_motor_fault(status):
                raise RuntimeError(
                    f"Motor index {i} reports fault 0x{status:X} "
                    f"({umc.motor_status_label(status)})"
                )
        if not np.isfinite(positions[self._active_indices]).all():
            raise RuntimeError("Fresh motor feedback contains non-finite positions")
        return positions

    def _disable_motors(self):
        """Send motor-disable CAN frames. Safe to call even after close()."""
        try:
            self._control.disable_all()
            time.sleep(0.05)  # let CAN frames flush before USB teardown
            print("[motor] Motors disabled.")
        except Exception as exc:
            print(f"[motor] disable_all error: {exc}", file=sys.stderr)

    def stop(self):
        if self._stopped:
            return
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._disable_motors()
        atexit.unregister(self._disable_motors)
        self._stopped = True
        # NOTE: _control.close() calls context.destroy() which segfaults
        # due to a libusb async I/O threading bug in libdm_device.so.
        # We skip it — the OS reclaims USB resources on process exit.
        print("[motor] Stopped.")

    def set_target(
        self,
        q: np.ndarray,
        feedforward_torque: np.ndarray | None = None,
    ):
        q = np.asarray(q, dtype=np.float64).ravel()
        if q.shape[0] != umc.NUM_MOTORS:
            raise ValueError(f"Expected {umc.NUM_MOTORS} targets, got {q.shape[0]}")
        if not np.isfinite(q).all():
            raise ValueError("Motor targets must contain only finite values")
        if feedforward_torque is None:
            feedforward_torque = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        feedforward_torque = np.asarray(
            feedforward_torque, dtype=np.float64
        ).ravel()
        if (
            feedforward_torque.shape != (umc.NUM_MOTORS,)
            or not np.isfinite(feedforward_torque).all()
        ):
            raise ValueError(
                "Feedforward torque must contain seven finite values"
            )
        with self._lock:
            if self._emergency_stop:
                return
            np.copyto(self._target_q, np.clip(q, self._limit_lo, self._limit_hi))
            np.copyto(
                self._target_feedforward_tau,
                np.clip(
                    feedforward_torque,
                    -self._feedforward_torque_limit,
                    self._feedforward_torque_limit,
                ),
            )

    def get_state(self):
        """Return feedback in MuJoCo coordinates (q, dq, torque, faults)."""
        with self._lock:
            return (self._last_read_q.copy(), self._last_read_dq.copy(),
                    self._last_read_torque.copy(), self._motor_err.copy())

    def get_sent_target(self):
        with self._lock:
            return self._last_sent_q.copy()

    def get_sent_feedforward_torque(self):
        with self._lock:
            return self._last_sent_feedforward_tau.copy()

    def get_control_timing(self):
        with self._lock:
            return {
                "frequency_hz": float(self._loop_frequency_hz),
                "mean_dt": float(self._loop_mean_dt),
                "max_dt": float(self._loop_max_dt),
                "target_frequency_hz": float(umc.MOTOR_CTRL_FREQ),
            }

    def emergency_stop(self):
        with self._lock:
            self._emergency_stop = True
            self._target_feedforward_tau[:] = 0.0
            self._last_sent_feedforward_tau[:] = 0.0
            self._running.clear()
            self._motor_connected = [False] * umc.NUM_MOTORS
        print("[motor] EMERGENCY STOP: disabling motors", file=sys.stderr)
        self._disable_motors()

    # ── Motor control thread ──────────────────────────────────

    @staticmethod
    def _rate_limited_position(
        last_q,
        target_q,
        max_speed,
        dt,
        target_weight,
    ):
        filtered_target = last_q + target_weight * (target_q - last_q)
        max_delta = max_speed * dt
        return last_q + np.clip(
            filtered_target - last_q,
            -max_delta,
            max_delta,
        )

    def _motor_thread(self):
        period = 1.0 / umc.MOTOR_CTRL_FREQ
        last_debug_ts = time.monotonic()
        _thread_start = time.monotonic()  # grace period reference
        previous_cycle_start = None
        timing_window_start = _thread_start
        timing_count = 0
        timing_dt_sum = 0.0
        timing_max_dt = 0.0

        while self._running.is_set():
            t_start = time.perf_counter()
            cycle_start = time.monotonic()
            raw_dt = (
                period
                if previous_cycle_start is None
                else max(cycle_start - previous_cycle_start, 1e-6)
            )
            previous_cycle_start = cycle_start
            slew_dt = min(raw_dt, umc.MAX_SLEW_DT_SEC)

            estop, target, target_tau, kp, kd = self._snapshot_targets()
            if estop:
                break

            sent_q = self._rate_limited_position(
                self._last_sent_q,
                target,
                self._max_speed,
                slew_dt,
                self._motor_smoothing,
            )
            sent_q = np.clip(
                sent_q,
                self._startup_limit_lo,
                self._startup_limit_hi,
            )
            max_tau_delta = self._feedforward_torque_slew * slew_dt
            sent_tau = self._last_sent_feedforward_tau + np.clip(
                target_tau - self._last_sent_feedforward_tau,
                -max_tau_delta,
                max_tau_delta,
            )
            sent_tau = np.clip(
                sent_tau,
                -self._feedforward_torque_limit,
                self._feedforward_torque_limit,
            )
            with self._lock:
                self._last_sent_q[:] = sent_q
                self._last_sent_feedforward_tau[:] = sent_tau

            for i in range(umc.NUM_MOTORS):
                if not self._motor_connected[i]:
                    continue
                motor = self._control.getMotor(self._can_ids[i])
                if motor is None:
                    continue
                try:
                    motor_q = self._direction[i] * sent_q[i]
                    motor_tau = self._direction[i] * sent_tau[i]
                    self._control.control_mit(
                        motor, float(kp[i]), float(kd[i]),
                        motor_q, 0.0, float(motor_tau),
                    )
                except Exception as exc:
                    print(f"[motor] control_mit error motor {i}: {exc}", file=sys.stderr)

            # Error flags + motor state readback (connected motors only)
            for i in range(umc.NUM_MOTORS):
                if not self._motor_connected[i]:
                    continue
                motor = self._control.getMotor(self._can_ids[i])
                if motor is not None:
                    status = int(motor.Get_err())
                    self._motor_err[i] = status if umc.is_motor_fault(status) else 0
                    self._last_read_q[i] = self._direction[i] * float(motor.state_q)
                    self._last_read_dq[i] = self._direction[i] * float(motor.state_dq)
                    self._last_read_torque[i] = (
                        self._direction[i] * float(getattr(motor, 'state_tau', 0.0)))

            # CAN timeout (skip during startup grace period, connected motors only)
            now = time.monotonic()
            timing_count += 1
            timing_dt_sum += raw_dt
            timing_max_dt = max(timing_max_dt, raw_dt)
            timing_elapsed = now - timing_window_start
            if timing_elapsed >= 1.0:
                with self._lock:
                    self._loop_frequency_hz = timing_count / timing_elapsed
                    self._loop_mean_dt = timing_dt_sum / timing_count
                    self._loop_max_dt = timing_max_dt
                timing_window_start = now
                timing_count = 0
                timing_dt_sum = 0.0
                timing_max_dt = 0.0

            if now - _thread_start > 5.0:
                for i in range(umc.NUM_MOTORS):
                    if not self._motor_connected[i]:
                        continue
                    motor = self._control.getMotor(self._can_ids[i])
                    if motor is not None:
                        dt = now - float(motor.last_time_)
                        if dt > umc.CAN_TIMEOUT_SEC:
                            print(f"[motor] CAN timeout motor {i} (dt={dt:.3f}s)", file=sys.stderr)
                            self.emergency_stop()
                            break

            if now - last_debug_ts >= 5.0:
                pos_str = " ".join(f"{self._last_sent_q[i]:.3f}" for i in range(umc.ARM_DOF))
                err_str = " ".join(str(self._motor_err[i]) for i in range(umc.NUM_MOTORS))
                timing = self.get_control_timing()
                tau_str = " ".join(
                    f"{self._last_sent_feedforward_tau[i]:+.2f}"
                    for i in range(umc.ARM_DOF)
                )
                print(
                    f"[motor] cmd=[{pos_str}] "
                    f"grip={self._last_sent_q[umc.ARM_DOF]:.4f} "
                    f"ff_tau=[{tau_str}] err=[{err_str}] "
                    f"loop={timing['frequency_hz']:.1f}Hz "
                    f"mean_dt={timing['mean_dt'] * 1e3:.2f}ms "
                    f"max_dt={timing['max_dt'] * 1e3:.2f}ms"
                )
                last_debug_ts = now

            elapsed = time.perf_counter() - t_start
            sleep_t = period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _snapshot_targets(self):
        with self._lock:
            return (
                bool(self._emergency_stop),
                self._target_q.copy(),
                self._target_feedforward_tau.copy(),
                self._kp.copy(),
                self._kd.copy(),
            )

    # ── Context manager ───────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="UPOO Arm VR Teleop — Hardware Edition v2 (servo mode)")

    # Hardware flags
    parser.add_argument("--motor-enable", action="store_true",
                        help="Enable hardware motor control via USB2CANFD")
    parser.add_argument("--device-sn", type=str, default=None,
                        help="USB2CANFD device serial number")
    parser.add_argument("--kp", type=float, nargs=7, default=None,
                        help="MIT position gains [j1..j6 gripper]")
    parser.add_argument("--kd", type=float, nargs=7, default=None,
                        help="MIT damping gains [j1..j6 gripper]")
    parser.add_argument("--motor-smoothing", type=float, default=umc.MOTOR_SMOOTHING)
    parser.add_argument("--motor-freq", type=float, default=None)
    parser.add_argument(
        "--calibration-record", type=Path, default=umc.CALIBRATION_RECORD,
        help="Validated direction/zero/control-test record required before arming")
    parser.add_argument(
        "--skip-mapped-control-check",
        action="store_true",
        help=(
            "Allow a commissioning run without mapped-control test records. "
            "Zero and direction records remain mandatory."
        ),
    )

    # Teleop flags
    parser.add_argument(
        "--ngrok", action="store_true",
        help="Use the legacy plain HTTP/ngrok mode instead of local TLS")
    parser.add_argument(
        "--local-cert", action="store_true",
        help="Use local TLS (default; retained for compatibility)")
    parser.add_argument("--cert-file", type=str, default=str(_DEFAULT_CERT_FILE))
    parser.add_argument("--key-file", type=str, default=str(_DEFAULT_KEY_FILE))
    parser.add_argument("--print-freq", action="store_true")
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--position-gain", type=float, default=1.0)
    parser.add_argument("--orientation-gain", type=float, default=0.8)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--max-dq", type=float, default=0.05)
    parser.add_argument("--ik-max-iters", type=int, default=3)
    parser.add_argument("--ik-tolerance", type=float, default=0.001)
    parser.add_argument("--joint-weights", type=float, nargs=6, default=None)
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument(
        "--cup-x", type=float, default=None,
        help="Collector-scene cup X; defaults to grid point 0")
    parser.add_argument(
        "--cup-y", type=float, default=None,
        help="Collector-scene cup Y; defaults to grid point 0")
    parser.add_argument("--robot-x", type=float, default=0.0)
    parser.add_argument("--robot-y", type=float, default=0.0)
    parser.add_argument("--robot-z", type=float, default=0.0)
    parser.add_argument("--base-roll-deg", type=float, default=0.0)
    parser.add_argument("--base-pitch-deg", type=float, default=0.0)
    parser.add_argument("--base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--calibration-delay-sec", type=float, default=5.0)
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument("--gripper-open-value", type=float, default=0.044)
    parser.add_argument("--gripper-close-value", type=float, default=0.005)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.25)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.75)
    parser.add_argument("--gripper-smoothing", type=float, default=0.3)
    parser.add_argument("--arm-smoothing", type=float, default=1.0)
    parser.add_argument("--thumb-tip-index", type=int, default=4)
    parser.add_argument("--index-tip-index", type=int, default=9)
    return parser.parse_args()


def create_teleop(args):
    # Local TLS is the safe default for Quest/WebXR. Plain HTTP/ngrok is
    # opt-in because it does not provide a secure local VR origin.
    ngrok_mode = bool(args.ngrok)
    return VuerTeleop(
        resolution=(480, 640), ngrok=ngrok_mode,
        cert_file=args.cert_file, key_file=args.key_file,
    )


def create_sim(args):
    return UPOOArmSimV2(
        print_freq=args.print_freq,
        orientation_weight=args.orientation_weight,
        position_gain=args.position_gain,
        orientation_gain=args.orientation_gain,
        damping=args.damping,
        max_dq=args.max_dq,
        position_scale=args.position_scale,
        robot_base_xyz=(args.robot_x, args.robot_y, args.robot_z),
        base_roll_deg=args.base_roll_deg,
        base_pitch_deg=args.base_pitch_deg,
        base_yaw_deg=args.base_yaw_deg,
        calibration_delay_sec=args.calibration_delay_sec,
        cup_x=args.cup_x,
        cup_y=args.cup_y,
        enable_gripper=not args.disable_gripper,
        gripper_open_value=args.gripper_open_value,
        gripper_close_value=args.gripper_close_value,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
        gripper_smoothing=args.gripper_smoothing,
        arm_smoothing=args.arm_smoothing,
        ik_max_iters=args.ik_max_iters,
        ik_tolerance=args.ik_tolerance,
        thumb_tip_index=args.thumb_tip_index,
        index_tip_index=args.index_tip_index,
        joint_weights=args.joint_weights,
    )


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    # damiao installs a SIGINT handler at import time which only logs the
    # signal. Restore normal KeyboardInterrupt handling for orderly shutdown.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    args = parse_args()

    if _simulation_import_error is not None:
        print(
            f"[ERROR] VR/MuJoCo dependencies could not be imported from "
            f"{_DEPLOY_DIR}: {_simulation_import_error}",
            file=sys.stderr,
        )
        print(
            "[ERROR] Set KIO_TELEOP_DEPLOY_DIR to the directory containing "
            "the TeleVision deployment dependencies.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.motor_enable and not _damiao_available:
        print("[ERROR] --motor-enable requested but damiao/dmcan not found.", file=sys.stderr)
        print("[ERROR] Ensure dmcan is installed and "
              "the damiao directory is accessible.", file=sys.stderr)
        sys.exit(1)

    if args.motor_freq is not None:
        umc.MOTOR_CTRL_FREQ = float(args.motor_freq)

    teleop = create_teleop(args)
    sim = create_sim(args)

    hw_bridge = None
    hardware_armed = False
    if args.motor_enable:
        hw_bridge = HardwareMotorBridge(
            kp=args.kp if args.kp is not None else None,
            kd=args.kd if args.kd is not None else None,
            motor_smoothing=args.motor_smoothing,
            device_sn=args.device_sn,
            enable_gripper=not args.disable_gripper,
            calibration_record=args.calibration_record,
            require_control_tests=not args.skip_mapped_control_check,
        )

    print("\n[teleop] ======== UPOO Arm Hardware Teleop v2 (servo mode) ========")
    print("[teleop] 6-DOF, 左手控制（与采集程序一致）")
    hardware_status = "READY (press A to arm)" if hw_bridge else "DISABLED (sim only)"
    print(f"[teleop] 硬件电机: {hardware_status}")
    if hw_bridge and args.skip_mapped_control_check:
        print("[teleop] 警告: 本次明确跳过 mapped-control 记录检查（首次联调模式）")
    print("[teleop] 仿真模式: servo (ctrl-driven PD actuators) + real-time steps")
    print("[teleop] A=真机上电  P=VR标定  R=复位杯子  E=急停")
    if hw_bridge:
        print(f"[teleop] kp={hw_bridge._kp}  kd={hw_bridge._kd}")
    print()

    viewer = None
    try:
        # Init stereo before viewer (GLContext needs active context)
        sim._init_stereo()

        try:
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            viewer.cam.lookat[:]  = sim.model.stat.center
            viewer.cam.distance   = sim.model.stat.extent * 0.8
            viewer.cam.azimuth    = sim.model.vis.global_.azimuth
            viewer.cam.elevation  = sim.model.vis.global_.elevation
        except Exception as e:
            print(f"[viewer] passive viewer unavailable: {e}")

        # State
        q_smoothed = sim.data.qpos.copy()
        alpha = float(args.arm_smoothing)
        frame_count = 0
        last_debug_ts = time.time()

        # Stdin reader
        stdin_stop = threading.Event()
        arm_requested = threading.Event()

        def _stdin_reader():
            while not stdin_stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.5)
                if r:
                    line = sys.stdin.readline().strip().lower()
                    if line == 'a':
                        arm_requested.set()
                    elif line == 'p':
                        sim._request_calibration()
                    elif line == 'r':
                        sim._reset_cup()
                    elif line == 'e':
                        if hw_bridge and hardware_armed:
                            hw_bridge.emergency_stop()
                        print("[stdin] Emergency stop triggered")

        stdin_thread = threading.Thread(target=_stdin_reader, daemon=True)
        stdin_thread.start()

        print("\nMuJoCo viewer open. A=arm hardware  P=calibrate  R=reset cup  E=estop\n")

        while viewer is None or viewer.is_running():
            tic = time.time()

            if arm_requested.is_set():
                arm_requested.clear()
                if hw_bridge is None:
                    print("[motor] --motor-enable was not supplied; staying in simulation mode")
                elif hardware_armed:
                    print("[motor] Hardware is already armed")
                else:
                    print("[motor] Arming hardware and reading physical joint state...")
                    hw_bridge.start()
                    motor_pos, _, _, motor_err = hw_bridge.get_state()
                    if np.any(motor_err[:umc.ARM_DOF] != 0):
                        hw_bridge.emergency_stop()
                        raise RuntimeError(f"Motor errors after arm: {motor_err}")
                    sim.sync_arm_from_hardware(motor_pos[:umc.ARM_DOF])
                    q_smoothed = sim.data.qpos.copy()
                    hardware_armed = True
                    print("[motor] Hardware armed and simulation synchronized. Press P to calibrate VR.")

            # ① VR
            t_vuer_head, t_vuer_left, left_lm = teleop.step()

            # ② IK
            q_result, dq_norm = sim.compute_ik(t_vuer_head, t_vuer_left)

            if q_result is not None and dq_norm > umc.IK_DIVERGENCE_THRESH:
                print(f"[ik] divergence dq={dq_norm:.3f} — skipping", file=sys.stderr)
                q_result = None

            now = time.time()

            if q_result is not None:
                q_smoothed[sim.arm_qpos_indices] = (
                    (1.0 - alpha) * q_smoothed[sim.arm_qpos_indices]
                    + alpha * q_result[sim.arm_qpos_indices])

            # ③ Gripper — always update sim.gripper_cmd for MuJoCo
            if sim.calibration_ready and sim.enable_gripper:
                if left_lm is not None:
                    if not sim._gripper_landmarks_ready:
                        sim.gripper_cmd = float(q_smoothed[sim.finger_left_qpos])
                        sim._gripper_landmarks_ready = True
                        print("[gripper] landmarks ready")
                    sim.gripper_cmd = sim._gripper_command_from_landmarks(left_lm)

            # ④ Hardware motor command
            if hw_bridge and hardware_armed and sim.calibration_ready:
                arm_targets = q_smoothed[sim.arm_qpos_indices].copy()
                full_target = np.append(arm_targets, sim.gripper_cmd)
                hw_bridge.set_target(full_target)

            # ⑤ Render — servo mode (ctrl-driven) or free-run pre-calibration
            if sim.calibration_ready:
                left_img, right_img = sim.apply_and_render(q_smoothed, t_vuer_head)
            else:
                sim.step_simulation_free()
                left_img, right_img = sim.render_only(t_vuer_head)

            if left_img is not None and teleop is not None:
                rgb_stereo = np.ascontiguousarray(np.hstack((left_img.copy(), right_img.copy())))
                np.copyto(teleop.img_array, rgb_stereo)

            # ⑥ Gripper debug
            if (sim.calibration_ready and sim.enable_gripper
                    and frame_count % 60 == 0):
                metric = getattr(sim, '_gripper_metric', float('nan'))
                fl_q = float(sim.data.qpos[sim.finger_left_qpos])
                fr_q = float(sim.data.qpos[sim.finger_right_qpos])
                print(f"[gripper] cmd={sim.gripper_cmd:.4f} metric={metric:.3f} "
                      f"qpos=(L={fl_q:.4f},R={fr_q:.4f})")

            if viewer is not None:
                viewer.sync()
            frame_count += 1

            # ⑦ Periodic diagnostic
            if args.print_freq and now - last_debug_ts >= 2.0:
                js = {}
                for name in sim.ARM_JOINT_NAMES:
                    try:
                        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                        js[name] = float(q_smoothed[sim.model.jnt_qposadr[jid]])
                    except Exception:
                        js[name] = 0.0
                lh = t_vuer_left
                print(f"[DIAG] LH=({lh[0,3]:.2f},{lh[1,3]:.2f},{lh[2,3]:.2f}) "
                      f"dq={dq_norm:.3f} "
                      + " ".join(f"j{i}={js.get(n,0):.3f}"
                                 for i, n in enumerate(sim.ARM_JOINT_NAMES, 1)))
                if hw_bridge:
                    pos, vel, torque, errs = hw_bridge.get_state()
                    motor_str = " ".join(f"{pos[i]:.3f}" for i in range(umc.ARM_DOF))
                    print(f"[DIAG] motor=[{motor_str}] "
                          f"grip={pos[umc.ARM_DOF]:.4f} err={errs}")
                last_debug_ts = now

            if args.print_freq:
                dt = time.time() - tic
                if dt > 0:
                    print(f"[main] {1.0 / dt:.1f} Hz")

        stdin_stop.set()

    except KeyboardInterrupt:
        print("\n[main] Interrupted.")
    except Exception as exc:
        print(f"\n[main] Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if hw_bridge:
            print("[main] Stopping motors...")
            hw_bridge.stop()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        if sim is not None:
            sim.close()
        if teleop is not None:
            try:
                teleop.shm.close()
                teleop.shm.unlink()
            except Exception:
                pass

    print("[main] Exit.")


if __name__ == "__main__":
    main()
