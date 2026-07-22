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
import select
import sys
import threading
import time
from multiprocessing import Event, Queue, shared_memory
from pathlib import Path

import numpy as np

# ── Paths (deployment layout: all relative) ──
_DEPLOY_DIR = Path(__file__).resolve().parent  # teleop_deploy/
_TELEVISION_DIR = _DEPLOY_DIR / "television"
sys.path.insert(0, str(_DEPLOY_DIR))       # for damiao.py, openarm_mujoco/
sys.path.insert(0, str(_TELEVISION_DIR))   # for TeleVision.py, constants_vuer.py, motion_utils.py

import upoo_motor_constants as umc
from pytransform3d import rotations

try:
    import mujoco
    import mujoco.viewer
    import openarm_mujoco.v2 as openarm_mujoco
    from TeleVision import OpenTeleVision
    from constants_vuer import grd_yup2grd_zup
    from motion_utils import mat_update, fast_mat_inv
except ImportError:
    pass  # Simulation/VR imports — not needed for HardwareMotorBridge

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
        from damiao import DmActData, DM_Motor_Type, Motor_Control, Control_Mode
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


# ═══════════════════════════════════════════════════════════════
# VR 预处理
# ═══════════════════════════════════════════════════════════════

class AbsoluteVuerPreprocessor:
    def __init__(self):
        self.vuer_head_mat = np.eye(4, dtype=np.float32)
        self.vuer_right_wrist_mat = np.eye(4, dtype=np.float32)

    def process(self, tv):
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
        t_vuer_head  = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        t_vuer_right = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        return (t_vuer_head.astype(np.float32), t_vuer_right.astype(np.float32))


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
        t_vuer_head, t_vuer_right = tuple(
            x.copy() for x in self.processor.process(self.tv))
        right_landmarks = safe_get_landmarks(self.tv, "right")
        return t_vuer_head, t_vuer_right, right_landmarks


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

    ARM_JOINT_NAMES = ["Base_J01", "J02", "J03", "J04", "J05", "J06"]
    FINGER_JOINT_NAMES = ["upoo_finger_left", "upoo_finger_right"]
    EE_BODY_NAME = "Link_06"
    BODY_LINK_NAME = "base_link"

    def __init__(
        self,
        print_freq=False,
        orientation_weight=1.0, position_gain=1.0, orientation_gain=0.8,
        damping=0.1, max_dq=0.05, position_scale=1.0,
        robot_base_xyz=(0.0, 0.0, 0.0),
        base_roll_deg=0.0, base_pitch_deg=0.0, base_yaw_deg=0.0,
        calibration_delay_sec=5.0,
        enable_gripper=True,
        gripper_open_value=5.0,
        gripper_close_value=0.0,
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
        self.t_robotbase_right_hand_ref = None
        self.t_robotbase_right_eef_ref  = None

        # ── 加载 MuJoCo 模型 ──
        xml_path = openarm_mujoco.openarm_upoo_xml()
        print(f"[mujoco] 加载模型: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
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

        # EE body
        self.ee_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.EE_BODY_NAME)

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

        # Finger joints
        self.finger_left_qpos  = self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "upoo_finger_left")]
        self.finger_right_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "upoo_finger_right")]

        # Finger actuators — for ctrl-based gripper control
        self.finger_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_left_ctrl")
        # Build set of finger actuator indices to skip in _apply_arm_ctrl
        self._finger_act_set = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ["finger_left_ctrl", "finger_right_ctrl"]
        }

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

    def _get_body_pose_world(self, body_id):
        pos  = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
        return make_transform(pos, quat_xyzw_to_matrix(quat_xyzw))

    # ── 标定 ──────────────────────────────────────────────────

    def _capture_calibration(self, t_vuer_inithead, t_vuer_righthand_ref):
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
        self.t_robotbase_right_hand_ref = (self.t_robotbase_vuer @ t_vuer_righthand_ref).astype(np.float32)
        self.t_robotbase_right_eef_ref  = (self.t_robotbase_world @ self._get_body_pose_world(self.ee_body)).astype(np.float32)
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

    def _maybe_capture_calibration(self, t_vuer_head, t_vuer_right):
        if self.calibration_ready or not self.calibration_requested:
            return
        remaining = self.calibration_capture_time - time.time()
        if remaining > 0:
            remaining_int = int(np.ceil(remaining))
            if remaining_int != self.last_countdown_print:
                print(f"[teleop] Calibration in {remaining_int}...")
                self.last_countdown_print = remaining_int
            return
        self._capture_calibration(t_vuer_head, t_vuer_right)
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

        cur_quat = self.data.xquat[self.ee_body]
        cur_quat_xyzw = np.array([cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0]],
                                  dtype=np.float32)
        pos_err = target_pos - self.data.xpos[self.ee_body]
        ori_err = quat_error(cur_quat_xyzw, target_quat_xyzw)

        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_body)

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

    def compute_ik(self, t_vuer_head, t_vuer_right):
        if self.print_freq:
            tic = time.time()

        self._maybe_capture_calibration(t_vuer_head, t_vuer_right)

        if not self.calibration_ready:
            return None, 0.0

        t_robotbase_right_current = (self.t_robotbase_vuer @ t_vuer_right).astype(np.float32)

        t_robotbase_right_target = self._target_pose_from_hand(
            t_robotbase_right_current, self.t_robotbase_right_hand_ref, self.t_robotbase_right_eef_ref)

        t_world_right_target = (self.t_world_robotbase @ t_robotbase_right_target).astype(np.float32)

        dq_norm = self._ik_solve_arm(t_world_right_target)

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


# ═══════════════════════════════════════════════════════════════
# Hardware Motor Bridge
# ═══════════════════════════════════════════════════════════════

class HardwareMotorBridge:
    """Thread-safe bridge between IK joint angles and DM motor CAN control.

    Runs a background thread at ~1 kHz that reads target joint angles,
    applies smoothing and safety clipping, and sends MIT commands.
    """

    def __init__(
        self,
        kp=None,
        kd=None,
        motor_smoothing=0.3,
        max_step=None,
        device_sn=None,
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

        if len(self._kp) != umc.NUM_MOTORS or len(self._kd) != umc.NUM_MOTORS:
            raise ValueError(
                f"kp/kd must have {umc.NUM_MOTORS} elements "
                f"(6 arm + 1 gripper), got kp={len(self._kp)} kd={len(self._kd)}"
            )

        # Shared state (protected by _lock)
        self._lock = threading.Lock()
        self._target_q = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self._emergency_stop = False
        self._last_sent_q = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self._motor_err = np.zeros(umc.NUM_MOTORS, dtype=np.int32)

        # Thread control
        self._running = threading.Event()
        self._thread = None

        # Build motor init data
        init_data = []
        for _joint_name, can_id, mst_id in umc.ARM_MOTOR_CONFIG:
            init_data.append(DmActData(
                motorType=DM_Motor_Type.DM4310_48V,
                mode=Control_Mode.MIT_MODE,
                can_id=can_id,
                mst_id=mst_id,
            ))
        init_data.append(DmActData(
            motorType=DM_Motor_Type.DM4310_48V,
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

        # Clear motor errors, then enable
        for _ in range(5):
            for can_id in self._can_ids:
                self._control.control_cmd(can_id, 0xFB, 0)
            time.sleep(0.005)
        self._control.enable_all()

        # Build soft limit arrays for fast clipping
        self._limit_lo = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self._limit_hi = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        for i, (joint_name, _, _) in enumerate(umc.ARM_MOTOR_CONFIG):
            lo, hi = umc.SOFT_POSITION_LIMITS[joint_name]
            self._limit_lo[i] = lo
            self._limit_hi[i] = hi
        lo, hi = umc.SOFT_POSITION_LIMITS["gripper"]
        self._limit_lo[umc.ARM_DOF] = lo
        self._limit_hi[umc.ARM_DOF] = hi

        # Per-motor step limit: prevents bug-induced target jumps (rad/tick at 1kHz)
        if max_step is not None:
            self._max_step = np.broadcast_to(np.asarray(max_step, dtype=np.float64),
                                             umc.NUM_MOTORS).copy()
        else:
            self._max_step = np.array(
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10], dtype=np.float64)

        # Track which motors are actually present on the bus
        self._motor_connected = [False] * umc.NUM_MOTORS

        # Safety net: disable motors on exit, even if caller forgets stop()
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
        with self._lock:
            self._target_q[:] = 0.0

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
        self._running.set()
        self._emergency_stop = False
        self._thread = threading.Thread(
            target=self._motor_thread, name="motor-ctrl", daemon=True)
        self._thread.start()
        print(f"[motor] Control thread started at {int(umc.MOTOR_CTRL_FREQ)} Hz")

    def _disable_motors(self):
        """Send motor-disable CAN frames. Safe to call even after close()."""
        try:
            self._control.disable_all()
            time.sleep(0.05)  # let CAN frames flush before USB teardown
            print("[motor] Motors disabled.")
        except Exception as exc:
            print(f"[motor] disable_all error: {exc}", file=sys.stderr)

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._disable_motors()
        # NOTE: _control.close() calls context.destroy() which segfaults
        # due to a libusb async I/O threading bug in libdm_device.so.
        # We skip it — the OS reclaims USB resources on process exit.
        print("[motor] Stopped.")

    def set_target(self, q: np.ndarray):
        q = np.asarray(q, dtype=np.float64).ravel()
        if q.shape[0] != umc.NUM_MOTORS:
            raise ValueError(f"Expected {umc.NUM_MOTORS} targets, got {q.shape[0]}")
        with self._lock:
            np.copyto(self._target_q, q)

    def get_state(self):
        with self._lock:
            return self._last_sent_q.copy(), self._motor_err.copy()

    def emergency_stop(self):
        with self._lock:
            self._emergency_stop = True
            self._target_q[:] = 0.0
        print("[motor] EMERGENCY STOP", file=sys.stderr)

    # ── Motor control thread ──────────────────────────────────

    def _motor_thread(self):
        period = 1.0 / umc.MOTOR_CTRL_FREQ
        time.sleep(0.1)
        self._seed_last_sent_from_motors()
        last_debug_ts = time.monotonic()
        _thread_start = time.monotonic()  # grace period reference

        while self._running.is_set():
            t_start = time.perf_counter()

            estop, target, kp, kd = self._snapshot_targets()
            if estop:
                target = np.zeros(umc.NUM_MOTORS, dtype=np.float64)

            # Per-motor step limit: clamp target deltas to prevent wild jumps
            for i in range(umc.NUM_MOTORS):
                diff = target[i] - self._last_sent_q[i]
                if diff > self._max_step[i]:
                    target[i] = self._last_sent_q[i] + self._max_step[i]
                elif diff < -self._max_step[i]:
                    target[i] = self._last_sent_q[i] - self._max_step[i]

            for i in range(umc.NUM_MOTORS):
                smooth_q = (
                    (1.0 - self._motor_smoothing) * self._last_sent_q[i]
                    + self._motor_smoothing * target[i]
                )
                clipped_q = float(np.clip(smooth_q, self._limit_lo[i], self._limit_hi[i]))
                self._last_sent_q[i] = clipped_q

                if not self._motor_connected[i]:
                    continue
                motor = self._control.getMotor(self._can_ids[i])
                if motor is None:
                    continue
                try:
                    self._control.control_mit(
                        motor, float(kp[i]), float(kd[i]),
                        clipped_q, 0.0, 0.0,
                    )
                except Exception as exc:
                    print(f"[motor] control_mit error motor {i}: {exc}", file=sys.stderr)

            # Error flags (connected motors only)
            for i in range(umc.NUM_MOTORS):
                if not self._motor_connected[i]:
                    continue
                motor = self._control.getMotor(self._can_ids[i])
                if motor is not None:
                    err = motor.Get_err()
                    if err != 0:
                        self._motor_err[i] = int(err)

            # CAN timeout (skip during startup grace period, connected motors only)
            now = time.monotonic()
            if now - _thread_start > 5.0:
                for i in range(umc.NUM_MOTORS):
                    if not self._motor_connected[i]:
                        continue
                    motor = self._control.getMotor(self._can_ids[i])
                    if motor is not None:
                        dt = motor.getTimeInterval()
                        if dt > umc.CAN_TIMEOUT_SEC:
                            print(f"[motor] CAN timeout motor {i} (dt={dt:.3f}s)", file=sys.stderr)
                            self.emergency_stop()
                            break

            if now - last_debug_ts >= 5.0:
                pos_str = " ".join(f"{self._last_sent_q[i]:.3f}" for i in range(umc.ARM_DOF))
                err_str = " ".join(str(self._motor_err[i]) for i in range(umc.NUM_MOTORS))
                print(f"[motor] cmd=[{pos_str}] grip={self._last_sent_q[umc.ARM_DOF]:.4f} err=[{err_str}]")
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
                self._kp.copy(),
                self._kd.copy(),
            )

    def _seed_last_sent_from_motors(self):
        for i in range(umc.NUM_MOTORS):
            motor = self._control.getMotor(self._can_ids[i])
            if motor is not None:
                pos = motor.Get_Position()
                self._last_sent_q[i] = pos
                self._target_q[i] = pos  # hold current position until IK takes over
                # Mark as connected if CAN frames have been received recently
                dt = motor.getTimeInterval()
                self._motor_connected[i] = dt > 0.0 and dt < 5.0
        connected = [f"{umc.ARM_MOTOR_CONFIG[i][0] if i < umc.ARM_DOF else 'gripper'}"
                      for i in range(umc.NUM_MOTORS) if self._motor_connected[i]]
        print(f"[motor] Connected: {connected if connected else '(none)'}")
        print(f"[motor] Seeded initial positions: {self._last_sent_q}")

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

    # Teleop flags
    parser.add_argument("--ngrok", action="store_true")
    parser.add_argument("--local-cert", action="store_true")
    parser.add_argument("--cert-file", type=str, default="./cert.pem")
    parser.add_argument("--key-file", type=str, default="./key.pem")
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
    parser.add_argument("--robot-x", type=float, default=0.0)
    parser.add_argument("--robot-y", type=float, default=0.0)
    parser.add_argument("--robot-z", type=float, default=0.0)
    parser.add_argument("--base-roll-deg", type=float, default=0.0)
    parser.add_argument("--base-pitch-deg", type=float, default=0.0)
    parser.add_argument("--base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--calibration-delay-sec", type=float, default=5.0)
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument("--gripper-open-value", type=float, default=5.0)
    parser.add_argument("--gripper-close-value", type=float, default=0.0)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.25)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.75)
    parser.add_argument("--gripper-smoothing", type=float, default=0.3)
    parser.add_argument("--arm-smoothing", type=float, default=1.0)
    parser.add_argument("--thumb-tip-index", type=int, default=4)
    parser.add_argument("--index-tip-index", type=int, default=9)
    return parser.parse_args()


def create_teleop(args):
    ngrok_mode = True if not args.local_cert else False
    if args.ngrok:
        ngrok_mode = True
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
    args = parse_args()

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
    if args.motor_enable:
        hw_bridge = HardwareMotorBridge(
            kp=args.kp if args.kp is not None else None,
            kd=args.kd if args.kd is not None else None,
            motor_smoothing=args.motor_smoothing,
            device_sn=args.device_sn,
        )
        hw_bridge.start()

    print("\n[teleop] ======== UPOO Arm Hardware Teleop v2 (servo mode) ========")
    print("[teleop] 6-DOF, 右手控制")
    print(f"[teleop] 硬件电机: {'ENABLED' if hw_bridge else 'DISABLED (sim only)'}")
    print("[teleop] 仿真模式: servo (ctrl-driven PD actuators) + real-time steps")
    print("[teleop] P=标定  R=复位杯子  E=急停")
    if hw_bridge:
        print(f"[teleop] kp={hw_bridge._kp}  kd={hw_bridge._kd}")
    print()

    try:
        # Init stereo before viewer (GLContext needs active context)
        sim._init_stereo()

        viewer = None
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

        def _stdin_reader():
            while not stdin_stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.5)
                if r:
                    line = sys.stdin.readline().strip().lower()
                    if line == 'p':
                        sim._request_calibration()
                    elif line == 'r':
                        sim._reset_cup()
                    elif line == 'e':
                        if hw_bridge:
                            hw_bridge.emergency_stop()
                        print("[stdin] Emergency stop triggered")

        stdin_thread = threading.Thread(target=_stdin_reader, daemon=True)
        stdin_thread.start()

        print("\nMuJoCo viewer open. P=calibrate  R=reset cup  E=estop\n")

        while viewer is None or viewer.is_running():
            tic = time.time()

            # ① VR
            t_vuer_head, t_vuer_right, right_lm = teleop.step()

            # ② IK
            q_result, dq_norm = sim.compute_ik(t_vuer_head, t_vuer_right)

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
                if right_lm is not None:
                    if not sim._gripper_landmarks_ready:
                        sim.gripper_cmd = float(q_smoothed[sim.finger_left_qpos])
                        sim._gripper_landmarks_ready = True
                        print("[gripper] landmarks ready")
                    sim.gripper_cmd = sim._gripper_command_from_landmarks(right_lm)

            # ④ Hardware motor command
            if hw_bridge and sim.calibration_ready:
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
            if now - last_debug_ts >= 2.0:
                js = {}
                for name in sim.ARM_JOINT_NAMES:
                    try:
                        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                        js[name] = float(q_smoothed[sim.model.jnt_qposadr[jid]])
                    except Exception:
                        js[name] = 0.0
                rh = t_vuer_right
                print(f"[DIAG] RH=({rh[0,3]:.2f},{rh[1,3]:.2f},{rh[2,3]:.2f}) "
                      f"dq={dq_norm:.3f} "
                      + " ".join(f"j{i}={js.get(n,0):.3f}"
                                 for i, n in enumerate(sim.ARM_JOINT_NAMES, 1)))
                if hw_bridge:
                    pos, errs = hw_bridge.get_state()
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
