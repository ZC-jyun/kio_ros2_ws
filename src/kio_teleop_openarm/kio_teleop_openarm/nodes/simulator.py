#!/usr/bin/env python3
"""simulator node — MuJoCo physics, stereo rendering, joint state feedback."""

import os
import sys
import time
import threading
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_srvs.srv import Trigger

# Path helpers for openarm_mujoco
_DEPLOY_DIR = Path("/home/kiorobot/kio_robot_zzc/openarm-main/teleop_deploy")
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

import mujoco
from mujoco import viewer as mujoco_viewer  # lazy module — explicit import required
import openarm_mujoco.v2 as openarm_mujoco

from kio_teleop_openarm.lib.transforms import (
    euler_xyz_deg_to_quat_xyzw, quat_xyzw_to_matrix, make_transform, project_to_rotation_matrix)
from kio_teleop_openarm.lib.stereo_renderer import (
    make_stereo_cameras, render_stereo, set_camera_free_pose)

# Joint name constants (from original)
LEFT_ARM_JOINT_NAMES = [
    "upoo_left_Base_J01", "upoo_left_J02", "upoo_left_J03",
    "upoo_left_J04", "upoo_left_J05", "upoo_left_J06",
]
RIGHT_ARM_JOINT_NAMES = [
    "upoo_right_Base_J01", "upoo_right_J02", "upoo_right_J03",
    "upoo_right_J04", "upoo_right_J05", "upoo_right_J06",
]
LEFT_FINGER_JOINT_NAMES = ["upoo_left_finger_left_joint", "upoo_left_finger_right_joint"]
RIGHT_FINGER_JOINT_NAMES = ["upoo_right_finger_left_joint", "upoo_right_finger_right_joint"]
LEFT_EE_BODY_NAME = "upoo_left_Link_06"
RIGHT_EE_BODY_NAME = "upoo_right_Link_06"
BODY_LINK_NAME = "upoo_left_base_link"
ALL_ARM_JOINTS = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
ALL_FINGER_JOINTS = LEFT_FINGER_JOINT_NAMES + RIGHT_FINGER_JOINT_NAMES
ALL_JOINT_NAMES = ALL_ARM_JOINTS + ALL_FINGER_JOINTS


class SimulatorNode(Node):
    def __init__(self):
        super().__init__("simulator")

        # ── Parameters ──
        self.declare_parameter("model_type", "upoo_bimanual")
        self.declare_parameter("robot_x", 0.0)
        self.declare_parameter("robot_y", 0.0)
        self.declare_parameter("robot_z", 0.0)
        self.declare_parameter("base_roll_deg", 0.0)
        self.declare_parameter("base_pitch_deg", 0.0)
        self.declare_parameter("base_yaw_deg", 0.0)
        self.declare_parameter("gripper_open_value", 0.044)
        self.declare_parameter("stereo_width", 1280)
        self.declare_parameter("stereo_height", 960)
        self.declare_parameter("ipd", 0.064)
        self.declare_parameter("head_camera_y_offset", 0.0)
        self.declare_parameter("head_camera_z_offset", 0.08)
        self.declare_parameter("enable_viewer", True)
        self.declare_parameter("sim_timestep_override", -1.0)

        # Load params
        model_type = self.get_parameter("model_type").value
        robot_base_xyz = np.array([
            self.get_parameter("robot_x").value,
            self.get_parameter("robot_y").value,
            self.get_parameter("robot_z").value,
        ], dtype=np.float32)
        base_roll = self.get_parameter("base_roll_deg").value
        base_pitch = self.get_parameter("base_pitch_deg").value
        base_yaw = self.get_parameter("base_yaw_deg").value
        self.gripper_open_value = self.get_parameter("gripper_open_value").value
        sw = self.get_parameter("stereo_width").value
        sh = self.get_parameter("stereo_height").value
        ipd = self.get_parameter("ipd").value
        head_y_off = self.get_parameter("head_camera_y_offset").value
        head_z_off = self.get_parameter("head_camera_z_offset").value
        self.enable_viewer = self.get_parameter("enable_viewer").value
        sim_override = self.get_parameter("sim_timestep_override").value

        # ── Load MuJoCo model ──
        model_loaders = {
            "upoo_cell": openarm_mujoco.openarm_upoo_cell_xml,
            "upoo_bimanual": openarm_mujoco.openarm_upoo_bimanual_xml,
        }
        if model_type in model_loaders:
            xml_path = model_loaders[model_type]()
        elif os.path.exists(model_type):
            xml_path = os.path.abspath(model_type)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")
        self.get_logger().info(f"Loading model: {xml_path}")

        with open(xml_path, 'r') as f:
            xml_str = f.read()

        # Scene defaults (from original)
        scene_defaults = """
    <default class="scene_collision">
      <geom contype="1" conaffinity="1" condim="3" solref="0.02 1" solimp="0.9 0.95 0.001" friction="0.5 0.01 0.01"/>
    </default>
    <default class="cup_collision">
      <geom contype="1" conaffinity="1" condim="6" solref="0.02 1" solimp="0.9 0.95 0.001" friction="1.5 0.3 0.05"/>
    </default>
"""
        xml_str = xml_str.replace("<default>", "<default>" + scene_defaults)

        if model_type != "upoo_cell":
            scene_injects = self._build_scene()
            xml_str = xml_str.replace("<worldbody>", "<worldbody>" + scene_injects)

        import tempfile
        xml_dir = os.path.dirname(xml_path)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', dir=xml_dir, delete=False) as tf:
            tf.write(xml_str)
            tmp_xml_path = tf.name
        try:
            self.model = mujoco.MjModel.from_xml_path(tmp_xml_path)
        finally:
            os.unlink(tmp_xml_path)

        if sim_override > 0:
            self.model.opt.timestep = sim_override

        self._configure_contacts()

        self.data = mujoco.MjData(self.model)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        # ── Joint indices ──
        def _qpos_idx(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_qposadr[jid]

        self.left_qpos = np.array([_qpos_idx(n) for n in LEFT_ARM_JOINT_NAMES], dtype=int)
        self.right_qpos = np.array([_qpos_idx(n) for n in RIGHT_ARM_JOINT_NAMES], dtype=int)
        self.left_finger_qpos = np.array([_qpos_idx(n) for n in LEFT_FINGER_JOINT_NAMES], dtype=int)
        self.right_finger_qpos = np.array([_qpos_idx(n) for n in RIGHT_FINGER_JOINT_NAMES], dtype=int)
        self.all_arm_qpos = np.concatenate([self.left_qpos, self.right_qpos])
        self.all_finger_qpos = np.concatenate([self.left_finger_qpos, self.right_finger_qpos])

        # EE body IDs
        self.left_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_BODY_NAME)
        self.right_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY_NAME)

        # Finger actuators
        self.left_finger_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "upoo_left_finger_ctrl")
        self.right_finger_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "upoo_right_finger_ctrl")

        # ── Set initial qpos from home_pose.yaml ──
        self._load_home_pose()

        # Cup
        self._cup_qpos_adr = -1
        self._cup_init_qpos = None
        self._cup_body_id = -1
        try:
            cup_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
            if cup_body >= 0:
                self._cup_body_id = cup_body
                cup_jnt = self.model.body_jntadr[cup_body]
                self._cup_qpos_adr = self.model.jnt_qposadr[cup_jnt]
                self.data.qpos[self._cup_qpos_adr:self._cup_qpos_adr + 7] = np.array(
                    [0.3, 0.0, 0.49, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                self._cup_init_qpos = self.data.qpos[self._cup_qpos_adr:self._cup_qpos_adr + 7].copy()
                self.get_logger().info(
                    f"Cup placed at {self._cup_init_qpos[:3]}, adr={self._cup_qpos_adr}")
        except Exception:
            pass

        # Transforms
        self.base_quat_xyzw = euler_xyz_deg_to_quat_xyzw(base_roll, base_pitch, base_yaw)
        self.t_world_robotbase = make_transform(robot_base_xyz, quat_xyzw_to_matrix(self.base_quat_xyzw))
        self.t_robotbase_world = np.linalg.inv(self.t_world_robotbase).astype(np.float32)

        # Body link for camera reference
        self.body_link_id = None
        try:
            self.body_link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BODY_LINK_NAME)
        except Exception:
            pass

        # Init ctrl
        self._sync_ctrl_from_qpos()
        self._apply_gripper_ctrl(np.array([0.044, 0.044]))

        mujoco.mj_forward(self.model, self.data)

        # ── Stereo rendering ──
        self.sw, self.sh = sw, sh
        self.ipd = ipd
        self._hidden_glfw_window = None
        self._init_stereo()

        # ── Viewer ──
        self.viewer = None
        if self.enable_viewer:
            try:
                self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
                self.viewer.cam.lookat[:] = self.model.stat.center
                self.viewer.cam.distance = self.model.stat.extent * 0.8
                self.viewer.cam.azimuth = self.model.vis.global_.azimuth
                self.viewer.cam.elevation = self.model.vis.global_.elevation
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
                self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
                self.get_logger().info("Passive viewer launched")
            except Exception as e:
                self.get_logger().warn(f"Viewer unavailable: {e}")

        # ── ROS2 interface ──
        self.joint_target_sub = self.create_subscription(
            JointState, "/joint_target", self._joint_target_cb, 10)
        self.joint_state_pub = self.create_publisher(JointState, "/joint_state", 10)
        self.stereo_pub = self.create_publisher(Image, "/stereo_image", 10)

        self._latest_target = None
        self._last_real_time = None
        self._last_n_steps = 0
        self._paused = False

        # Lock for joint target access
        self._target_lock = threading.Lock()

        # Services
        self.create_service(Trigger, "/reset_cup", self._reset_cup_srv_cb)
        self.create_service(Trigger, "/simulator/toggle_pause", self._toggle_pause_srv_cb)

        # Timer (matches vsync, ~60 Hz)
        self._timer = self.create_timer(1.0 / 60.0, self._sim_step)

        self.get_logger().info("simulator node started")

    def _load_home_pose(self):
        """Load initial joint positions from config/home_pose.yaml."""
        import yaml
        config_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "share", "kio_teleop_openarm", "config")
        yaml_path = os.path.join(config_dir, "home_pose.yaml")
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            home_pose = config.get("home_pose", {})
            if home_pose:
                for jname, jval in home_pose.items():
                    try:
                        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                        self.data.qpos[self.model.jnt_qposadr[jid]] = float(jval)
                    except Exception:
                        self.get_logger().warn(f"home_pose: unknown joint '{jname}'")
                self.get_logger().info(f"Loaded home_pose from {yaml_path}")
                return
        except Exception as e:
            self.get_logger().warn(f"Could not read home_pose.yaml: {e}")
        # Fallback: all zeros + grippers open
        for idx in self.all_arm_qpos:
            self.data.qpos[idx] = 0.0
        for idx in self.left_finger_qpos:
            self.data.qpos[idx] = self.gripper_open_value
        for idx in self.right_finger_qpos:
            self.data.qpos[idx] = self.gripper_open_value

    # ═══════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════

    def _build_scene(self):
        lines = []
        grid_z = 0.011
        grid_half = 1.5
        grid_spacing = 0.3
        for i in np.arange(-grid_half, grid_half + 0.001, grid_spacing):
            lines.append(
                f'    <geom name="grid_x{i:.1f}" type="box" size="1.5 0.003 0.001" '
                f'pos="0 {i:.2f} {grid_z}" rgba="0.2 0.4 0.8 0.5" contype="0" conaffinity="0"/>')
            lines.append(
                f'    <geom name="grid_y{i:.1f}" type="box" size="0.003 1.5 0.001" '
                f'pos="{i:.2f} 0 {grid_z}" rgba="0.2 0.4 0.8 0.5" contype="0" conaffinity="0"/>')
        axis_z = 0.012
        lines.append(
            f'    <geom name="axis_x" type="box" size="1.5 0.005 0.002" '
            f'pos="0.75 0 {axis_z}" rgba="1 0.2 0.2 0.9" contype="0" conaffinity="0"/>')
        lines.append(
            f'    <geom name="axis_y" type="box" size="0.005 1.5 0.002" '
            f'pos="0 0.75 {axis_z}" rgba="0.2 1 0.2 0.9" contype="0" conaffinity="0"/>')
        lines.append(
            '    <body name="table" pos="0.3 0 0.24">'
            '<geom name="table_top" type="box" size="0.25 0.2 0.2" rgba="0.5 0.5 0.5 1" class="scene_collision"/>'
            '</body>')
        lines.append(
            '    <body name="cup" pos="0.3 0 0.49">'
            '<freejoint/>'
            '<geom name="cup_col" type="box" size="0.025 0.025 0.025" mass="0.05" '
            'rgba="0.9 0.15 0.15 1" class="cup_collision"/>'
            '</body>')
        return "\n".join(lines)

    def _configure_contacts(self):
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "finger" in name and "col" in name and "pad" not in name:
                self.model.geom_contype[i] = 1
                self.model.geom_conaffinity[i] = 1
                self.model.geom_condim[i] = 6
                self.model.geom_friction[i, :] = [3.0, 1.0, 0.1]
                self.model.geom_solref[i, :] = [0.008, 1.0]
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "cup" in name and "col" in name:
                self.model.geom_friction[i, :] = [3.0, 1.0, 0.1]
                self.model.geom_solref[i, :] = [0.008, 1.0]
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "col" in name:
                self.model.geom_contype[i] = 3
                self.model.geom_conaffinity[i] = 3
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and "finger" in name:
                self.model.actuator_gainprm[i, 0] = max(self.model.actuator_gainprm[i, 0], 200.0)
                self.model.actuator_forcelimited[i] = 1
                self.model.actuator_forcerange[i, :] = [-30.0, 30.0]

    def _sync_ctrl_from_qpos(self):
        skip = {self.left_finger_act, self.right_finger_act}
        for i in range(self.model.nu):
            if i in skip:
                continue
            jid = self.model.actuator_trnid[i, 0]
            if jid >= 0:
                self.data.ctrl[i] = self.data.qpos[self.model.jnt_qposadr[jid]]

    def _apply_gripper_ctrl(self, gripper_cmds):
        """gripper_cmds: [left, right]"""
        self.data.ctrl[self.left_finger_act] = gripper_cmds[0]
        self.data.ctrl[self.right_finger_act] = gripper_cmds[1]

    def _init_stereo(self):
        import glfw as _glfw
        if not _glfw.init():
            raise RuntimeError("GLFW init failed")
        _glfw.window_hint(_glfw.VISIBLE, _glfw.FALSE)
        _glfw.window_hint(_glfw.DOUBLEBUFFER, _glfw.TRUE)
        self._hidden_glfw_window = _glfw.create_window(self.sw, self.sh, "hidden_stereo", None, None)
        _glfw.make_context_current(self._hidden_glfw_window)

        lookat = np.array([0.0, 0.0, 0.85], dtype=np.float32)
        (self._cam_left, self._cam_right, self._r_left,
         self._r_right, self._vp) = make_stereo_cameras(
            self.model, self.scene, cam_lookat=lookat, cam_distance=1.0,
            cam_azimuth=0.0, cam_elevation=-25.0,
            width=self.sw, height=self.sh, ipd=self.ipd)
        self.get_logger().info("Stereo renderer initialized")

    def _set_head_tracked_cameras(self, t_world_head):
        """Update stereo camera poses from head transform (plugin from controller)."""
        pass  # static cameras for now — extend with head tracking if needed

    # ═══════════════════════════════════════════════
    # ROS2 callbacks
    # ═══════════════════════════════════════════════

    def _joint_target_cb(self, msg: JointState):
        with self._target_lock:
            self._latest_target = msg

    def _reset_cup_srv_cb(self, request, response):
        if self._cup_qpos_adr >= 0 and self._cup_init_qpos is not None:
            self.data.qpos[self._cup_qpos_adr:self._cup_qpos_adr + 7] = self._cup_init_qpos.copy()
            response.success = True
            response.message = "Cup reset"
            self.get_logger().info("Cup reset")
        else:
            response.success = False
            response.message = "No cup in scene"
        return response

    def _toggle_pause_srv_cb(self, request, response):
        self._paused = not self._paused
        response.success = True
        response.message = f"Simulation {'PAUSED' if self._paused else 'RESUMED'}"
        self.get_logger().info(response.message)
        return response

    def _sim_step(self):
        """Main simulation loop."""

        # Always sync viewer so user can interact with sliders
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        if self._paused:
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            for names, qpos_indices in [(LEFT_ARM_JOINT_NAMES, self.left_qpos),
                                         (RIGHT_ARM_JOINT_NAMES, self.right_qpos),
                                         (LEFT_FINGER_JOINT_NAMES, self.left_finger_qpos),
                                         (RIGHT_FINGER_JOINT_NAMES, self.right_finger_qpos)]:
                js.name.extend(names)
                js.position.extend([float(self.data.qpos[i]) for i in qpos_indices])
            self.joint_state_pub.publish(js)
            return

        # Apply latest joint target
        with self._target_lock:
            target = self._latest_target
        if target is not None:
            self._apply_joint_target(target)

        # Real-time stepping
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
        self._last_n_steps = n_steps

        # Render stereo
        import glfw as _glfw
        _glfw.make_context_current(self._hidden_glfw_window)
        left_img, right_img = render_stereo(
            self.model, self.data, self.scene,
            self._cam_left, self._cam_right,
            self._r_left, self._r_right, self._vp)

        # Publish stereo image (side-by-side)
        stereo = np.ascontiguousarray(np.hstack((left_img.copy(), right_img.copy())))
        img_msg = Image()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.height = stereo.shape[0]
        img_msg.width = stereo.shape[1]
        img_msg.encoding = "rgb8"
        img_msg.is_bigendian = False
        img_msg.step = stereo.shape[1] * 3
        img_msg.data = stereo.tobytes()
        self.stereo_pub.publish(img_msg)

        # Publish joint state
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        for names, qpos_indices in [(LEFT_ARM_JOINT_NAMES, self.left_qpos),
                                     (RIGHT_ARM_JOINT_NAMES, self.right_qpos),
                                     (LEFT_FINGER_JOINT_NAMES, self.left_finger_qpos),
                                     (RIGHT_FINGER_JOINT_NAMES, self.right_finger_qpos)]:
            js.name.extend(names)
            js.position.extend([float(self.data.qpos[i]) for i in qpos_indices])
        self.joint_state_pub.publish(js)

    def _apply_joint_target(self, msg: JointState):
        """Apply joint_target to arm ctrl (position servo mode)."""
        if not msg.name or len(msg.name) != len(msg.position):
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        skip = {self.left_finger_act, self.right_finger_act}
        for i in range(self.model.nu):
            if i in skip:
                continue
            jid = self.model.actuator_trnid[i, 0]
            if jid >= 0:
                jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if jname and jname in name_to_pos:
                    self.data.ctrl[i] = name_to_pos[jname]
        # Finger actuators: match by name in JointState
        for finger_name, act_id in [
            ("upoo_left_finger_left_joint", self.left_finger_act),
            ("upoo_right_finger_left_joint", self.right_finger_act),
        ]:
            if finger_name in name_to_pos:
                self.data.ctrl[act_id] = name_to_pos[finger_name]

    def close(self):
        if self._hidden_glfw_window is not None:
            import glfw as _glfw
            _glfw.destroy_window(self._hidden_glfw_window)
            _glfw.terminate()
            self._hidden_glfw_window = None
        if self.viewer is not None:
            self.viewer.close()


def main(args=None):
    rclpy.init(args=args)
    node = SimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
