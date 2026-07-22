#!/usr/bin/env python3
"""controller node — DLS IK solver, calibration, auto grasp, gripper control."""

import os
import sys
import time
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from kio_teleop_openarm.srv import GripperCmd

_DEPLOY_DIR = Path("/home/kiorobot/kio_robot_zzc/openarm-main/teleop_deploy")
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

import mujoco
import openarm_mujoco.v2 as openarm_mujoco

from kio_teleop_openarm.lib.transforms import (
    euler_xyz_deg_to_quat_xyzw, quat_xyzw_to_matrix, make_transform, project_to_rotation_matrix)
from kio_teleop_openarm.lib.ik_solver import IKSolver, target_pose_from_hand
from kio_teleop_openarm.lib.calibration import Calibrator
from kio_teleop_openarm.lib.auto_grasp import AutoGrasp
from kio_teleop_openarm.lib.gripper import gripper_command_from_landmarks

# Reuse joint name constants
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
ALL_ARM_JOINTS = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
ALL_FINGER_JOINTS = LEFT_FINGER_JOINT_NAMES + RIGHT_FINGER_JOINT_NAMES
ALL_JOINTS = ALL_ARM_JOINTS + ALL_FINGER_JOINTS


def pose_stamped_to_mat4(msg: PoseStamped):
    """Convert PoseStamped to 4x4 transform matrix."""
    from pytransform3d import rotations
    qx, qy, qz, qw = (msg.pose.orientation.x, msg.pose.orientation.y,
                       msg.pose.orientation.z, msg.pose.orientation.w)
    quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
    rmat = rotations.matrix_from_quaternion(quat_wxyz)
    pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    return make_transform(pos, rmat)


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller")

        # ── Parameters ──
        self._declare_all_params()
        p = lambda n: self.get_parameter(n).value

        model_type = p("model_type")
        self.position_gain = p("position_gain")
        self.orientation_gain = p("orientation_gain")
        self.orientation_weight = p("orientation_weight")
        self.damping = p("damping")
        self.max_dq = p("max_dq")
        self.ik_max_iters = p("ik_max_iters")
        self.ik_tolerance = p("ik_tolerance")
        self.position_scale = p("position_scale")
        self.arm_smoothing = p("arm_smoothing")
        self.enable_gripper = p("enable_gripper")
        self.gripper_open_value = p("gripper_open_value")
        self.gripper_close_value = p("gripper_close_value")
        self.gripper_close_threshold = p("gripper_close_threshold")
        self.gripper_open_threshold = p("gripper_open_threshold")
        self.gripper_smoothing = p("gripper_smoothing")
        self.thumb_tip_index = p("thumb_tip_index")
        self.index_tip_index = p("index_tip_index")
        self.calibration_delay_sec = p("calibration_delay_sec")
        joint_weights = p("joint_weights")
        if joint_weights is None or joint_weights == "null":
            joint_weights = None

        robot_base_xyz = np.array([p("robot_x"), p("robot_y"), p("robot_z")], dtype=np.float32)
        base_quat_xyzw = euler_xyz_deg_to_quat_xyzw(p("base_roll_deg"), p("base_pitch_deg"), p("base_yaw_deg"))

        # ── Load MuJoCo model (kinematics only) ──
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
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ── Joint indices ──
        def _qpos_idx(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_qposadr[jid]

        def _dof_idx(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_dofadr[jid]

        self.left_qpos = np.array([_qpos_idx(n) for n in LEFT_ARM_JOINT_NAMES], dtype=int)
        self.right_qpos = np.array([_qpos_idx(n) for n in RIGHT_ARM_JOINT_NAMES], dtype=int)
        self.left_dofs = np.array([_dof_idx(n) for n in LEFT_ARM_JOINT_NAMES], dtype=int)
        self.right_dofs = np.array([_dof_idx(n) for n in RIGHT_ARM_JOINT_NAMES], dtype=int)
        self.left_finger_qpos = np.array([_qpos_idx(n) for n in LEFT_FINGER_JOINT_NAMES], dtype=int)
        self.right_finger_qpos = np.array([_qpos_idx(n) for n in RIGHT_FINGER_JOINT_NAMES], dtype=int)
        self.arm_qpos_indices = np.concatenate([self.left_qpos, self.right_qpos])

        # EE body IDs
        self.left_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_BODY_NAME)
        self.right_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY_NAME)

        # ── Set home qpos from home_pose.yaml ──
        self._load_home_pose()
        mujoco.mj_forward(self.model, self.data)

        # ── IK Solvers ──
        ik_common = dict(
            position_gain=self.position_gain, orientation_gain=self.orientation_gain,
            orientation_weight=self.orientation_weight, damping=self.damping,
            max_dq=self.max_dq, ik_max_iters=self.ik_max_iters,
            ik_tolerance=self.ik_tolerance,
        )
        self.left_ik = IKSolver(
            self.model, self.data, self.left_body, self.left_dofs, self.left_qpos,
            joint_weights=joint_weights, **ik_common)
        self.right_ik = IKSolver(
            self.model, self.data, self.right_body, self.right_dofs, self.right_qpos,
            joint_weights=joint_weights, **ik_common)

        # ── Transforms ──
        self.t_world_robotbase = make_transform(robot_base_xyz, quat_xyzw_to_matrix(base_quat_xyzw))
        self.t_robotbase_world = np.linalg.inv(self.t_world_robotbase).astype(np.float32)

        # Body link for head height estimation
        robot_head_height = 0.0
        robot_head_pos_robotbase = np.array([-1.0, 0, 0.0], dtype=np.float32)
        try:
            body_link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                             "upoo_left_base_link")
            body_pos_world = self.data.xpos[body_link_id]
            body_pos_robotbase = (
                self.t_robotbase_world @ np.r_[body_pos_world, 1.0].astype(np.float32))[:3]
            robot_head_pos_robotbase = body_pos_robotbase.copy()
            robot_head_height = float(body_pos_robotbase[2])
        except Exception:
            pass

        # ── Calibrator ──
        static_cam_lookat = np.array([0.0, 0.0, 0.85], dtype=np.float32)
        self.calibrator = Calibrator(
            t_world_robotbase=self.t_world_robotbase,
            t_robotbase_world=self.t_robotbase_world,
            robot_head_pos_robotbase=robot_head_pos_robotbase,
            static_cam_lookat=static_cam_lookat,
            static_cam_distance=1.0,
            static_cam_elevation=-25.0,
            static_cam_azimuth=0.0,
            calibration_delay_sec=self.calibration_delay_sec,
        )

        # ── Auto Grasp ──
        self.auto_grasp = AutoGrasp(
            gripper_open_value=self.gripper_open_value,
            gripper_close_value=self.gripper_close_value,
        )

        # ── Gripper state ──
        self.left_gripper_cmd = self.gripper_open_value
        self.right_gripper_cmd = self.gripper_open_value
        self._gripper_metrics = {}
        self._gripper_manual_override = None  # None=pinch, True=open, False=close

        # ── Smoothed qpos ──
        self.q_smoothed = self.data.qpos.copy()

        # ── VR data cache ──
        self._t_vuer_head = None
        self._t_vuer_left = None
        self._t_vuer_right = None
        self._left_landmarks = None
        self._right_landmarks = None
        self._left_pinch = 0.0
        self._right_pinch = 0.0
        self._data_lock = None  # set after import

        # ── ROS2 Subscribers ──
        self.head_sub = self.create_subscription(
            PoseStamped, "/vr/head_pose", self._head_cb, 10)
        self.left_hand_sub = self.create_subscription(
            PoseStamped, "/vr/left_hand_pose", self._left_hand_cb, 10)
        self.right_hand_sub = self.create_subscription(
            PoseStamped, "/vr/right_hand_pose", self._right_hand_cb, 10)
        self.left_lm_sub = self.create_subscription(
            Float32MultiArray, "/vr/landmarks_left", self._left_lm_cb, 10)
        self.right_lm_sub = self.create_subscription(
            Float32MultiArray, "/vr/landmarks_right", self._right_lm_cb, 10)
        self.left_pinch_sub = self.create_subscription(
            Float32, "/vr/left_pinch", self._left_pinch_cb, 10)
        self.right_pinch_sub = self.create_subscription(
            Float32, "/vr/right_pinch", self._right_pinch_cb, 10)

        # Publisher
        self.joint_target_pub = self.create_publisher(JointState, "/joint_target", 10)

        # Services
        self.create_service(Trigger, "/calibrate", self._calibrate_srv_cb)
        self.create_service(Trigger, "/auto_grasp", self._auto_grasp_srv_cb)
        self.create_service(GripperCmd, "/set_gripper", self._gripper_srv_cb)

        # Timer: IK + publish at ~60 Hz
        self._timer = self.create_timer(1.0 / 60.0, self._control_loop)

        self.get_logger().info("controller node started")

    # ── Parameter declarations ──
    def _declare_all_params(self):
        self.declare_parameter("model_type", "upoo_bimanual")
        self.declare_parameter("position_gain", 1.0)
        self.declare_parameter("orientation_gain", 0.8)
        self.declare_parameter("orientation_weight", 1.0)
        self.declare_parameter("damping", 0.05)
        self.declare_parameter("max_dq", 0.05)
        self.declare_parameter("ik_max_iters", 2)
        self.declare_parameter("ik_tolerance", 0.001)
        self.declare_parameter("joint_weights", value=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("position_scale", 1.0)
        self.declare_parameter("robot_x", 0.0)
        self.declare_parameter("robot_y", 0.0)
        self.declare_parameter("robot_z", 0.0)
        self.declare_parameter("base_roll_deg", 0.0)
        self.declare_parameter("base_pitch_deg", 0.0)
        self.declare_parameter("base_yaw_deg", 0.0)
        self.declare_parameter("calibration_delay_sec", 5.0)
        self.declare_parameter("enable_gripper", True)
        self.declare_parameter("gripper_open_value", 0.044)
        self.declare_parameter("gripper_close_value", 0.0)
        self.declare_parameter("gripper_close_threshold", 0.05)
        self.declare_parameter("gripper_open_threshold", 0.95)
        self.declare_parameter("gripper_smoothing", 0.15)
        self.declare_parameter("arm_smoothing", 1.0)
        self.declare_parameter("thumb_tip_index", 4)
        self.declare_parameter("index_tip_index", 9)

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
        for idx in self.arm_qpos_indices:
            self.data.qpos[idx] = 0.0
        for idx in self.left_finger_qpos:
            self.data.qpos[idx] = self.gripper_open_value
        for idx in self.right_finger_qpos:
            self.data.qpos[idx] = self.gripper_open_value

    # ── VR data callbacks ──
    def _head_cb(self, msg): self._t_vuer_head = pose_stamped_to_mat4(msg)
    def _left_hand_cb(self, msg): self._t_vuer_left = pose_stamped_to_mat4(msg)
    def _right_hand_cb(self, msg): self._t_vuer_right = pose_stamped_to_mat4(msg)
    def _left_lm_cb(self, msg):
        self._left_landmarks = np.array(msg.data, dtype=np.float32).reshape(-1, 3)
    def _right_lm_cb(self, msg):
        self._right_landmarks = np.array(msg.data, dtype=np.float32).reshape(-1, 3)
    def _left_pinch_cb(self, msg): self._left_pinch = float(msg.data)
    def _right_pinch_cb(self, msg): self._right_pinch = float(msg.data)

    # ── Body pose helper ──
    def _get_body_pose_world(self, side):
        body_id = self.left_body if side == "left" else self.right_body
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
        return make_transform(pos, quat_xyzw_to_matrix(quat_xyzw))

    # ── Service callbacks ──
    def _calibrate_srv_cb(self, request, response):
        msg = self.calibrator.request()
        response.success = True
        response.message = msg
        self.get_logger().info(msg)
        return response

    def _auto_grasp_srv_cb(self, request, response):
        if not self.calibrator.calibration_ready:
            response.success = False
            response.message = "Please calibrate first (P)"
            return response
        cup_body_id = -1
        try:
            cup_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        except Exception:
            pass
        msg = self.auto_grasp.advance(
            cup_body_id, self.left_body, self.right_body,
            lambda bid: self.data.xpos[bid],
            lambda bid: self.data.xquat[bid],
        )
        response.success = True
        response.message = msg
        self.get_logger().info(msg)
        return response

    def _gripper_srv_cb(self, request, response):
        cmd = request.command
        if cmd == "toggle":
            if self._gripper_manual_override is None:
                self._gripper_manual_override = False  # close
            elif self._gripper_manual_override is False:
                self._gripper_manual_override = True   # open
            else:
                self._gripper_manual_override = None   # pinch
            if self._gripper_manual_override is not None:
                self.left_gripper_cmd = (self.gripper_open_value
                                         if self._gripper_manual_override
                                         else self.gripper_close_value)
                self.right_gripper_cmd = self.left_gripper_cmd
            state = {None: "pinch", True: "OPEN", False: "CLOSE"}[self._gripper_manual_override]
            self.get_logger().info(f"Gripper → {state} ({self.left_gripper_cmd:.4f})")
        elif cmd == "close_step":
            self._gripper_manual_override = False
            step = 0.005
            self.left_gripper_cmd = max(self.gripper_close_value, self.left_gripper_cmd - step)
            self.right_gripper_cmd = self.left_gripper_cmd
            self.get_logger().info(f"Gripper close step → {self.left_gripper_cmd:.4f}")
        elif cmd == "open_step":
            self._gripper_manual_override = False
            step = 0.005
            self.left_gripper_cmd = min(self.gripper_open_value, self.left_gripper_cmd + step)
            self.right_gripper_cmd = self.left_gripper_cmd
            self.get_logger().info(f"Gripper open step → {self.left_gripper_cmd:.4f}")
        response.success = True
        return response

    # ── Main control loop ──
    def _control_loop(self):
        cal = self.calibrator

        if self._t_vuer_head is not None:
            # Calibration check (only when VR data available)
            status = cal.maybe_capture(
                self._t_vuer_head, self._t_vuer_left, self._t_vuer_right,
                lambda side: self._get_body_pose_world(side))
            if status:
                self.get_logger().info(status)

        if not cal.calibration_ready:
            # Publish home joint position even before calibration
            self._publish_joint_target()
            return

        # ── IK ──
        t_robotbase_left_current = (cal.t_robotbase_vuer @ self._t_vuer_left).astype(np.float32)
        t_robotbase_right_current = (cal.t_robotbase_vuer @ self._t_vuer_right).astype(np.float32)

        t_robotbase_left_target = target_pose_from_hand(
            t_robotbase_left_current, cal.t_robotbase_left_hand_ref,
            cal.t_robotbase_left_eef_ref, self.position_scale)
        t_robotbase_right_target = target_pose_from_hand(
            t_robotbase_right_current, cal.t_robotbase_right_hand_ref,
            cal.t_robotbase_right_eef_ref, self.position_scale)

        t_world_left_target = (self.t_world_robotbase @ t_robotbase_left_target).astype(np.float32)
        t_world_right_target = (self.t_world_robotbase @ t_robotbase_right_target).astype(np.float32)

        # Auto grasp override
        if self.auto_grasp.active and self.auto_grasp.target_matrix is not None:
            ee_body = (self.left_body if self.auto_grasp.arm == "left"
                       else self.right_body)
            ee_pos = self.data.xpos[ee_body].copy()
            smoothed = self.auto_grasp.get_smoothed_target(ee_pos)
            if smoothed is not None:
                if self.auto_grasp.arm == "left":
                    t_world_left_target = smoothed
                else:
                    t_world_right_target = smoothed

        # Solve IK
        dq_l = self.left_ik.solve(t_world_left_target)
        dq_r = self.right_ik.solve(t_world_right_target)

        # Smooth + update q_smoothed
        alpha = self.arm_smoothing
        if alpha < 1.0:
            self.q_smoothed[self.arm_qpos_indices] = (
                (1.0 - alpha) * self.q_smoothed[self.arm_qpos_indices]
                + alpha * self.data.qpos[self.arm_qpos_indices])
        else:
            self.q_smoothed[self.arm_qpos_indices] = self.data.qpos[self.arm_qpos_indices]

        # ── Gripper ──
        if self._gripper_manual_override is None:
            lm_left_ok = self._left_landmarks is not None and self._left_landmarks.shape[0] >= 10
            lm_right_ok = self._right_landmarks is not None and self._right_landmarks.shape[0] >= 10
            if lm_left_ok:
                self.left_gripper_cmd, self._gripper_metrics = gripper_command_from_landmarks(
                    self._left_landmarks, "left", self._gripper_metrics,
                    self.enable_gripper, self.gripper_open_value, self.gripper_close_value,
                    self.gripper_close_threshold, self.gripper_open_threshold,
                    self.gripper_smoothing, self.thumb_tip_index, self.index_tip_index)
            if lm_right_ok:
                self.right_gripper_cmd, self._gripper_metrics = gripper_command_from_landmarks(
                    self._right_landmarks, "right", self._gripper_metrics,
                    self.enable_gripper, self.gripper_open_value, self.gripper_close_value,
                    self.gripper_close_threshold, self.gripper_open_threshold,
                    self.gripper_smoothing, self.thumb_tip_index, self.index_tip_index)

        # Set finger qpos
        self.q_smoothed[self.left_finger_qpos[0]] = self.left_gripper_cmd
        self.q_smoothed[self.left_finger_qpos[1]] = self.left_gripper_cmd
        self.q_smoothed[self.right_finger_qpos[0]] = self.right_gripper_cmd
        self.q_smoothed[self.right_finger_qpos[1]] = self.right_gripper_cmd

        self._publish_joint_target()

    def _publish_joint_target(self):
        """Publish current q_smoothed as joint target."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        all_names = ALL_ARM_JOINTS + ALL_FINGER_JOINTS
        all_qpos = np.concatenate([self.left_qpos, self.right_qpos,
                                   self.left_finger_qpos, self.right_finger_qpos])
        for name, qidx in zip(all_names, all_qpos):
            msg.name.append(name)
            msg.position.append(float(self.q_smoothed[qidx]))
        self.joint_target_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
