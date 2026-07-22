#!/usr/bin/env python3
"""auto_grasp_state_node — semi-autonomous grasp state machine.

Subscribes /motor_state, /vr/left_hand_pose, /vr/right_hand_pose.
Calls /perception/detect and /grasp/plan services.
Publishes /trajectory/playback and /auto_grasp/state.
Provides /auto_grasp/start and /auto_grasp/reset services.
"""

import time
import json
import numpy as np
import threading
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Trigger

from kio_teleop_openarm.lib.auto_grasp_state import (
    AutoGraspController, GraspState, check_grasp_success)

LEFT_FINGER_NAMES = ["upoo_left_finger_left_joint", "upoo_left_finger_right_joint"]
RIGHT_FINGER_NAMES = ["upoo_right_finger_left_joint", "upoo_right_finger_right_joint"]
ALL_FINGER_NAMES = LEFT_FINGER_NAMES + RIGHT_FINGER_NAMES


class AutoGraspStateNode(Node):
    def __init__(self):
        super().__init__("auto_grasp_state_node")

        self.declare_parameter("select_timeout", 30.0)
        self.declare_parameter("vr_takeover_threshold", 0.05)
        self.declare_parameter("publish_rate", 20.0)
        select_timeout = self.get_parameter("select_timeout").value
        self.vr_threshold = self.get_parameter("vr_takeover_threshold").value
        publish_rate = self.get_parameter("publish_rate").value

        # State machine
        self._ctrl = AutoGraspController(self, select_timeout=select_timeout)

        # VR takeover: sliding window of hand poses
        self._vr_lock = threading.Lock()
        self._vr_left_history = deque(maxlen=10)
        self._vr_right_history = deque(maxlen=10)
        self._vr_left_current = None
        self._vr_right_current = None

        # Motor state
        self._motor_state = {}
        self._motor_lock = threading.Lock()

        # Execution tracking
        self._exec_traj = None
        self._exec_start_time = None
        self._exec_cancelled = False

        # ── Subscribers ──
        self._motor_sub = self.create_subscription(
            JointState, "/motor_state", self._motor_cb, 10)
        self._vr_left_sub = self.create_subscription(
            PoseStamped, "/vr/left_hand_pose", self._vr_left_cb, 10)
        self._vr_right_sub = self.create_subscription(
            PoseStamped, "/vr/right_hand_pose", self._vr_right_cb, 10)

        # ── Publishers ──
        self._traj_pub = self.create_publisher(
            JointTrajectory, "/trajectory/playback", 10)
        self._state_pub = self.create_publisher(String, "/auto_grasp/state", 10)

        # ── Services ──
        self.create_service(Trigger, "/auto_grasp/start", self._start_cb)
        self.create_service(Trigger, "/auto_grasp/reset", self._reset_cb)

        # ── Timers ──
        self._tick_timer = self.create_timer(1.0 / publish_rate, self._tick)

        # ── Clients ──
        self._perception_client = None
        self._plan_client = None

        self.get_logger().info("auto_grasp_state_node started")

    # ── Subscriber callbacks ──
    def _motor_cb(self, msg: JointState):
        with self._motor_lock:
            for name, pos in zip(msg.name, msg.position):
                self._motor_state[name] = float(pos)

    def _vr_left_cb(self, msg: PoseStamped):
        mat = self._pose_to_mat4(msg)
        with self._vr_lock:
            self._vr_left_history.append(mat)
            self._vr_left_current = mat

    def _vr_right_cb(self, msg: PoseStamped):
        mat = self._pose_to_mat4(msg)
        with self._vr_lock:
            self._vr_right_history.append(mat)
            self._vr_right_current = mat

    @staticmethod
    def _pose_to_mat4(msg: PoseStamped):
        from scipy.spatial.transform import Rotation
        q = msg.pose.orientation
        r = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        t = np.eye(4)
        t[:3, :3] = r
        t[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        return t

    # ── Service callbacks ──
    def _start_cb(self, request, response):
        ok = self._ctrl.start_perception()
        response.success = ok
        response.message = "Perception started" if ok else f"Cannot start from {self._ctrl.state.value}"
        if ok:
            self._do_perception()
        return response

    def _reset_cb(self, request, response):
        self._ctrl.reset()
        self._exec_traj = None
        self._exec_cancelled = True
        response.success = True
        response.message = "Reset to IDLE"
        return response

    # ── Async operations ──
    def _do_perception(self):
        """Call /perception/detect service asynchronously."""
        try:
            from kio_teleop_openarm.srv import DetectObjects
            if self._perception_client is None:
                self._perception_client = self.create_client(
                    DetectObjects, "/perception/detect")
            if not self._perception_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Perception service not available")
                dets = []
            else:
                req = DetectObjects.Request()
                req.text_prompt = "medicine box.towel.door handle.takeout bag.cup.bottle"
                req.box_threshold = 0.25
                req.text_threshold = 0.20
                future = self._perception_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                if future.done() and future.result() is not None:
                    result = future.result()
                    dets = []
                    if result.success:
                        for d in result.detections.detections:
                            dets.append({
                                "class_name": d.class_name,
                                "confidence": d.confidence,
                                "bbox": list(d.bbox),
                            })
                    if result.depth.height > 0 and result.depth.width > 0:
                        depth = np.frombuffer(
                            result.depth.data, dtype=np.float32).reshape(
                                result.depth.height, result.depth.width)
                    else:
                        depth = np.zeros((480, 640), dtype=np.float32)
                else:
                    dets = []
                    depth = np.zeros((480, 640), dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Perception failed: {e}")
            dets = []
            depth = np.zeros((480, 640), dtype=np.float32)

        self._ctrl.on_perception_result(dets, depth)

    def _do_plan(self):
        """Call /grasp/plan service with the selected candidate."""
        if self._ctrl.selected is None:
            self._ctrl.on_plan_result(None)
            return
        try:
            from kio_teleop_openarm.srv import PlanGrasp
            from kio_teleop_openarm.msg import GraspCandidate
            if self._plan_client is None:
                self._plan_client = self.create_client(PlanGrasp, "/grasp/plan")
            if not self._plan_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Plan service not available")
                self._ctrl.on_plan_result(None)
                return
            req = PlanGrasp.Request()
            sel = self._ctrl.selected
            obj = self._ctrl.candidates[sel["obj_idx"]]
            grasp = obj["grasps"][sel["grasp_idx"]]
            req.candidate = self._make_grasp_candidate_msg(grasp)
            future = self._plan_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.done() and future.result() is not None and future.result().success:
                self._ctrl.on_plan_result(future.result().trajectory)
            else:
                self._ctrl.on_plan_result(None)
        except Exception as e:
            self.get_logger().error(f"Planning failed: {e}")
            self._ctrl.on_plan_result(None)

    def _do_execute(self):
        """Publish trajectory to /trajectory/playback."""
        if self._ctrl.trajectory is None:
            self._ctrl.on_execution_complete(interrupted=True)
            return
        self._exec_traj = self._ctrl.trajectory
        self._exec_start_time = time.time()
        self._exec_cancelled = False
        self._traj_pub.publish(self._exec_traj)
        self.get_logger().info(
            f"Executing trajectory: {len(self._exec_traj.points)} points")
        # Execution completion will be checked in _tick_exec monitor

    def _do_recovery(self):
        """Publish safe-position trajectory to /trajectory/playback."""
        jt = JointTrajectory()
        jt.joint_names = [
            "upoo_left_Base_J01", "upoo_left_J02", "upoo_left_J03",
            "upoo_left_J04", "upoo_left_J05", "upoo_left_J06",
            "upoo_right_Base_J01", "upoo_right_J02", "upoo_right_J03",
            "upoo_right_J04", "upoo_right_J05", "upoo_right_J06",
            "upoo_left_finger_left_joint", "upoo_left_finger_right_joint",
            "upoo_right_finger_left_joint", "upoo_right_finger_right_joint",
        ]
        # L-shape safe position
        safe_q = [
            -1.57, 0.0, 0.0, 1.57, -1.57, 0.0,
            -1.57, 0.0, 0.0, 1.57, -1.57, 0.0,
            0.044, 0.044, 0.044, 0.044,
        ]
        for alpha in [0.3, 0.6, 1.0]:
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in safe_q]
            pt.time_from_start.sec = int(alpha * 3.0)
            pt.time_from_start.nanosec = int((alpha * 3.0 - int(alpha * 3.0)) * 1e9)
            jt.points.append(pt)
        self._traj_pub.publish(jt)
        self.get_logger().info("Recovery trajectory published")
        # Will be marked complete after estimated duration
        self._recovery_start = time.time()

    # ── Main tick ──
    def _tick(self):
        status = self._ctrl.tick()

        # Act on state transitions
        if self._ctrl.state == GraspState.PLANNING and self._ctrl.selected is not None:
            self._do_plan()

        if self._ctrl.state == GraspState.EXECUTING and self._exec_traj is None:
            self._do_execute()

        if self._ctrl.state == GraspState.RECOVERY and not hasattr(self, '_recovery_start'):
            self._do_recovery()

        # Monitor execution
        if self._ctrl.state == GraspState.EXECUTING and self._exec_traj is not None:
            # Check VR takeover
            if self._check_vr_takeover():
                self.get_logger().info("VR takeover detected")
                self._exec_cancelled = True
                self._ctrl.on_execution_complete(interrupted=True)
                self._exec_traj = None
            # Check if trajectory duration elapsed
            elif self._exec_start_time is not None:
                total_t = (self._exec_traj.points[-1].time_from_start.sec +
                           self._exec_traj.points[-1].time_from_start.nanosec * 1e-9)
                if time.time() - self._exec_start_time > total_t + 0.5:
                    with self._motor_lock:
                        motor_copy = dict(self._motor_state)
                    success = check_grasp_success(motor_copy, LEFT_FINGER_NAMES)
                    if not success:
                        self._ctrl.state = GraspState.FAILED
                        self._ctrl.on_execution_complete(interrupted=False)
                    self._exec_traj = None

        # Monitor recovery
        if self._ctrl.state == GraspState.RECOVERY and hasattr(self, '_recovery_start'):
            if time.time() - self._recovery_start > 4.0:
                del self._recovery_start
                self._ctrl.on_recovery_complete()

        # Publish state for app_bridge
        if status:
            msg = String()
            msg.data = json.dumps(status)
            self._state_pub.publish(msg)

    def _check_vr_takeover(self) -> bool:
        with self._vr_lock:
            if len(self._vr_left_history) < 2 or len(self._vr_right_history) < 2:
                return False
            left_delta = np.linalg.norm(
                self._vr_left_history[-1][:3, 3] - self._vr_left_history[0][:3, 3])
            right_delta = np.linalg.norm(
                self._vr_right_history[-1][:3, 3] - self._vr_right_history[0][:3, 3])
            return max(left_delta, right_delta) > self.vr_threshold

    def _make_grasp_candidate_msg(self, grasp: dict):
        from kio_teleop_openarm.msg import GraspCandidate
        from geometry_msgs.msg import Pose, Point, Quaternion
        from scipy.spatial.transform import Rotation

        gc = GraspCandidate()
        gc.grasp_id = str(grasp.get("grasp_id", ""))
        gc.description = grasp.get("description", "")
        gc.score = float(grasp.get("score", 0.0))

        for key, target in [("pre_grasp_pose", gc.pre_grasp_pose),
                            ("grasp_pose", gc.grasp_pose)]:
            mat = np.array(grasp[key])
            target.position = Point(x=float(mat[0, 3]), y=float(mat[1, 3]), z=float(mat[2, 3]))
            r = Rotation.from_matrix(mat[:3, :3])
            q = r.as_quat()
            target.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        return gc


def main(args=None):
    rclpy.init(args=args)
    node = AutoGraspStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
