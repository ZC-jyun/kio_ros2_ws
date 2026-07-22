#!/usr/bin/env python3
"""demo_state_node — door open and takeout demo state machines.

Subscribes to /motor_state.
Provides /demo/trigger (DemoTrigger) and /demo/reset (Trigger) services.
Publishes /trajectory/playback and /demo/state.
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_srvs.srv import Trigger

from kio_teleop_openarm.lib.demo_state import (
    DoorDemoController, TakeoutDemoController)


class DemoStateNode(Node):
    def __init__(self):
        super().__init__("demo_state_node")

        self.declare_parameter("trajectories_dir", "")
        self.declare_parameter("publish_rate", 20.0)
        traj_dir = self.get_parameter("trajectories_dir").value
        publish_rate = self.get_parameter("publish_rate").value

        # Demo controllers
        if not traj_dir:
            from pathlib import Path
            traj_dir = str(Path(__file__).parents[2] / "config" / "trajectories")
        self._door_ctrl = DoorDemoController(self, traj_dir)
        self._takeout_ctrl = TakeoutDemoController(self, traj_dir)

        # Subscribers
        self._motor_sub = self.create_subscription(
            JointState, "/motor_state", lambda m: None, 10)

        # Publishers
        self._traj_pub = self.create_publisher(
            JointTrajectory, "/trajectory/playback", 10)
        self._state_pub = self.create_publisher(String, "/demo/state", 10)

        # Services
        try:
            from kio_teleop_openarm.srv import DemoTrigger
            self.create_service(DemoTrigger, "/demo/trigger", self._trigger_cb)
        except ImportError:
            self.get_logger().warn("DemoTrigger srv not available")
        self.create_service(Trigger, "/demo/reset", self._reset_cb)

        # Timer
        self._timer = self.create_timer(1.0 / publish_rate, self._tick)

        # Service clients (for takeout demo)
        self._perception_client = None
        self._plan_client = None

        self.get_logger().info("demo_state_node started")

    def _trigger_cb(self, request, response):
        demo_type = request.demo_type
        if demo_type == "door_open":
            ok = self._door_ctrl.start()
            response.success = ok
            response.message = "Door demo started" if ok else "Already running"
        elif demo_type == "takeout":
            ok = self._takeout_ctrl.start()
            response.success = ok
            response.message = "Takeout demo started" if ok else "Already running"
        else:
            response.success = False
            response.message = f"Unknown demo type: {demo_type}"
        return response

    def _reset_cb(self, request, response):
        self._door_ctrl.reset()
        self._takeout_ctrl.reset()
        response.success = True
        response.message = "Demos reset"
        return response

    def _tick(self):
        # Door demo
        if self._door_ctrl._active:
            status = self._door_ctrl.tick()
            if status:
                self._state_pub.publish(String(data=json.dumps(status)))
            traj = self._door_ctrl.get_trajectory()
            if traj is not None:
                self._traj_pub.publish(traj)

        # Takeout demo
        if self._takeout_ctrl._active:
            status = self._takeout_ctrl.tick()
            if status:
                self._state_pub.publish(String(data=json.dumps(status)))

            # Handle state transitions
            if self._takeout_ctrl.state.value == "detect_bag":
                self._do_detect_bag()
            elif self._takeout_ctrl.state.value == "grasp_bag":
                self._do_plan_grasp()

            traj = self._takeout_ctrl.get_trajectory()
            if traj is not None:
                self._traj_pub.publish(traj)

    def _do_detect_bag(self):
        try:
            from kio_teleop_openarm.srv import DetectObjects
            if self._perception_client is None:
                self._perception_client = self.create_client(
                    DetectObjects, "/perception/detect")
            if not self._perception_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Perception service not available")
                self._takeout_ctrl.on_detection_result([], None)
                return
            req = DetectObjects.Request()
            req.text_prompt = "takeout bag"
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
                self._takeout_ctrl.on_detection_result(dets, None)
            else:
                self._takeout_ctrl.on_detection_result([], None)
        except Exception as e:
            self.get_logger().error(f"Detect bag failed: {e}")
            self._takeout_ctrl.on_detection_result([], None)

    def _do_plan_grasp(self):
        try:
            from kio_teleop_openarm.srv import PlanGrasp
            from kio_teleop_openarm.msg import GraspCandidate
            from geometry_msgs.msg import Pose, Point, Quaternion
            if self._plan_client is None:
                self._plan_client = self.create_client(PlanGrasp, "/grasp/plan")
            if not self._plan_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Plan service not available")
                self._takeout_ctrl.on_plan_result(None)
                return
            req = PlanGrasp.Request()
            req.candidate = GraspCandidate()
            req.candidate.description = "takeout bag top grasp"
            req.candidate.pre_grasp_pose = Pose()
            req.candidate.pre_grasp_pose.position = Point(x=0.3, y=0.0, z=0.9)
            req.candidate.pre_grasp_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            req.candidate.grasp_pose = Pose()
            req.candidate.grasp_pose.position = Point(x=0.3, y=0.0, z=0.8)
            req.candidate.grasp_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            future = self._plan_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.done() and future.result() is not None and future.result().success:
                self._takeout_ctrl.on_plan_result(future.result().trajectory)
            else:
                self._takeout_ctrl.on_plan_result(None)
        except Exception as e:
            self.get_logger().error(f"Plan grasp failed: {e}")
            self._takeout_ctrl.on_plan_result(None)


def main(args=None):
    rclpy.init(args=args)
    node = DemoStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
