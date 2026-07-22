#!/usr/bin/env python3
"""trajectory_executor_node — executes JointTrajectory by publishing to /joint_target.

Subscribes to /trajectory/playback and /motor_state.
Publishes /joint_target.
Provides /trajectory/cancel service.
"""

import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_srvs.srv import Trigger


class TrajectoryExecutorNode(Node):
    def __init__(self):
        super().__init__("trajectory_executor_node")

        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("emergency_stop_topic", "/emergency_stop")
        self.publish_rate = self.get_parameter("publish_rate").value

        self._lock = threading.Lock()
        self._trajectory = None
        self._start_time = None
        self._cancelled = False
        self._emergency_stop = False

        # Latest motor state for gripper feedback
        self._latest_motor_state = {}

        # Subscriptions
        self._traj_sub = self.create_subscription(
            JointTrajectory, "/trajectory/playback", self._traj_cb, 10)
        self._motor_sub = self.create_subscription(
            JointState, "/motor_state", self._motor_cb, 10)

        # Publisher
        self._target_pub = self.create_publisher(JointState, "/joint_target", 10)

        # Services
        self._cancel_srv = self.create_service(
            Trigger, "/trajectory/cancel", self._cancel_cb)

        # Execution timer
        self._timer = self.create_timer(1.0 / self.publish_rate, self._exec_tick)

        self.get_logger().info("trajectory_executor_node started")

    def _motor_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._latest_motor_state[name] = float(pos)

    def _traj_cb(self, msg: JointTrajectory):
        if len(msg.points) == 0:
            self.get_logger().warn("Received empty trajectory")
            return

        with self._lock:
            self._trajectory = msg
            self._start_time = time.time()
            self._cancelled = False
            total_t = msg.points[-1].time_from_start.sec + msg.points[-1].time_from_start.nanosec * 1e-9
            self.get_logger().info(
                f"Queued trajectory: {len(msg.points)} points, {total_t:.1f}s")

    def _cancel_cb(self, request, response):
        with self._lock:
            self._cancelled = True
            self._trajectory = None
        response.success = True
        response.message = "Trajectory cancelled"
        self.get_logger().info("Trajectory cancelled")
        return response

    def _exec_tick(self):
        traj = None
        start_time = None
        cancelled = False

        with self._lock:
            traj = self._trajectory
            start_time = self._start_time
            cancelled = self._cancelled

        if traj is None or start_time is None or cancelled:
            return

        elapsed = time.time() - start_time
        points = traj.points

        # Find the two surrounding points
        if elapsed <= 0:
            q = list(points[0].positions)
        elif elapsed >= self._point_time(points[-1]):
            q = list(points[-1].positions)
            # Trajectory complete, stop execution
            with self._lock:
                self._trajectory = None
            self.get_logger().info("Trajectory execution complete")
        else:
            # Linear interpolation between surrounding points
            idx = 0
            for i, pt in enumerate(points):
                if self._point_time(pt) > elapsed:
                    idx = i
                    break

            t0 = self._point_time(points[idx - 1])
            t1 = self._point_time(points[idx])
            alpha = (elapsed - t0) / (t1 - t0) if t1 > t0 else 0.0
            alpha = max(0.0, min(1.0, alpha))

            q = []
            for j in range(len(points[idx].positions)):
                v0 = points[idx - 1].positions[j]
                v1 = points[idx].positions[j]
                q.append(v0 + alpha * (v1 - v0))

        # Publish
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for name, pos in zip(traj.joint_names, q):
            msg.name.append(name)
            msg.position.append(float(pos))
        self._target_pub.publish(msg)

    @staticmethod
    def _point_time(pt):
        return pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
