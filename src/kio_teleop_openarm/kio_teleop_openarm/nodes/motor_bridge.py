#!/usr/bin/env python3
"""motor_bridge node — drives real UPOO motors via USB2CANFD (MIT mode)."""

import threading
import time
import os

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

_hardware_available = False
_HardwareMotorBridge = None
try:
    from pathlib import Path
    _LIB_DIR = Path(__file__).resolve().parent.parent / "lib"
    os.chdir(str(_LIB_DIR))  # dmcan needs ./dlls/ relative to cwd
    from kio_teleop_openarm.lib import teleop_upoo_hardware as _hw_mod
    _HardwareMotorBridge = _hw_mod.HardwareMotorBridge
    _hardware_available = True
except ImportError:
    pass


class MotorBridgeNode(Node):
    def __init__(self):
        super().__init__("motor_bridge")

        self.declare_parameter("motor_smoothing", 0.3)
        self.declare_parameter("device_sn", "")
        self.declare_parameter("kp", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.declare_parameter("kd", [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("gripper_open_pos", 5.0)
        self.declare_parameter("gripper_open_value", 0.044)

        motor_smoothing = self.get_parameter("motor_smoothing").value
        device_sn = self.get_parameter("device_sn").value or None
        kp = self.get_parameter("kp").value
        kd = self.get_parameter("kd").value
        publish_rate = self.get_parameter("publish_rate").value
        self.gripper_open_pos = self.get_parameter("gripper_open_pos").value
        self.gripper_open_value = self.get_parameter("gripper_open_value").value

        if not _hardware_available:
            self.get_logger().fatal("Hardware motor bridge not available. Check teleop_upoo_hardware import.")
            raise RuntimeError("HardwareMotorBridge unavailable")

        self._bridge = _HardwareMotorBridge(
            kp=kp, kd=kd, motor_smoothing=motor_smoothing, device_sn=device_sn)
        self._bridge.start()
        self.get_logger().info("Hardware motor bridge started")

        self.joint_target_sub = self.create_subscription(
            JointState, "/joint_target", self._target_cb, 10)
        self.joint_state_pub = self.create_publisher(JointState, "/motor_state", 10)
        self.estop_srv = self.create_service(Trigger, "/emergency_stop", self._estop_cb)
        self.zero_srv = self.create_service(Trigger, "/set_zero", self._zero_cb)

        self._timer = self.create_timer(1.0 / publish_rate, self._publish_state)
        self.get_logger().info("motor_bridge node started")

    def _target_cb(self, msg: JointState):
        if not msg.name or len(msg.name) != len(msg.position):
            return
        # 7-motor UPOO arm: 6 arm joints + gripper
        arm_names = ["upoo_right_Base_J01", "upoo_right_J02", "upoo_right_J03",
                     "upoo_right_J04", "upoo_right_J05", "upoo_right_J06",
                     "upoo_right_finger_left_joint"]
        target = []
        for n in arm_names:
            if n in msg.name:
                val = float(msg.position[msg.name.index(n)])
                # Scale gripper from sim units to motor radians
                if n == "upoo_right_finger_left_joint":
                    val = val * (self.gripper_open_pos / max(self.gripper_open_value, 1e-6))
                target.append(val)
            else:
                return  # incomplete target, skip
        self._bridge.set_target(np.array(target, dtype=np.float32))

    def _publish_state(self):
        sent, errs = self._bridge.get_state()
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ["motor_j1", "motor_j2", "motor_j3", "motor_j4", "motor_j5",
                   "motor_j6", "motor_gripper"]
        js.position = [float(s) for s in sent]
        js.effort = [float(e) for e in errs]
        self.joint_state_pub.publish(js)

    def _estop_cb(self, request, response):
        self._bridge.emergency_stop()
        response.success = True
        response.message = "Emergency stop executed"
        self.get_logger().warn("EMERGENCY STOP")
        return response

    def _zero_cb(self, request, response):
        self._bridge.set_zero_all()
        response.success = True
        response.message = "Zero set"
        self.get_logger().info("Zero set for all motors")
        return response

    def stop(self):
        if hasattr(self, '_bridge') and self._bridge is not None:
            self._bridge.stop()


def main(args=None):
    rclpy.init(args=args)
    node = MotorBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
