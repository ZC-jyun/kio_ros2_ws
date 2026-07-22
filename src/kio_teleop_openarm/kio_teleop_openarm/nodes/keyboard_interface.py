#!/usr/bin/env python3
"""keyboard_interface node — reads stdin and calls services on controller."""

import select
import sys
import threading

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from kio_teleop_openarm.srv import GripperCmd


class KeyboardInterfaceNode(Node):
    def __init__(self):
        super().__init__("keyboard_interface")

        # Service clients to controller
        self._calibrate_cli = self.create_client(Trigger, "/calibrate")
        self._reset_cup_cli = self.create_client(Trigger, "/reset_cup_sim")
        self._auto_grasp_cli = self.create_client(Trigger, "/auto_grasp")
        self._estop_cli = self.create_client(Trigger, "/emergency_stop")
        self._zero_cli = self.create_client(Trigger, "/set_zero_motor")
        self._gripper_cli = self.create_client(GripperCmd, "/set_gripper")

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._stdin_reader, daemon=True)
        self._thread.start()

        self.get_logger().info(
            "keyboard_interface started. Keys: P=calibrate R=reset T=auto_grasp "
            "G=toggle_gripper [=close ]=open E=estop Z=zero")

    def _stdin_reader(self):
        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.5)
            if r:
                line = sys.stdin.readline().strip().lower()
                if not line:
                    continue
                cmd = line[0]
                if cmd == 'p':
                    self._call_trigger(self._calibrate_cli, "calibrate")
                elif cmd == 'r':
                    self._call_trigger(self._reset_cup_cli, "reset_cup")
                elif cmd == 't':
                    self._call_trigger(self._auto_grasp_cli, "auto_grasp")
                elif cmd == 'e':
                    self._call_trigger(self._estop_cli, "estop")
                elif cmd == 'z':
                    self._call_trigger(self._zero_cli, "zero")
                elif cmd == 'g':
                    self._call_gripper("toggle")
                elif cmd == '[':
                    self._call_gripper("close_step")
                elif cmd == ']':
                    self._call_gripper("open_step")

    def _call_trigger(self, cli, name):
        if not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Service /{name} not available")
            return
        req = Trigger.Request()
        future = cli.call_async(req)
        future.add_done_callback(lambda f: self._trigger_done(f, name))

    def _trigger_done(self, future, name):
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info(f"[{name}] {resp.message}")
            else:
                self.get_logger().warn(f"[{name}] failed: {resp.message}")
        except Exception as e:
            self.get_logger().error(f"[{name}] error: {e}")

    def _call_gripper(self, command):
        if not self._gripper_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Service /set_gripper not available")
            return
        req = GripperCmd.Request()
        req.command = command
        future = self._gripper_cli.call_async(req)
        future.add_done_callback(self._gripper_done)

    def _gripper_done(self, future):
        try:
            resp = future.result()
            self.get_logger().info(f"[gripper] {'OK' if resp.success else 'FAIL'}")
        except Exception as e:
            self.get_logger().error(f"[gripper] error: {e}")

    def shutdown(self):
        self._stop.set()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
