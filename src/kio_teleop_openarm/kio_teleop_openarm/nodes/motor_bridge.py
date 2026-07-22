#!/usr/bin/env python3
"""motor_bridge node — drives real UPOO motors via USB2CANFD (MIT mode).
Includes per-joint trapezoidal velocity profile interpolation."""

import threading
import time
import os
import copy

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

ARM_NAMES = ["upoo_right_Base_J01", "upoo_right_J02", "upoo_right_J03",
             "upoo_right_J04", "upoo_right_J05", "upoo_right_J06",
             "upoo_right_finger_left_joint"]
NUM_MOTORS = len(ARM_NAMES)


class TrapezoidalProfile:
    """Per-joint trapezoidal velocity profile generator.

    Phases: accel -> cruise -> decel -> done
    A new target aborts the current trajectory immediately and replans from
    the current state.
    """

    def __init__(self, max_vel: float, max_acc: float):
        self.max_vel = float(max_vel)
        self.max_acc = float(max_acc)
        self.reset(0.0)

    def reset(self, current_pos: float):
        self.start_pos = float(current_pos)
        self.target_pos = float(current_pos)
        self.current_pos = float(current_pos)
        self.current_vel = 0.0
        self._phase = "done"

    def set_target(self, new_target: float):
        """Replan from current state to new_target."""
        self.start_pos = self.current_pos
        self.target_pos = float(new_target)
        if abs(self.target_pos - self.current_pos) < 1e-8:
            self.current_vel = 0.0
            self._phase = "done"
        else:
            self._phase = "accel"

    def step(self, dt: float) -> float:
        """Advance profile by dt seconds, return new position setpoint."""
        if self._phase == "done":
            return self.target_pos

        dt = max(dt, 1e-9)
        remaining = self.target_pos - self.current_pos
        direction = 1.0 if remaining > 0.0 else -1.0

        if self._phase == "accel":
            self.current_vel += self.max_acc * dt * direction
            if abs(self.current_vel) >= self.max_vel:
                self.current_vel = self.max_vel * direction
                self._phase = "cruise"

        if self._phase == "cruise":
            # Check if it's time to start decelerating
            # v^2 = 2*a*d  =>  d_decel = v^2 / (2*a)
            decel_dist = (self.current_vel ** 2) / (2.0 * self.max_acc)
            if abs(remaining) <= decel_dist + 1e-6:
                self._phase = "decel"

        if self._phase == "decel":
            step_decel = self.max_acc * dt
            if abs(self.current_vel) <= step_decel:
                self.current_vel = 0.0
                self.current_pos = self.target_pos
                self._phase = "done"
                return self.target_pos
            self.current_vel -= step_decel * direction

        self.current_pos += self.current_vel * dt

        # Clamp overshoot
        if direction > 0 and self.current_pos >= self.target_pos:
            self.current_pos = self.target_pos
            self.current_vel = 0.0
            self._phase = "done"
        elif direction < 0 and self.current_pos <= self.target_pos:
            self.current_pos = self.target_pos
            self.current_vel = 0.0
            self._phase = "done"

        return self.current_pos

    @property
    def is_done(self) -> bool:
        return self._phase == "done"


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
        self.declare_parameter("max_vel", [3.0] * NUM_MOTORS)
        self.declare_parameter("max_acc", [10.0] * NUM_MOTORS)
        self.declare_parameter("profile_rate", 200.0)

        motor_smoothing = self.get_parameter("motor_smoothing").value
        device_sn = self.get_parameter("device_sn").value or None
        kp = self.get_parameter("kp").value
        kd = self.get_parameter("kd").value
        publish_rate = self.get_parameter("publish_rate").value
        self.gripper_open_pos = self.get_parameter("gripper_open_pos").value
        self.gripper_open_value = self.get_parameter("gripper_open_value").value
        max_vel = self.get_parameter("max_vel").value
        max_acc = self.get_parameter("max_acc").value
        profile_rate = self.get_parameter("profile_rate").value

        if not _hardware_available:
            self.get_logger().fatal("Hardware motor bridge not available. Check teleop_upoo_hardware import.")
            raise RuntimeError("HardwareMotorBridge unavailable")

        self._bridge = _HardwareMotorBridge(
            kp=kp, kd=kd, motor_smoothing=motor_smoothing, device_sn=device_sn)
        self._bridge.start()
        self.get_logger().info("Hardware motor bridge started")

        # Per-joint trapezoidal profiles
        self._profiles = [
            TrapezoidalProfile(max_vel[i], max_acc[i])
            for i in range(NUM_MOTORS)
        ]
        self._profile_lock = threading.Lock()
        self._profile_active = False

        # Seed profiles with current motor position (read from bridge after start)
        time.sleep(0.2)  # let motor thread run a few cycles
        cur_pos, _ = self._bridge.get_state()
        for i in range(NUM_MOTORS):
            self._profiles[i].reset(float(cur_pos[i]))
        self.get_logger().info(
            f"Profiles seeded at {[round(p, 4) for p in cur_pos]}")

        # Background profile stepping thread
        self._profile_thread = threading.Thread(
            target=self._profile_loop,
            args=(1.0 / profile_rate,),
            daemon=True,
            name="profile-thread",
        )
        self._profile_running = threading.Event()
        self._profile_running.set()
        self._profile_thread.start()

        # ROS2 interfaces
        self.joint_target_sub = self.create_subscription(
            JointState, "/joint_target", self._target_cb, 10)
        self.joint_state_pub = self.create_publisher(JointState, "/motor_state", 10)
        self.estop_srv = self.create_service(Trigger, "/emergency_stop", self._estop_cb)
        self.zero_srv = self.create_service(Trigger, "/set_zero", self._zero_cb)

        self._timer = self.create_timer(1.0 / publish_rate, self._publish_state)
        self.get_logger().info("motor_bridge node started")

    # ── Target callback: replan all profiles ──────────────────────

    def _target_cb(self, msg: JointState):
        if not msg.name or len(msg.name) != len(msg.position):
            return
        targets = []
        for n in ARM_NAMES:
            if n in msg.name:
                val = float(msg.position[msg.name.index(n)])
                if n == "upoo_right_finger_left_joint":
                    val = val * (self.gripper_open_pos / max(self.gripper_open_value, 1e-6))
                targets.append(val)
            else:
                return  # incomplete target

        with self._profile_lock:
            for i, t in enumerate(targets):
                self._profiles[i].set_target(t)
            self._profile_active = True

    # ── Profile stepping thread ───────────────────────────────────

    def _profile_loop(self, period: float):
        """Run at ``profile_rate`` Hz. Step each profile and push
        intermediate setpoints to HardwareMotorBridge."""
        while self._profile_running.is_set():
            t_start = time.perf_counter()

            with self._profile_lock:
                any_running = False
                setpoints = np.zeros(NUM_MOTORS, dtype=np.float32)
                for i, prof in enumerate(self._profiles):
                    setpoints[i] = prof.step(period)
                    if not prof.is_done:
                        any_running = True
                self._profile_active = any_running

            self._bridge.set_target(setpoints)

            elapsed = time.perf_counter() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── State publishing ──────────────────────────────────────────

    def _publish_state(self):
        sent, errs = self._bridge.get_state()
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ["motor_j1", "motor_j2", "motor_j3", "motor_j4", "motor_j5",
                   "motor_j6", "motor_gripper"]
        js.position = [float(s) for s in sent]
        js.effort = [float(e) for e in errs]
        self.joint_state_pub.publish(js)

    # ── Services ──────────────────────────────────────────────────

    def _estop_cb(self, request, response):
        self._bridge.emergency_stop()
        with self._profile_lock:
            cur, _ = self._bridge.get_state()
            for i, prof in enumerate(self._profiles):
                prof.reset(float(cur[i]))
        response.success = True
        response.message = "Emergency stop executed"
        self.get_logger().warn("EMERGENCY STOP")
        return response

    def _zero_cb(self, request, response):
        self._bridge.set_zero_all()
        with self._profile_lock:
            for prof in self._profiles:
                prof.reset(0.0)
        response.success = True
        response.message = "Zero set"
        self.get_logger().info("Zero set for all motors")
        return response

    def stop(self):
        self._profile_running.clear()
        if hasattr(self, '_profile_thread') and self._profile_thread is not None:
            self._profile_thread.join(timeout=1.0)
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
