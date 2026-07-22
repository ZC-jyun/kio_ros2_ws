"""Semi-autonomous grasp state machine — ROS2 adapted.

State flow: IDLE → PERCEIVING → WAITING_SELECTION → PLANNING → EXECUTING
  → SUCCESS | FAILED → RECOVERY → IDLE
  → INTERRUPTED (VR takeover / estop) → IDLE
"""

import time
import numpy as np
from enum import Enum


class GraspState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    WAITING_SELECTION = "waiting"
    PLANNING = "planning"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    RECOVERY = "recovery"
    INTERRUPTED = "interrupted"


class AutoGraspController:
    def __init__(self, node, select_timeout=30.0):
        self._node = node
        self.state = GraspState.IDLE
        self.candidates = []
        self.selected = None
        self.trajectory = None
        self.select_timeout = select_timeout
        self._state_start_ts = 0.0
        self._perception_done = False
        self._perception_results = None  # (detections, depth)
        self._plan_done = False
        self._plan_result = None
        self._exec_done = False
        self._exec_interrupted = False
        self._recovery_done = False
        self._active_arm = "left"

    # ── Public API ──

    def start_perception(self):
        if self.state != GraspState.IDLE:
            self._node.get_logger().warn(
                f"Cannot start perception in state {self.state.value}")
            return False
        self._enter(GraspState.PERCEIVING)
        return True

    def handle_selection(self, obj_idx: int, grasp_idx: int):
        if self.state != GraspState.WAITING_SELECTION:
            return False
        if obj_idx >= len(self.candidates):
            return False
        grasps = self.candidates[obj_idx].get("grasps", [])
        if grasp_idx >= len(grasps):
            return False
        self.selected = {"obj_idx": obj_idx, "grasp_idx": grasp_idx}
        self._node.get_logger().info(
            f"Selection: obj={obj_idx}, grasp={grasp_idx}")
        return True

    def reset(self):
        self.state = GraspState.IDLE
        self.candidates = []
        self.selected = None
        self.trajectory = None
        self._perception_done = False
        self._perception_results = None
        self._plan_done = False
        self._plan_result = None
        self._exec_done = False
        self._exec_interrupted = False
        self._recovery_done = False

    # ── Called by node when async operations complete ──

    def on_perception_result(self, detections, depth):
        self._perception_results = (detections, depth)
        self._perception_done = True

    def on_plan_result(self, trajectory):
        self._plan_result = trajectory
        self._plan_done = True

    def on_execution_complete(self, interrupted=False):
        self._exec_done = True
        self._exec_interrupted = interrupted

    def on_recovery_complete(self):
        self._recovery_done = True

    # ── Tick (called periodically by node) ──

    def tick(self) -> dict | None:
        """Advance state machine. Returns JSON-serializable status dict for app_bridge, or None."""
        status = {"type": "task_status", "state": self.state.value, "ts": time.time()}

        if self.state == GraspState.PERCEIVING:
            self._tick_perceiving()
        elif self.state == GraspState.WAITING_SELECTION:
            self._tick_waiting()
        elif self.state == GraspState.PLANNING:
            self._tick_planning()
        elif self.state == GraspState.EXECUTING:
            self._tick_executing()
        elif self.state == GraspState.RECOVERY:
            self._tick_recovery()

        status["state"] = self.state.value
        status["ts"] = time.time()
        return status

    def _tick_perceiving(self):
        if not self._perception_done:
            return
        detections, depth = self._perception_results
        self._perception_done = False

        if not detections:
            self._node.get_logger().info("No objects detected")
            self._enter(GraspState.IDLE)
            return

        self.candidates = detections  # list of {class_name, confidence, bbox, ...} with depth
        if not self.candidates:
            self._enter(GraspState.IDLE)
            return

        # Build candidate list for app
        self._enter(GraspState.WAITING_SELECTION)

    def _tick_waiting(self):
        elapsed = time.time() - self._state_start_ts
        if elapsed > self.select_timeout:
            self._node.get_logger().info("Selection timeout")
            self._enter(GraspState.IDLE)
            return
        if self.selected is not None:
            self._enter(GraspState.PLANNING)

    def _tick_planning(self):
        if not self._plan_done:
            return
        self._plan_done = False
        trajectory = self._plan_result
        if trajectory is None:
            self._node.get_logger().error("Trajectory planning failed")
            self._enter(GraspState.IDLE)
            return
        self.trajectory = trajectory
        self._node.get_logger().info(
            f"Planned trajectory: {len(trajectory.points)} points")
        self._enter(GraspState.EXECUTING)

    def _tick_executing(self):
        if not self._exec_done:
            return
        self._exec_done = False
        if self._exec_interrupted:
            self._enter(GraspState.INTERRUPTED)
        else:
            # Grasp success will be checked separately by the node
            self._enter(GraspState.SUCCESS)

    def _tick_recovery(self):
        if not self._recovery_done:
            return
        self._recovery_done = False
        self._node.get_logger().info("Recovery complete")
        self._enter(GraspState.IDLE)

    def _enter(self, new_state: GraspState):
        old = self.state
        self.state = new_state
        self._state_start_ts = time.time()
        if old != new_state:
            self._node.get_logger().info(f"{old.value} -> {new_state.value}")


def check_grasp_success(motor_state: dict, finger_names: list,
                        close_threshold=0.01) -> bool:
    """Check if grasp succeeded by comparing actual gripper position to fully-closed.

    In sim space: 0.0 = fully closed, 0.044 = fully open.
    If actual position > close_threshold, the gripper didn't fully close → grasped something.
    """
    if not motor_state:
        return False
    for name in finger_names:
        actual = motor_state.get(name, 0.0)
        if abs(actual) > close_threshold:
            return True
    return False
