"""Demo state machines — door open + takeout delivery.

ROS2-adapted: pre-recorded trajectories played via /trajectory/playback.
Navigation phases removed (no mobile base).
"""

import time
import json
import numpy as np
from enum import Enum
from pathlib import Path
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ALL_JOINT_NAMES = [
    "upoo_left_Base_J01", "upoo_left_J02", "upoo_left_J03",
    "upoo_left_J04", "upoo_left_J05", "upoo_left_J06",
    "upoo_right_Base_J01", "upoo_right_J02", "upoo_right_J03",
    "upoo_right_J04", "upoo_right_J05", "upoo_right_J06",
    "upoo_left_finger_left_joint", "upoo_left_finger_right_joint",
    "upoo_right_finger_left_joint", "upoo_right_finger_right_joint",
]


def load_trajectory(json_path: str) -> JointTrajectory | None:
    """Load a pre-recorded trajectory from a JSON file.

    Expected format: list of {rel_time_s, q_joints (16 floats in radians)}.
    """
    path = Path(json_path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        return None
    jt = JointTrajectory()
    jt.joint_names = ALL_JOINT_NAMES
    for frame in data:
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in frame.get("q_joints", frame.get("positions", []))]
        t = float(frame.get("rel_time_s", frame.get("time_from_start", 0.0)))
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
        jt.points.append(pt)
    return jt


def make_safe_trajectory(duration=3.0) -> JointTrajectory:
    """Generate a trajectory to the L-shape safe position."""
    jt = JointTrajectory()
    jt.joint_names = ALL_JOINT_NAMES
    safe_q = [
        -1.57, 0.0, 0.0, 1.57, -1.57, 0.0,
        -1.57, 0.0, 0.0, 1.57, -1.57, 0.0,
        0.044, 0.044, 0.044, 0.044,
    ]
    for alpha in [0.3, 0.7, 1.0]:
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in safe_q]
        t = alpha * duration
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
        jt.points.append(pt)
    return jt


# ═══════════════════════════════════════════════════════════════════
# Door Open Demo
# ═══════════════════════════════════════════════════════════════════

class DoorDemoState(Enum):
    IDLE = "idle"
    APPROACHING = "approaching"
    OPENING = "opening"
    RETREATING = "retreating"
    COMPLETED = "completed"
    FAILED = "failed"


class DoorDemoController:
    def __init__(self, node, trajectories_dir: str = ""):
        self._node = node
        self.state = DoorDemoState.IDLE
        self._traj_dir = Path(trajectories_dir) if trajectories_dir else Path(__file__).parents[3] / "config" / "trajectories"
        self._current_traj = None
        self._traj_start = 0.0
        self._traj_duration = 0.0
        self._active = False

    def start(self) -> bool:
        if self.state != DoorDemoState.IDLE:
            return False
        self._enter(DoorDemoState.APPROACHING)
        self._active = True
        return True

    def reset(self):
        self.state = DoorDemoState.IDLE
        self._current_traj = None
        self._active = False

    def tick(self) -> dict | None:
        status = {"type": "demo_status", "demo": "door_open",
                  "state": self.state.value, "ts": time.time()}

        if self.state == DoorDemoState.APPROACHING and self._current_traj is None:
            self._load_and_play("door_open_approach.json")
        elif self.state == DoorDemoState.OPENING and self._current_traj is None:
            self._load_and_play("door_open.json")
        elif self.state == DoorDemoState.RETREATING and self._current_traj is None:
            self._load_and_play("door_open_retreat.json")

        # Check if current trajectory is done
        if self._current_traj is not None:
            elapsed = time.time() - self._traj_start
            if elapsed > self._traj_duration + 0.5:
                self._current_traj = None
                if self.state == DoorDemoState.APPROACHING:
                    self._enter(DoorDemoState.OPENING)
                elif self.state == DoorDemoState.OPENING:
                    self._enter(DoorDemoState.RETREATING)
                elif self.state == DoorDemoState.RETREATING:
                    self._enter(DoorDemoState.COMPLETED)
                    self._active = False

        return status

    def get_trajectory(self) -> JointTrajectory | None:
        """Return the trajectory to publish, or None."""
        return self._current_traj

    def _load_and_play(self, filename: str):
        path = self._traj_dir / filename
        traj = load_trajectory(str(path))
        if traj is None:
            self._node.get_logger().warn(f"Trajectory not found: {path}, using safe fallback")
            traj = make_safe_trajectory()
        self._current_traj = traj
        self._traj_start = time.time()
        if traj.points:
            last = traj.points[-1]
            self._traj_duration = last.time_from_start.sec + last.time_from_start.nanosec * 1e-9
        self._node.get_logger().info(f"Playing: {filename} ({self._traj_duration:.1f}s)")

    def _enter(self, new_state: DoorDemoState):
        old = self.state
        self.state = new_state
        self._current_traj = None
        if old != new_state:
            self._node.get_logger().info(f"[door_demo] {old.value} -> {new_state.value}")


# ═══════════════════════════════════════════════════════════════════
# Takeout Demo
# ═══════════════════════════════════════════════════════════════════

class TakeoutDemoState(Enum):
    IDLE = "idle"
    OPEN_DOOR = "open_door"
    DETECT_BAG = "detect_bag"
    GRASP_BAG = "grasp_bag"
    DELIVER = "deliver"
    COMPLETED = "completed"
    FAILED = "failed"


class TakeoutDemoController:
    def __init__(self, node, trajectories_dir: str = ""):
        self._node = node
        self.state = TakeoutDemoState.IDLE
        self._traj_dir = Path(trajectories_dir) if trajectories_dir else Path(__file__).parents[3] / "config" / "trajectories"
        self._door_ctrl = DoorDemoController(node, str(self._traj_dir))
        self._current_traj = None
        self._traj_start = 0.0
        self._traj_duration = 0.0
        self._active = False

        # Async operation flags
        self._detection_done = False
        self._detection_result = None
        self._plan_done = False
        self._plan_result = None

    def start(self) -> bool:
        if self.state != TakeoutDemoState.IDLE:
            return False
        self._enter(TakeoutDemoState.OPEN_DOOR)
        self._active = True
        return True

    def reset(self):
        self.state = TakeoutDemoState.IDLE
        self._current_traj = None
        self._door_ctrl.reset()
        self._active = False
        self._detection_done = False
        self._plan_done = False

    def on_detection_result(self, detections, depth):
        self._detection_result = (detections, depth)
        self._detection_done = True

    def on_plan_result(self, trajectory):
        self._plan_result = trajectory
        self._plan_done = True

    def tick(self) -> dict | None:
        status = {"type": "demo_status", "demo": "takeout",
                  "state": self.state.value, "ts": time.time()}

        if self.state == TakeoutDemoState.OPEN_DOOR:
            door_status = self._door_ctrl.tick()
            if not self._door_ctrl._active:
                if self._door_ctrl.state == DoorDemoState.COMPLETED:
                    self._enter(TakeoutDemoState.DETECT_BAG)
                else:
                    self._enter(TakeoutDemoState.FAILED)

        elif self.state == TakeoutDemoState.DETECT_BAG:
            if self._detection_done:
                self._detection_done = False
                dets, _ = self._detection_result
                if dets and len(dets) > 0:
                    bag_dets = [d for d in dets if "bag" in d.get("class_name", "").lower()]
                    if bag_dets:
                        self._node.get_logger().info(f"Found takeout bag")
                        self._enter(TakeoutDemoState.GRASP_BAG)
                    else:
                        self._node.get_logger().warn("No takeout bag detected, using recorded grasp")
                        self._load_and_play("takeout_grasp.json")
                        self._enter(TakeoutDemoState.DELIVER)
                else:
                    self._node.get_logger().warn("Detection failed, using recorded trajectory")
                    self._load_and_play("takeout_grasp.json")
                    self._enter(TakeoutDemoState.DELIVER)

        elif self.state == TakeoutDemoState.GRASP_BAG:
            if self._plan_done:
                self._plan_done = False
                if self._plan_result is not None:
                    self._current_traj = self._plan_result
                    self._traj_start = time.time()
                    if self._plan_result.points:
                        last = self._plan_result.points[-1]
                        self._traj_duration = last.time_from_start.sec + last.time_from_start.nanosec * 1e-9
                    self._node.get_logger().info(f"Executing planned grasp trajectory")
                else:
                    self._node.get_logger().warn("Planning failed, using recorded trajectory")
                    self._load_and_play("takeout_grasp.json")
                self._enter(TakeoutDemoState.DELIVER)

        elif self.state == TakeoutDemoState.DELIVER:
            if self._current_traj is None:
                self._load_and_play("takeout_deliver.json")
            elif time.time() - self._traj_start > self._traj_duration + 0.5:
                self._node.get_logger().info("Delivery complete")
                self._enter(TakeoutDemoState.COMPLETED)
                self._active = False

        return status

    def get_trajectory(self) -> JointTrajectory | None:
        if self._door_ctrl._active:
            return self._door_ctrl.get_trajectory()
        return self._current_traj

    def _load_and_play(self, filename: str):
        path = self._traj_dir / filename
        traj = load_trajectory(str(path))
        if traj is None:
            self._node.get_logger().warn(f"Trajectory not found: {path}")
            traj = make_safe_trajectory()
        self._current_traj = traj
        self._traj_start = time.time()
        if traj.points:
            last = traj.points[-1]
            self._traj_duration = last.time_from_start.sec + last.time_from_start.nanosec * 1e-9

    def _enter(self, new_state: TakeoutDemoState):
        old = self.state
        self.state = new_state
        self._current_traj = None
        if old != new_state:
            self._node.get_logger().info(f"[takeout_demo] {old.value} -> {new_state.value}")
