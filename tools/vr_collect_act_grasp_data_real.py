#!/usr/bin/env python3
"""VR ACT collection with CAN-mirrored real left-arm control."""

import argparse
import json
import os
import select
import shutil
import signal
import sys
import tempfile
import termios
import threading
import time
import tty
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import mujoco
import mujoco.viewer
import numpy as np


WS = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
HARDWARE_LIB_DIR = (
    WS / "src" / "kio_teleop_openarm" / "kio_teleop_openarm" / "lib"
)
DEPLOY_DIR = Path("/home/kiorobot/kio_robot_zzc/openarm-main/teleop_deploy")

for path in (TOOLS_DIR, HARDWARE_LIB_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import upoo_motor_constants as umc
from teleop_upoo_hardware import HardwareMotorBridge
import vr_collect_act_grasp_data as sim_collector
from vr_real_runtime import (
    DryRunHardwareBridge,
    KeyboardSafetyMonitor,
    append_jsonl,
    next_contiguous_index,
)

# sim_collector imports another deployment tree. Keep the validated local bridge first.
if str(HARDWARE_LIB_DIR) in sys.path:
    sys.path.remove(str(HARDWARE_LIB_DIR))
sys.path.insert(0, str(HARDWARE_LIB_DIR))


REAL_OUTPUT_DIR = WS / "data" / "real_vr_grasp"
DEFAULT_CERT_FILE = DEPLOY_DIR / "192.168.0.5+2.pem"
DEFAULT_KEY_FILE = DEPLOY_DIR / "192.168.0.5+2-key.pem"
DEFAULT_GRIPPER_MOTOR_OPEN_POS = 5.0


def _jsonable_config(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _write_episode_file(path, buffers, outcome):
    t_actual = len(buffers["action"])
    if t_actual == 0:
        raise ValueError("Cannot save an empty episode")

    with h5py.File(path, "w", rdcc_nbytes=2 * 1024**2) as root:
        root.attrs["sim"] = False
        root.attrs["valid_length"] = t_actual
        root.attrs["outcome"] = outcome
        root.attrs["control_source"] = "vr_left_hand"
        root.attrs["joint_observation_source"] = "dm_motor_can_feedback"
        root.attrs["image_source"] = "mujoco_renderer"
        root.attrs["initial_object_pose"] = np.asarray(
            buffers["initial_object_pose"], dtype=np.float32
        )
        root.attrs["initial_motor_position"] = np.asarray(
            buffers["initial_motor_position"], dtype=np.float32
        )
        root.attrs["calibration_q"] = np.asarray(
            buffers["calibration_q"], dtype=np.float32
        )
        root.attrs["calibration_hand_pose"] = np.asarray(
            buffers["calibration_hand_pose"], dtype=np.float32
        )
        root.attrs["calibration_tcp_pose"] = np.asarray(
            buffers["calibration_tcp_pose"], dtype=np.float32
        )
        root.attrs["started_at_utc"] = buffers["started_at_utc"]
        root.attrs["finished_at_utc"] = buffers["finished_at_utc"]
        root.attrs["run_config_json"] = json.dumps(
            buffers["run_config"], ensure_ascii=True, sort_keys=True
        )
        for key, value in buffers["summary"].items():
            root.attrs[key] = value
        for name in (
            "source_evaluation_file",
            "source_evaluation_rollout_id",
            "source_evaluation_outcome",
        ):
            if name in buffers:
                root.attrs[name] = buffers[name]

        observations = root.create_group("observations")
        images = observations.create_group("images")
        for name in buffers["camera_names"]:
            frames = buffers["images"][name]
            height, width, channels = frames[0].shape
            dataset = images.create_dataset(
                name,
                (t_actual, height, width, channels),
                dtype="uint8",
                chunks=(1, height, width, channels),
                compression="lzf",
            )
            for frame_index, frame in enumerate(frames):
                dataset[frame_index] = frame

        observations.create_dataset(
            "qpos", (sim_collector.MAX_TIMESTEPS, 14), dtype="float32"
        )
        observations.create_dataset(
            "qvel", (sim_collector.MAX_TIMESTEPS, 14), dtype="float32"
        )
        root.create_dataset(
            "action", (sim_collector.MAX_TIMESTEPS, 14), dtype="float32"
        )
        root["/action"][:t_actual] = np.asarray(
            buffers["action"], dtype=np.float32
        )
        root["/observations/qpos"][:t_actual] = np.asarray(
            buffers["qpos"], dtype=np.float32
        )
        root["/observations/qvel"][:t_actual] = np.asarray(
            buffers["qvel"], dtype=np.float32
        )
        for key, last in (
            ("/action", buffers["action"][-1]),
            ("/observations/qpos", buffers["qpos"][-1]),
            ("/observations/qvel", buffers["qvel"][-1]),
        ):
            root[key][t_actual:] = last

        diagnostics = root.create_group("diagnostics")
        for name, values in buffers["diagnostics"].items():
            diagnostics.create_dataset(name, data=np.asarray(values))

        timestamps = root.create_group("timestamps")
        for name, values in buffers["timestamps"].items():
            timestamps.create_dataset(
                name, data=np.asarray(values, dtype=np.float64)
            )

        root.flush()
        handle = root.id.get_vfd_handle()
        if isinstance(handle, int):
            os.fsync(handle)


def save_real_episode(output_dir, prefix, episode_idx, buffers, outcome):
    """Atomically write one ACT-compatible episode without overwriting."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{prefix}_{episode_idx}.hdf5"
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite {final_path}")

    with tempfile.NamedTemporaryFile(
        prefix=f".{prefix}_{episode_idx}.",
        suffix=".tmp",
        dir=output_dir,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        _write_episode_file(temporary_path, buffers, outcome)
        os.link(temporary_path, final_path)
        temporary_path.unlink()
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(
        f"[record] Saved {final_path.name} "
        f"({len(buffers['action'])} frames; outcome={outcome})"
    )
    return final_path


class RealCollector(sim_collector.Collector):
    """Guarded real-arm collector with CAN-mirrored MuJoCo rendering."""

    DISARMED = "DISARMED"
    ARMED = "ARMED_UNCALIBRATED"
    CALIBRATING = "CALIBRATING"
    READY_MODE = "READY"
    RECORDING = "RECORDING"
    RETURN_REQUIRED = "RETURN_REQUIRED"
    FAULT = "FAULT"

    def __init__(self, args):
        self.hardware_bridge = None
        self.hardware_armed = False
        self.hardware_estopped = False
        self.latest_motor_q = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self.latest_motor_dq = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self.latest_motor_torque = np.zeros(umc.NUM_MOTORS, dtype=np.float64)
        self.latest_motor_faults = np.zeros(umc.NUM_MOTORS, dtype=np.int32)
        self.latest_can_timestamp = 0.0

        super().__init__(args)

        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.diagnostic_dir = (
            Path(args.diagnostic_dir).expanduser().resolve()
            if args.diagnostic_dir
            else self.output_dir.with_name(self.output_dir.name + "_diagnostics")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self.safety_log_path = self.output_dir.parent / "safety_events.jsonl"
        self.ep = next_contiguous_index(self.output_dir, "episode")
        self.failure_ep = next_contiguous_index(self.diagnostic_dir, "failure")

        self.model_left_limits = self.left_limits.copy()
        for index, (joint_name, _, _) in enumerate(umc.ARM_MOTOR_CONFIG):
            soft_lo, soft_hi = umc.SOFT_POSITION_LIMITS[joint_name]
            self.left_limits[index, 0] = max(
                self.left_limits[index, 0], soft_lo
            )
            self.left_limits[index, 1] = min(
                self.left_limits[index, 1], soft_hi
            )
        if np.any(self.left_limits[:, 0] >= self.left_limits[:, 1]):
            raise RuntimeError("MuJoCo and hardware joint limits do not overlap")

        if args.motor_freq is not None:
            umc.MOTOR_CTRL_FREQ = float(args.motor_freq)

        bridge_type = DryRunHardwareBridge if args.dry_run else HardwareMotorBridge
        bridge_args = {
            "kp": args.kp,
            "kd": args.kd,
            "motor_smoothing": args.motor_smoothing,
            "device_sn": args.device_sn,
            "enable_gripper": not args.disable_gripper,
            "calibration_record": args.calibration_record,
            "require_control_tests": not args.skip_mapped_control_check,
        }
        if args.dry_run:
            bridge_args = {
                "enable_gripper": not args.disable_gripper,
                "control_frequency": args.motor_freq or umc.MOTOR_CTRL_FREQ,
            }
        self.hardware_bridge = bridge_type(**bridge_args)

        self.mode = self.DISARMED
        self.keyboard = None
        self.calibration_q = None
        self.calibration_hand_pose = None
        self.calibration_tcp_pose = None
        self.command_q = self.q.copy()
        self.last_action_timestamp = time.monotonic()
        self.motion_active = False
        self.needs_rebase = False
        self.require_enable_release = False
        self.tracking_valid = False
        self.tracking_loss_since = None
        self.latest_left_hand = None
        self.tracking_error_since = None
        self.return_stable_since = None
        self.scene_stable_since = time.monotonic()
        self.target_position_error = 0.0
        self.target_rotation_error = 0.0
        self.target_clamped = False
        self.target_clamp_active = False
        self.targets_complete_after_return = False
        self.exit_after_return = False
        self.run_config = _jsonable_config(args)
        self.episode_stats = None
        self.control_disabled_since = None
        self.left_chain_bodies = []
        for joint_name in sim_collector.LEFT_ARM:
            joint_id = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            body_id = int(self.m.jnt_bodyid[joint_id])
            if not self.left_chain_bodies or self.left_chain_bodies[-1] != body_id:
                self.left_chain_bodies.append(body_id)

        self._reset_virtual_scene()
        print(
            f"[data] success={self.output_dir} next=episode_{self.ep}; "
            f"diagnostics={self.diagnostic_dir} next=failure_{self.failure_ep}"
        )
        if args.dry_run:
            print("[dry-run] No CAN device or real motor will be used")

    def _motor_gripper_target(self):
        if self.a.disable_gripper:
            return 0.0
        normalized = np.clip(self.g / sim_collector.GRIPPER_OPEN, 0.0, 1.0)
        return float(normalized * self.a.gripper_motor_open_pos)

    def _normalized_gripper_feedback(self, position):
        if self.a.disable_gripper:
            return float(np.clip(self.g / sim_collector.GRIPPER_OPEN, 0.0, 1.0))
        return float(np.clip(position / self.a.gripper_motor_open_pos, 0.0, 1.0))

    def _normalized_gripper_velocity(self, velocity):
        if self.a.disable_gripper:
            return 0.0
        return float(velocity / self.a.gripper_motor_open_pos)

    def _bridge_feedback_timestamp(self):
        getter = getattr(self.hardware_bridge, "get_feedback_timestamp", None)
        if getter is not None:
            return float(getter())

        timestamps = []
        active_count = umc.ARM_DOF + (0 if self.a.disable_gripper else 1)
        for index in range(active_count):
            motor = self.hardware_bridge._control.getMotor(
                self.hardware_bridge._can_ids[index]
            )
            if motor is not None:
                timestamps.append(float(motor.last_time_))
        return min(timestamps) if timestamps else 0.0

    def _sent_motor_target(self):
        getter = getattr(self.hardware_bridge, "get_sent_target", None)
        if getter is not None:
            return np.asarray(getter(), dtype=np.float64)
        with self.hardware_bridge._lock:
            return self.hardware_bridge._last_sent_q.copy()

    def _read_hardware_feedback(self, require_running=True):
        if self.hardware_bridge is None:
            raise RuntimeError("Hardware bridge is not initialized")
        if require_running and not self.hardware_bridge._running.is_set():
            raise RuntimeError("Hardware control thread stopped")

        positions, velocities, torques, faults = self.hardware_bridge.get_state()
        active_count = umc.ARM_DOF + (0 if self.a.disable_gripper else 1)
        active = slice(0, active_count)
        if not (
            np.isfinite(positions[active]).all()
            and np.isfinite(velocities[active]).all()
            and np.isfinite(torques[active]).all()
        ):
            raise RuntimeError("Non-finite CAN feedback received")
        if np.any(faults[active] != 0):
            labels = [
                f"{index}=0x{int(status):X}"
                for index, status in enumerate(faults[:active_count])
                if status != 0
            ]
            raise RuntimeError("Motor fault: " + ", ".join(labels))

        feedback_timestamp = self._bridge_feedback_timestamp()
        if (
            require_running
            and feedback_timestamp > 0.0
            and time.monotonic() - feedback_timestamp > self.a.can_timeout
        ):
            raise TimeoutError(
                f"CAN feedback stale for "
                f"{time.monotonic() - feedback_timestamp:.3f}s"
            )

        self.latest_motor_q = np.asarray(positions, dtype=np.float64)
        self.latest_motor_dq = np.asarray(velocities, dtype=np.float64)
        self.latest_motor_torque = np.asarray(torques, dtype=np.float64)
        self.latest_motor_faults = np.asarray(faults, dtype=np.int32)
        self.latest_can_timestamp = feedback_timestamp
        return self.latest_motor_q, self.latest_motor_dq

    def _wait_for_arm_feedback(self):
        deadline = time.monotonic() + self.a.arm_feedback_timeout
        last_age = float("inf")
        while time.monotonic() < deadline:
            self._read_hardware_feedback(require_running=False)
            now = time.monotonic()
            if self.latest_can_timestamp > 0.0:
                last_age = now - self.latest_can_timestamp
                if last_age <= self.a.can_timeout:
                    return
            if not self.hardware_bridge._running.is_set():
                raise RuntimeError(
                    "Hardware control thread stopped while arming"
                )
            time.sleep(0.01)
        raise TimeoutError(
            "No consecutive fresh CAN feedback after motor enable; "
            f"last age={last_age:.3f}s"
        )

    def _mirror_model_from_hardware(self, positions=None, velocities=None):
        positions = self.latest_motor_q if positions is None else positions
        velocities = self.latest_motor_dq if velocities is None else velocities
        tolerance = umc.ZERO_VERIFY_TOLERANCE_RAD

        for index, (qpos_address, dof_address) in enumerate(zip(self.lq, self.lv)):
            lo, hi = self.model_left_limits[index]
            position = float(positions[index])
            if position < lo - tolerance or position > hi + tolerance:
                joint_name = umc.ARM_MOTOR_CONFIG[index][0]
                raise RuntimeError(
                    f"Hardware {joint_name}={position:+.4f} is outside MuJoCo "
                    f"range [{lo:+.4f}, {hi:+.4f}]"
                )
            self.d.qpos[qpos_address] = np.clip(position, lo, hi)
            self.d.qvel[dof_address] = float(velocities[index])

        if not self.a.disable_gripper:
            gripper = (
                self._normalized_gripper_feedback(positions[umc.ARM_DOF])
                * sim_collector.GRIPPER_OPEN
            )
            for qpos_address in self.lf:
                self.d.qpos[qpos_address] = gripper
            self.d.qvel[self.lfv] = (
                self._normalized_gripper_velocity(velocities[umc.ARM_DOF])
                * sim_collector.GRIPPER_OPEN
            )
            self.g = gripper

        mujoco.mj_forward(self.m, self.d)

    def _synchronize_control_from_feedback(self):
        self._mirror_model_from_hardware()
        self.q = self.latest_motor_q[:umc.ARM_DOF].copy()
        self.command_q = self.q.copy()
        self.last_q_velocity = np.zeros(umc.ARM_DOF)
        self.last_teleop = None
        self.hand_ref = None
        self.eref = None
        self.cal = False
        self.calibrate_at = None
        self.motion_active = False
        self.needs_rebase = False
        self.controls(force=True)

    def _prepare_bridge_rearm(self):
        thread = getattr(self.hardware_bridge, "_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        if thread is not None and not thread.is_alive():
            self.hardware_bridge._thread = None

    def arm_hardware(self):
        if self.hardware_armed:
            print("[motor] Hardware is already armed")
            return
        if self.keyboard is None:
            print("[motor] Refusing to arm: keyboard safety input is unavailable")
            return
        if not self.keyboard.healthy:
            print("[motor] Reopening keyboard safety input...")
            self.keyboard.stop()
            self.keyboard = KeyboardSafetyMonitor(self.a.keyboard_device)
            try:
                self.keyboard.start()
            except Exception as exc:
                print(f"[motor] Keyboard safety input is still unavailable: {exc}")
                return
        if any(self.keyboard.is_pressed(key) for key in ("e", "i", "r")):
            print("[motor] Release E, I and R before pressing A")
            return

        print("[motor] Arming and synchronizing from fresh feedback...")
        try:
            self._prepare_bridge_rearm()
            self.hardware_bridge.start()
            self._wait_for_arm_feedback()
            self._synchronize_control_from_feedback()
        except Exception as exc:
            self._safety_fault("arm_failed", exc)
            return

        self.hardware_armed = True
        self.hardware_estopped = False
        self.mode = self.ARMED
        print("[motor] Armed at current pose; press P to calibrate VR")

    def calibrate(self):
        if not self.hardware_armed:
            print("[calib] Press A first")
            return
        if self.mode not in (self.ARMED, self.READY_MODE):
            print(f"[calib] P is unavailable in state {self.mode}")
            return
        if self.keyboard.is_pressed("i"):
            print("[calib] Release I before pressing P")
            return
        try:
            self._read_hardware_feedback()
            self._mirror_model_from_hardware()
        except Exception as exc:
            self._safety_fault("calibration_feedback_failed", exc)
            return
        self.q = self.latest_motor_q[:umc.ARM_DOF].copy()
        self.command_q = self.q.copy()
        self.last_q_velocity[:] = 0.0
        super().calibrate()
        self.mode = self.CALIBRATING

    def finish_calibration(self, head, left_hand):
        if self.mode != self.CALIBRATING:
            return
        if not self.tracking_valid:
            return
        was_calibrated = self.cal
        super().finish_calibration(head, left_hand)
        if not was_calibrated and self.cal:
            self.calibration_q = self.latest_motor_q[:umc.ARM_DOF].copy()
            self.calibration_hand_pose = self.hand_ref.copy()
            self.calibration_tcp_pose = self.eref.copy()
            self.command_q = self.calibration_q.copy()
            self.mode = self.READY_MODE
            self.scene_stable_since = None
            print("[calib] Calibration pose stored for I+R return")
        elif not self.cal and self.calibrate_at is None:
            self.mode = self.ARMED
            print("[calib] Calibration capture failed; press P to retry")

    def _rebase_control(self, left_hand):
        left_hand = sim_collector.rigid_pose(left_hand)
        if left_hand is None:
            return False
        self._mirror_model_from_hardware()
        hand_current = self.robotbase_vuer @ left_hand
        hand_current[:3, :3] = sim_collector.rot(hand_current[:3, :3])
        tcp_position, tcp_rotation = sim_collector.site_pose(self.d, self.tcp)
        self.hand_ref = hand_current.copy()
        self.eref = self.robotbase_world @ sim_collector.pose(
            tcp_position, tcp_rotation
        )
        self.q = self.latest_motor_q[:umc.ARM_DOF].copy()
        self.command_q = self.q.copy()
        self.last_q_velocity[:] = 0.0
        self.last_teleop = time.monotonic()
        self.needs_rebase = False
        self.last_action_timestamp = self.last_teleop
        print("[control] Reference rebased at held TCP pose")
        return True

    def _freeze_motion_target(self):
        if not self.hardware_armed:
            return
        sent = self._sent_motor_target()
        self.command_q = sent[:umc.ARM_DOF].copy()
        self.hardware_bridge.set_target(sent)
        self.motion_active = False
        self.needs_rebase = bool(self.cal)

    def _send_smoothed_target(self, desired_q, dt):
        alpha = 1.0 - np.exp(-dt / self.a.command_smoothing_tau)
        self.command_q += alpha * (np.asarray(desired_q) - self.command_q)
        lower = self.left_limits[:, 0] + self.a.joint_limit_margin
        upper = self.left_limits[:, 1] - self.a.joint_limit_margin
        self.command_q = np.clip(self.command_q, lower, upper)
        target = np.append(self.command_q, self._motor_gripper_target())
        self.hardware_bridge.set_target(target)
        self.last_action_timestamp = time.monotonic()
        self.motion_active = True

    def incremental_ik(self, target_p_base, target_r_base, dt):
        q_actual = self.d.qpos[self.lq].copy()
        self.d.qpos[self.lq] = self.q
        mujoco.mj_forward(self.m, self.d)
        current_p, current_r = sim_collector.site_pose(self.d, self.tcp)
        target_p = (
            self.world_robotbase[:3, :3] @ target_p_base
            + self.world_robotbase[:3, 3]
        )
        target_r = self.world_robotbase[:3, :3] @ target_r_base
        position_error = target_p - current_p
        rotation_error = sim_collector.rotvec(target_r @ current_r.T)
        self.target_position_error = float(np.linalg.norm(position_error))
        self.target_rotation_error = float(np.linalg.norm(rotation_error))

        jacobian_position = np.zeros((3, self.m.nv))
        jacobian_rotation = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(
            self.m,
            self.d,
            jacobian_position,
            jacobian_rotation,
            self.tcp,
        )
        orientation_weight = self.a.ik_orientation_weight
        jacobian = np.vstack(
            (
                jacobian_position[:, self.lv],
                orientation_weight * jacobian_rotation[:, self.lv],
            )
        )
        self.d.qpos[self.lq] = q_actual
        mujoco.mj_forward(self.m, self.d)

        linear_velocity = sim_collector.clip_norm(
            self.a.tcp_linear_gain * position_error,
            self.a.tcp_max_linear_speed,
        )
        angular_velocity = sim_collector.clip_norm(
            self.a.tcp_angular_gain * rotation_error,
            self.a.tcp_max_angular_speed,
        )
        requested_velocity = np.r_[
            linear_velocity, orientation_weight * angular_velocity
        ]
        speed_limit = np.full(umc.ARM_DOF, self.a.joint_max_speed)
        acceleration_limit = self.a.joint_max_acceleration * dt
        lower_limit = self.left_limits[:, 0] + self.a.joint_limit_margin
        upper_limit = self.left_limits[:, 1] - self.a.joint_limit_margin
        lower = np.maximum.reduce(
            (
                -speed_limit,
                self.last_q_velocity - acceleration_limit,
                (lower_limit - self.q) / dt,
            )
        )
        upper = np.minimum.reduce(
            (
                speed_limit,
                self.last_q_velocity + acceleration_limit,
                (upper_limit - self.q) / dt,
            )
        )
        dq = sim_collector.bounded_weighted_dls(
            jacobian,
            requested_velocity,
            self.joint_weight_matrix,
            self.a.ik_damping,
            lower,
            upper,
        )
        previous_q = self.q.copy()
        self.q = np.clip(previous_q + dq * dt, lower_limit, upper_limit)
        self.last_q_velocity = (self.q - previous_q) / dt
        near_limit = np.any(
            (self.q - lower_limit < 1e-4) | (upper_limit - self.q < 1e-4)
        )
        stalled = self.target_position_error > 0.02 and np.linalg.norm(dq) < 1e-3
        self.target_clamped = bool(near_limit or stalled)
        return True

    def teleop(self, left_hand):
        if not self.cal or not self.tracking_valid:
            return
        super().teleop(left_hand)

    def _control_tick(self, left_hand, now, dt):
        if not self.hardware_armed or self.mode == self.FAULT:
            return

        enable = self.keyboard.is_pressed("i")
        returning = self.keyboard.is_pressed("r")
        if self.require_enable_release:
            if not enable:
                self.require_enable_release = False
            else:
                self._freeze_motion_target()
                return

        if self.mode == self.RETURN_REQUIRED:
            if enable and returning:
                self.q = self.calibration_q.copy()
                self._send_smoothed_target(self.calibration_q, dt)
                self._update_return_completion(now)
            else:
                if self.motion_active:
                    self._freeze_motion_target()
                self.return_stable_since = None
            return

        requested = (
            self.mode in (self.READY_MODE, self.RECORDING)
            and enable
            and not returning
            and self.tracking_valid
        )
        if not requested:
            if self.motion_active:
                self._freeze_motion_target()
                self.needs_rebase = self.cal
            return

        if self.needs_rebase and not self._rebase_control(left_hand):
            self._freeze_motion_target()
            return
        self.teleop(left_hand)
        self._send_smoothed_target(self.q, dt)

    def _update_return_completion(self, now):
        error = np.abs(
            self.latest_motor_q[:umc.ARM_DOF] - self.calibration_q
        )
        velocity = np.abs(self.latest_motor_dq[:umc.ARM_DOF])
        settled = (
            np.all(error <= np.deg2rad(self.a.return_tolerance_deg))
            and np.all(velocity <= self.a.return_velocity_tolerance)
        )
        if not settled:
            self.return_stable_since = None
            return
        if self.return_stable_since is None:
            self.return_stable_since = now
            return
        if now - self.return_stable_since < self.a.return_settle_time:
            return

        self._freeze_motion_target()
        self._reset_virtual_scene()
        self.mode = self.READY_MODE
        self.state = self.READY
        self.require_enable_release = True
        self.needs_rebase = True
        self.return_stable_since = None
        print("[return] Calibration pose reached; release I before continuing")
        if self.targets_complete_after_return:
            self.exit_after_return = True

    def _reset_virtual_scene(self):
        x, y = self.current_cup_xy()
        sim_collector.reset_episode_state(
            self.m,
            self.d,
            self.rq,
            self.lq,
            self.rf,
            self.cup,
            self.cqa,
            x,
            y,
        )
        if self.hardware_armed:
            self._mirror_model_from_hardware()
            self.q = self.latest_motor_q[:umc.ARM_DOF].copy()
            self.command_q = self.q.copy()
        else:
            for index, qpos_address in enumerate(self.lq):
                self.d.qpos[qpos_address] = self.home_q[index]
            self.q = self.home_q.copy()
            self.command_q = self.q.copy()
        for qpos_address in self.lf + self.rf:
            self.d.qpos[qpos_address] = sim_collector.GRIPPER_OPEN
        self.g = sim_collector.GRIPPER_OPEN
        self.last_q_velocity[:] = 0.0
        self.grasp_since = None
        self.grasped = False
        self.basket_since = None
        self.scene_stable_since = None
        self.t = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.controls(force=True)
        print(f"[scene] Cup reset to ({x:.3f}, {y:.3f})")

    def _cup_is_stable(self):
        joint_id = int(self.m.body_jntadr[self.cup])
        dof_address = int(self.m.jnt_dofadr[joint_id])
        velocity = self.d.qvel[dof_address:dof_address + 6]
        return bool(
            np.linalg.norm(velocity[:3]) < 0.02
            and np.linalg.norm(velocity[3:]) < 0.2
        )

    def _update_scene_stability(self, now):
        if self.mode != self.READY_MODE:
            return
        if self._cup_is_stable():
            if self.scene_stable_since is None:
                self.scene_stable_since = now
        else:
            self.scene_stable_since = None

    def _scene_ready(self, now):
        return (
            self.scene_stable_since is not None
            and now - self.scene_stable_since >= self.a.scene_settle_time
        )

    def _new_episode_stats(self):
        return {
            "control_release_count": 0,
            "control_disabled_duration": 0.0,
            "vr_loss_count": 0,
            "vr_loss_duration": 0.0,
            "target_clamp_count": 0,
            "max_tracking_error_rad": 0.0,
        }

    def start(self):
        now = time.monotonic()
        reasons = []
        if not self.hardware_armed:
            reasons.append("press A")
        if not self.cal or self.mode != self.READY_MODE:
            reasons.append("press P / finish return")
        if not self.keyboard.is_pressed("i"):
            reasons.append("hold I")
        if self.keyboard.is_pressed("r"):
            reasons.append("release R")
        if not self.tracking_valid:
            reasons.append("restore VR hand tracking")
        if self.require_enable_release:
            reasons.append("release I once")
        if not self._scene_ready(now):
            reasons.append("wait for cup to settle")
        if reasons:
            print("[record] Cannot start: " + "; ".join(reasons))
            return

        self.mode = self.RECORDING
        self.state = self.REC
        self.next = self.t
        self.grasp_since = None
        self.grasped = False
        self.basket_since = None
        self.episode_stats = self._new_episode_stats()
        self.control_disabled_since = None
        self.target_clamp_active = False
        self.buf = {
            "action": [],
            "qpos": [],
            "qvel": [],
            "images": {"vr_center": [], "wrist": []},
            "camera_names": ["vr_center", "wrist"],
            "initial_object_pose": self.d.qpos[self.cqa:self.cqa + 7].copy(),
            "initial_motor_position": self.latest_motor_q.copy(),
            "calibration_q": self.calibration_q.copy(),
            "calibration_hand_pose": self.calibration_hand_pose.copy(),
            "calibration_tcp_pose": self.calibration_tcp_pose.copy(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_config": self.run_config,
            "summary": {},
            "diagnostics": {
                "motor_command_q": [],
                "tracking_error": [],
                "control_enabled": [],
                "vr_tracking_valid": [],
                "target_clamped": [],
                "tcp_residual": [],
            },
            "timestamps": {
                "record": [],
                "can": [],
                "action": [],
                "vr_center": [],
                "wrist": [],
            },
        }
        target = self.current_failure_target()
        if target is not None:
            self.buf.update(
                {
                    "source_evaluation_file": str(self.failure_source),
                    "source_evaluation_rollout_id": target["rollout_id"],
                    "source_evaluation_outcome": target["outcome"],
                }
            )
        print("[record] START: CAN qpos/qvel + IK action + MuJoCo RGB")

    def _close_episode_stats(self, now):
        if self.control_disabled_since is not None:
            self.episode_stats["control_disabled_duration"] += (
                now - self.control_disabled_since
            )
            self.control_disabled_since = now
        if self.tracking_loss_since is not None:
            self.episode_stats["vr_loss_duration"] += (
                now - self.tracking_loss_since
            )
        self.buf["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.buf["summary"] = dict(self.episode_stats)

    def _finish_success(self):
        now = time.monotonic()
        self._close_episode_stats(now)
        save_real_episode(
            self.output_dir,
            "episode",
            self.ep,
            self.buf,
            "success",
        )
        self.ep = next_contiguous_index(self.output_dir, "episode")
        self.buf = None
        self.episode_stats = None
        self.state = self.PENDING
        self.mode = self.RETURN_REQUIRED
        self._freeze_motion_target()
        if self.advance_failure_target():
            self.targets_complete_after_return = True
            print("[targets] All requested failure poses have been recorded")
        print("[record] SUCCESS; hold I+R to return to calibration pose")

    def _finish_diagnostic(self, outcome):
        now = time.monotonic()
        self._close_episode_stats(now)
        save_real_episode(
            self.diagnostic_dir,
            "failure",
            self.failure_ep,
            self.buf,
            outcome,
        )
        self.failure_ep = next_contiguous_index(
            self.diagnostic_dir, "failure"
        )
        self.buf = None
        self.episode_stats = None
        self.state = self.PENDING
        self.mode = self.RETURN_REQUIRED
        self._freeze_motion_target()
        print(f"[record] {outcome}; diagnostic saved; hold I+R to return")

    def _discard_episode(self, reason, require_return=True):
        if self.mode == self.RECORDING:
            print(f"[record] Discarded current episode: {reason}")
        if self.hardware_armed:
            self._freeze_motion_target()
        self.buf = None
        self.episode_stats = None
        self.control_disabled_since = None
        self.state = self.PENDING if require_return else self.READY
        if require_return and self.hardware_armed and self.calibration_q is not None:
            self.mode = self.RETURN_REQUIRED
            print("[return] Hold I+R to return to calibration pose")
        elif self.hardware_armed:
            self.mode = self.READY_MODE

    def _log_safety_event(self, event_type, detail=None):
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_time": time.monotonic(),
            "event": event_type,
            "detail": None if detail is None else str(detail),
            "mode": self.mode,
            "can_timestamp": self.latest_can_timestamp,
            "qpos": self.latest_motor_q.tolist(),
            "qvel": self.latest_motor_dq.tolist(),
            "motor_target": (
                self._sent_motor_target().tolist()
                if self.hardware_bridge is not None
                else None
            ),
            "run_config": self.run_config,
        }
        append_jsonl(self.safety_log_path, record)

    def _safety_fault(self, event_type, detail=None):
        running_event = (
            getattr(self.hardware_bridge, "_running", None)
            if self.hardware_bridge is not None
            else None
        )
        bridge_running = (
            running_event is not None and running_event.is_set()
        )
        if (
            self.mode == self.FAULT
            and self.hardware_estopped
            and not bridge_running
        ):
            return
        try:
            self._log_safety_event(event_type, detail)
        except Exception as log_error:
            print(f"[safety] Failed to write safety log: {log_error}", file=sys.stderr)
        if self.mode == self.RECORDING:
            self._discard_episode(event_type, require_return=False)
        if self.hardware_bridge is not None:
            try:
                self.hardware_bridge.emergency_stop()
            except Exception as stop_error:
                print(f"[safety] Emergency stop error: {stop_error}", file=sys.stderr)
        self.hardware_armed = False
        self.hardware_estopped = True
        self.cal = False
        self.calibrate_at = None
        self.calibration_q = None
        self.motion_active = False
        self.mode = self.FAULT
        self.state = self.READY
        print(f"[safety] FAULT {event_type}: {detail or ''}", file=sys.stderr)
        print("[safety] Resolve the fault, release I/R, then press A and P")

    def _update_tracking(self, raw_left_hand, left_hand, now):
        update_time = float(getattr(self.tv, "left_hand_update_time", 0.0))
        raw_valid = sim_collector.rigid_pose(raw_left_hand) is not None
        fresh = update_time > 0.0 and now - update_time <= self.a.vr_stale_timeout
        valid = bool(raw_valid and fresh)
        self.latest_left_hand = left_hand

        if valid:
            if not self.tracking_valid:
                if self.tracking_loss_since is not None:
                    duration = now - self.tracking_loss_since
                    if self.episode_stats is not None:
                        self.episode_stats["vr_loss_duration"] += duration
                    print(f"[vr] Left-hand tracking recovered after {duration:.3f}s")
                self.needs_rebase = self.cal
            self.tracking_valid = True
            self.tracking_loss_since = None
            return

        if self.tracking_valid or self.tracking_loss_since is None:
            if fresh:
                loss_start = now
            else:
                loss_start = max(0.0, update_time)
                if update_time <= 0.0:
                    loss_start = now
            self.tracking_loss_since = loss_start
            if self.episode_stats is not None:
                self.episode_stats["vr_loss_count"] += 1
            print("[vr] Left-hand tracking lost; target frozen")
        self.tracking_valid = False
        if self.motion_active:
            self._freeze_motion_target()

        duration = now - self.tracking_loss_since
        if (
            self.mode == self.RECORDING
            and duration >= self.a.vr_loss_discard_time
        ):
            self._discard_episode("vr_tracking_loss", require_return=True)

    def _process_keyboard_events(self):
        for event, key, event_time in self.keyboard.drain_events():
            if event == "fault":
                self._safety_fault("keyboard_input_failed", self.keyboard.error)
                continue
            if key == "e" and event == "down":
                self._safety_fault("keyboard_estop")
                continue
            if key != "i" or self.episode_stats is None:
                continue
            if event == "up" and self.control_disabled_since is None:
                self.episode_stats["control_release_count"] += 1
                self.control_disabled_since = event_time
            elif event == "down" and self.control_disabled_since is not None:
                self.episode_stats["control_disabled_duration"] += (
                    event_time - self.control_disabled_since
                )
                self.control_disabled_since = None

    def _safety_checks(self, now):
        if not self.hardware_armed:
            return
        if not self.hardware_bridge._running.is_set():
            self._safety_fault("motor_bridge_stopped")
            return
        if (
            self.latest_can_timestamp <= 0.0
            or now - self.latest_can_timestamp > self.a.can_timeout
        ):
            self._safety_fault(
                "can_timeout",
                f"age={now - self.latest_can_timestamp:.3f}s",
            )
            return

        sent = self._sent_motor_target()[:umc.ARM_DOF]
        error = np.abs(sent - self.latest_motor_q[:umc.ARM_DOF])
        max_error = float(np.max(error))
        if self.episode_stats is not None:
            self.episode_stats["max_tracking_error_rad"] = max(
                self.episode_stats["max_tracking_error_rad"], max_error
            )
        if max_error > np.deg2rad(self.a.tracking_error_deg):
            if self.tracking_error_since is None:
                self.tracking_error_since = now
            elif (
                now - self.tracking_error_since
                >= self.a.tracking_error_duration
            ):
                self._safety_fault(
                    "tracking_error",
                    f"max={np.rad2deg(max_error):.2f}deg",
                )
        else:
            self.tracking_error_since = None

    def controls(self, force=False):
        left = (
            self.latest_motor_q[:umc.ARM_DOF]
            if self.hardware_armed
            else self.d.qpos[self.lq]
        )
        for index, actuator in enumerate(self.la):
            self.d.ctrl[actuator] = left[index]
        for index, actuator in enumerate(self.ra):
            self.d.ctrl[actuator] = sim_collector.HOME_Q[index]
        self.d.ctrl[self.lg] = self.g
        self.d.ctrl[self.rg] = sim_collector.GRIPPER_OPEN

    def _step_mirrored_physics(self, steps):
        for _ in range(steps):
            if self.hardware_armed:
                for index, (qpos_address, dof_address) in enumerate(
                    zip(self.lq, self.lv)
                ):
                    self.d.qpos[qpos_address] = self.latest_motor_q[index]
                    self.d.qvel[dof_address] = self.latest_motor_dq[index]
            self.controls(force=True)
            mujoco.mj_step(self.m, self.d)
        if self.hardware_armed:
            self._mirror_model_from_hardware()

    def _record_target_clamp_sample(self):
        if self.target_clamped and not self.target_clamp_active:
            self.episode_stats["target_clamp_count"] += 1
        self.target_clamp_active = bool(self.target_clamped)

    def _target_ghost_points(self):
        if not self.a.teleop_debug or not self.cal:
            return None
        actual_q = self.d.qpos[self.lq].copy()
        actual_dq = self.d.qvel[self.lv].copy()
        self.d.qpos[self.lq] = self.q
        self.d.qvel[self.lv] = 0.0
        mujoco.mj_forward(self.m, self.d)
        points = [self.d.xpos[body].copy() for body in self.left_chain_bodies]
        points.append(self.d.site_xpos[self.tcp].copy())
        self.d.qpos[self.lq] = actual_q
        self.d.qvel[self.lv] = actual_dq
        mujoco.mj_forward(self.m, self.d)
        return points

    @staticmethod
    def _append_target_ghost(scene, points):
        if points is None:
            return
        for start, end in zip(points[:-1], points[1:]):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([0.0, 0.8, 1.0, 0.25], dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.012,
                start,
                end,
            )
            scene.ngeom += 1

    def _render_camera(self, name, renderer):
        renderer.update_scene(self.d, camera=name)
        self._append_target_ghost(renderer.scene, self._target_ghost_points())
        return renderer.render().copy()

    def render(self, head):
        direction = (
            np.eye(3)
            if not self.cal
            else sim_collector.rot(
                (self.world_vuer @ head)[:3, :3] @ self.head_rot.T
            )
        )
        points = self._target_ghost_points()
        self.gl.make_current()
        for camera, offset, context, output in (
            (self.cl, self.off[0], self.rl, self.img[:, :640]),
            (self.cr, self.off[1], self.rr, self.img[:, 640:]),
        ):
            position = self.hp + direction @ offset
            sim_collector.setcam(
                camera, position, position + direction @ self.f
            )
            mujoco.mjv_updateScene(
                self.m,
                self.d,
                self.op,
                None,
                camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.sc,
            )
            self._append_target_ghost(self.sc, points)
            mujoco.mjr_render(self.vp, self.sc, context)
            image = np.empty((480, 640, 3), np.uint8)
            mujoco.mjr_readPixels(image, None, self.vp, context)
            output[:] = image[::-1]
        if self.a.show_wrist:
            wrist = self._render_camera("wrist", self.ren["wrist"])
            cv2.imshow("Wrist RGB", cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        if (
            self.a.teleop_debug
            and time.monotonic() - self.log > 5.0
        ):
            self.log = time.monotonic()
            print(
                f"[vr] frame max={self.img.max()} "
                f"nonzero={self.img.any(axis=2).mean() * 100:.1f}%"
            )

    def frame(self):
        if self.mode != self.RECORDING or self.t < self.next:
            return

        while self.next <= self.t:
            self.next += 1.0 / self.a.record_fps
        buffer = self.buf
        record_timestamp = time.monotonic()

        action = np.r_[
            self.q,
            self.g / sim_collector.GRIPPER_OPEN,
            sim_collector.HOME_Q,
            1.0,
        ]
        qpos = np.r_[
            self.latest_motor_q[:umc.ARM_DOF],
            self._normalized_gripper_feedback(
                self.latest_motor_q[umc.ARM_DOF]
            ),
            [self.d.qpos[address] for address in self.rq],
            self.d.qpos[self.rf[0]] / sim_collector.GRIPPER_OPEN,
        ]
        qvel = np.r_[
            self.latest_motor_dq[:umc.ARM_DOF],
            self._normalized_gripper_velocity(
                self.latest_motor_dq[umc.ARM_DOF]
            ),
            [self.d.qvel[address] for address in self.rv],
            self.d.qvel[self.rfv] / sim_collector.GRIPPER_OPEN,
        ]
        buffer["action"].append(action)
        buffer["qpos"].append(qpos)
        buffer["qvel"].append(qvel)

        for name, renderer in self.ren.items():
            image = self._render_camera(name, renderer)
            image_timestamp = time.monotonic()
            buffer["images"][name].append(image)
            buffer["timestamps"][name].append(image_timestamp)
            if name == "wrist":
                self.wrist_frame = image

        sent = self._sent_motor_target()
        tracking_error = (
            sent[:umc.ARM_DOF] - self.latest_motor_q[:umc.ARM_DOF]
        )
        enabled = int(
            self.motion_active
            and self.keyboard.is_pressed("i")
            and self.tracking_valid
        )
        diagnostics = buffer["diagnostics"]
        diagnostics["motor_command_q"].append(sent)
        diagnostics["tracking_error"].append(tracking_error)
        diagnostics["control_enabled"].append(enabled)
        diagnostics["vr_tracking_valid"].append(int(self.tracking_valid))
        diagnostics["target_clamped"].append(int(self.target_clamped))
        diagnostics["tcp_residual"].append(
            [self.target_position_error, self.target_rotation_error]
        )
        self._record_target_clamp_sample()

        timestamps = buffer["timestamps"]
        timestamps["record"].append(record_timestamp)
        timestamps["can"].append(self.latest_can_timestamp)
        timestamps["action"].append(self.last_action_timestamp)

        contact, _ = sim_collector.pad_contact_summary(
            self.m, self.d, self.cg, self.pads
        )
        bilateral = contact[0] > 0 and contact[1] > 0
        if bilateral:
            if self.grasp_since is None:
                self.grasp_since = self.t
            if (
                not self.grasped
                and self.t - self.grasp_since
                >= sim_collector.GRASP_HOLD_SECONDS
            ):
                self.grasped = True
                print("[record] Stable simulated grasp acquired")
        else:
            self.grasp_since = None

        in_basket = self.grasped and self.in_basket()
        if in_basket and self.basket_since is None:
            self.basket_since = self.t
        elif not in_basket:
            self.basket_since = None

        if (
            self.basket_since is not None
            and self.t - self.basket_since
            >= sim_collector.BASKET_HOLD_SECONDS
        ):
            self._finish_success()
        elif len(buffer["action"]) >= sim_collector.MAX_TIMESTEPS:
            self._finish_diagnostic("timeout")

    def _handle_terminal_key(self, key):
        if key == "a":
            if self.mode == self.FAULT and (
                self.keyboard.is_pressed("i")
                or self.keyboard.is_pressed("r")
            ):
                print("[motor] Release I and R before re-arming")
                return True
            self.arm_hardware()
        elif key == "p":
            self.calibrate()
        elif key == " ":
            if self.mode == self.READY_MODE:
                self.start()
            elif self.mode == self.RECORDING:
                self._discard_episode("operator_abort", require_return=True)
        elif key == "o":
            if self.mode != self.READY_MODE or self.keyboard.is_pressed("i"):
                print("[scene] O requires READY state with I released")
            else:
                error = np.max(
                    np.abs(
                        self.latest_motor_q[:umc.ARM_DOF]
                        - self.calibration_q
                    )
                )
                if error > np.deg2rad(self.a.return_tolerance_deg):
                    print("[scene] Return with I+R before resetting the scene")
                else:
                    self._reset_virtual_scene()
                    self.require_enable_release = True
                    self.needs_rebase = True
        elif key == "e":
            self._safety_fault("terminal_estop")
        elif key == "v":
            self._toggle_video()
        elif key == "d":
            print("[record] D is no longer needed; Space aborts and discards")
        elif key == "m":
            print("[panel] Manual MuJoCo arm control is disabled in real mode")
        elif key in ("q", "\x1b"):
            return False
        return True

    def run(self):
        print(
            "Real VR ACT collector: A arm, P calibrate, hold I control, "
            "Space record/abort, hold I+R return, E estop, Q/Esc quit"
        )
        print("[safety] Motors remain disabled until A is pressed")
        terminal_keys = []
        stop_reader = threading.Event()
        old_terminal = None
        terminal_thread = None
        try:
            self.keyboard = KeyboardSafetyMonitor(self.a.keyboard_device)
            self.keyboard.start()
            print(
                f"[safety] Global keyboard input: "
                f"{self.keyboard.device_name} ({self.keyboard.device})"
            )
            self.connect()

            old_terminal = (
                termios.tcgetattr(0) if sys.stdin.isatty() else None
            )
            if old_terminal:
                tty.setcbreak(0)

            def terminal_reader():
                while not stop_reader.is_set():
                    readable, _, _ = select.select(
                        [sys.stdin], [], [], 0.1
                    )
                    if readable:
                        key = sys.stdin.read(1)
                        if not key:
                            stop_reader.set()
                            return
                        terminal_keys.append(key.lower())

            terminal_thread = threading.Thread(
                target=terminal_reader, name="terminal-keys", daemon=True
            )
            terminal_thread.start()
            running = True
            wall_time = time.monotonic()
            self.next_control = wall_time
            last_control = wall_time

            with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
                while running and viewer.is_running():
                    raw_left = np.asarray(self.tv.left_hand, dtype=np.float64)
                    head, left_hand = self.v.read(self.tv)
                    now = time.monotonic()
                    self._update_tracking(raw_left, left_hand, now)
                    self._process_keyboard_events()

                    while terminal_keys:
                        running = self._handle_terminal_key(
                            terminal_keys.pop(0)
                        )
                        if not running:
                            break

                    if self.hardware_armed:
                        try:
                            self._read_hardware_feedback()
                            self._mirror_model_from_hardware()
                            self._safety_checks(now)
                        except Exception as exc:
                            self._safety_fault("hardware_feedback_failed", exc)

                    self.finish_calibration(head, left_hand)
                    if now >= self.next_control:
                        control_dt = np.clip(
                            now - last_control,
                            1e-4,
                            0.1,
                        )
                        last_control = now
                        self._control_tick(
                            left_hand, now, float(control_dt)
                        )
                        skipped = max(
                            1,
                            int(
                                (now - self.next_control)
                                * self.a.control_hz
                            )
                            + 1,
                        )
                        self.next_control += skipped / self.a.control_hz

                    self.controls()
                    steps = min(
                        50,
                        max(
                            1,
                            int(
                                (now - wall_time)
                                / self.m.opt.timestep
                            ),
                        ),
                    )
                    wall_time = now
                    self._step_mirrored_physics(steps)
                    self.t += steps * self.m.opt.timestep
                    self._update_scene_stability(now)
                    self.frame()
                    self.render(head)

                    if (
                        self.recording
                        and self.video_writer is not None
                        and self.t >= self.video_next_frame
                    ):
                        frame = self._render_camera(
                            "vr_center", self.video_renderer
                        )
                        self.video_writer.write(
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        )
                        self.video_next_frame += 1.0 / 30.0

                    viewer.sync()
                    if self.exit_after_return:
                        running = False
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print("\n[exit] Keyboard interrupt; shutting down safely")
        except Exception as exc:
            if self.hardware_armed:
                self._safety_fault("unhandled_exception", exc)
            raise
        finally:
            stop_reader.set()
            if terminal_thread is not None:
                terminal_thread.join(timeout=0.5)
            if old_terminal:
                termios.tcsetattr(
                    0, termios.TCSADRAIN, old_terminal
                )
            if self.mode == self.RECORDING:
                self._discard_episode(
                    "process_exit", require_return=False
                )
            if self.keyboard is not None:
                self.keyboard.stop()
            if self.hardware_bridge is not None:
                self.hardware_bridge.stop()
            self._close_resources()

    def _toggle_video(self):
        if not self.recording:
            path = self.output_dir / (
                f"video_{self.ep}_{int(time.time())}.mp4"
            )
            self.video_writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (self.a.image_width, self.a.image_height),
            )
            if self.video_writer.isOpened():
                self.recording = True
                self.video_next_frame = self.t
                print(f"[video] REC: {path}")
            else:
                print(f"[video] Cannot open {path}")
                self.video_writer = None
        else:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("[video] stopped")

    def _close_resources(self):
        def close_safely(label, callback):
            try:
                callback()
            except Exception as exc:
                print(f"[cleanup] {label}: {exc}", file=sys.stderr)

        for name, renderer in self.ren.items():
            close_safely(f"renderer {name}", renderer.close)
        close_safely("video renderer", self.video_renderer.close)
        if self.video_writer is not None:
            close_safely("video writer", self.video_writer.release)
        if self.a.show_wrist:
            close_safely(
                "wrist window",
                lambda: cv2.destroyWindow("Wrist RGB"),
            )

        if hasattr(self, "tv") and hasattr(self.tv, "process"):
            if self.tv.process.is_alive():
                close_safely(
                    "VR process terminate",
                    self.tv.process.terminate,
                )
                close_safely(
                    "VR process join",
                    lambda: self.tv.process.join(timeout=2.0),
                )
        if hasattr(self, "rl"):
            close_safely("left render context", self.rl.free)
        if hasattr(self, "rr"):
            close_safely("right render context", self.rr.free)
        if hasattr(self, "gl"):
            close_safely("GL context", self.gl.free)
        if hasattr(self, "sh"):
            close_safely("shared memory close", self.sh.close)
            try:
                self.sh.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect ACT episodes while VR-controls a real UPOO arm"
    )
    parser.add_argument(
        "--motor-enable",
        action="store_true",
        help="Required acknowledgement for real motor control",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use simulated motor feedback and never connect to CAN",
    )
    parser.add_argument("--device-sn", default=None)
    parser.add_argument("--kp", type=float, nargs=7, default=None)
    parser.add_argument("--kd", type=float, nargs=7, default=None)
    parser.add_argument(
        "--motor-smoothing", type=float, default=umc.MOTOR_SMOOTHING
    )
    parser.add_argument("--motor-freq", type=float, default=None)
    parser.add_argument(
        "--calibration-record",
        type=Path,
        default=umc.CALIBRATION_RECORD,
    )
    parser.add_argument("--skip-mapped-control-check", action="store_true")
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument(
        "--gripper-motor-open-pos",
        type=float,
        default=DEFAULT_GRIPPER_MOTOR_OPEN_POS,
    )
    parser.add_argument("--keyboard-device", type=Path)
    parser.add_argument("--home-joint-margin", type=float, default=0.0)
    parser.add_argument("--output-dir", default=str(REAL_OUTPUT_DIR))
    parser.add_argument("--diagnostic-dir", default=None)
    parser.add_argument(
        "--show-wrist",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--position-scale", type=float, default=0.75)
    parser.add_argument("--record-fps", type=int, default=50)
    parser.add_argument("--control-hz", type=float, default=60.0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--tcp-max-linear-speed", type=float, default=0.6)
    parser.add_argument("--tcp-max-angular-speed", type=float, default=2.5)
    parser.add_argument("--tcp-linear-gain", type=float, default=5.0)
    parser.add_argument("--tcp-angular-gain", type=float, default=2.0)
    parser.add_argument(
        "--joint-weights",
        type=float,
        nargs=6,
        default=sim_collector.DEFAULT_JOINT_WEIGHTS,
        metavar=("W1", "W2", "W3", "W4", "W5", "W6"),
    )
    parser.add_argument("--ik-damping", type=float, default=0.08)
    parser.add_argument("--ik-orientation-weight", type=float, default=0.35)
    parser.add_argument("--joint-max-speed", type=float, default=2.0)
    parser.add_argument("--joint-max-acceleration", type=float, default=10.0)
    parser.add_argument("--joint-limit-margin", type=float, default=0.0)
    parser.add_argument("--command-smoothing-tau", type=float, default=0.06)
    parser.add_argument("--arm-feedback-timeout", type=float, default=2.0)
    parser.add_argument(
        "--can-timeout",
        type=float,
        default=umc.CAN_TIMEOUT_SEC,
    )
    parser.add_argument("--tracking-error-deg", type=float, default=10.0)
    parser.add_argument("--tracking-error-duration", type=float, default=0.3)
    parser.add_argument("--vr-stale-timeout", type=float, default=0.25)
    parser.add_argument("--vr-loss-discard-time", type=float, default=0.5)
    parser.add_argument("--return-tolerance-deg", type=float, default=2.0)
    parser.add_argument(
        "--return-velocity-tolerance", type=float, default=0.05
    )
    parser.add_argument("--return-settle-time", type=float, default=0.3)
    parser.add_argument("--scene-settle-time", type=float, default=0.5)
    parser.add_argument("--gripper-open-distance", type=float, default=0.15)
    parser.add_argument("--gripper-smoothing-tau", type=float, default=0.08)
    parser.add_argument("--gripper-max-speed", type=float, default=0.12)
    parser.add_argument("--cup-y", type=float, default=None)
    parser.add_argument("--failure-rollouts", type=Path)
    parser.add_argument("--teleop-debug", action="store_true")

    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--ngrok", action="store_true")
    transport.add_argument(
        "--local-cert",
        action="store_true",
        help="Use local TLS; default when --ngrok is absent",
    )
    parser.add_argument("--cert-file", default=str(DEFAULT_CERT_FILE))
    parser.add_argument("--key-file", default=str(DEFAULT_KEY_FILE))

    args = parser.parse_args(argv)
    if args.motor_enable == args.dry_run:
        parser.error("choose exactly one of --motor-enable or --dry-run")
    if args.gripper_motor_open_pos <= 0:
        parser.error("--gripper-motor-open-pos must be positive")
    positive = (
        "record_fps",
        "control_hz",
        "command_smoothing_tau",
        "arm_feedback_timeout",
        "can_timeout",
        "tracking_error_deg",
        "tracking_error_duration",
        "vr_stale_timeout",
        "vr_loss_discard_time",
        "return_tolerance_deg",
        "return_velocity_tolerance",
        "return_settle_time",
        "scene_settle_time",
        "ik_orientation_weight",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.vr_stale_timeout >= args.vr_loss_discard_time:
        parser.error(
            "--vr-stale-timeout must be less than "
            "--vr-loss-discard-time"
        )
    return args


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    RealCollector(parse_args()).run()


if __name__ == "__main__":
    main()
