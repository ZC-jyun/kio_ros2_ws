#!/usr/bin/env python3
"""Tests for the real VR ACT collector safety and storage paths."""

import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import vr_collect_act_grasp_data_real as collector
from vr_real_runtime import (
    DryRunHardwareBridge,
    KeyboardSafetyMonitor,
    append_jsonl,
    next_contiguous_index,
    parse_keyboard_devices,
)


class FakeKeyboard:
    def __init__(self, *pressed):
        self.pressed = set(pressed)
        self.healthy = True
        self.error = None

    def is_pressed(self, key):
        return key.lower() in self.pressed

    def drain_events(self):
        return []

    def stop(self):
        pass


def load_act_utils():
    path = TOOLS.parent / "act-main" / "utils.py"
    spec = importlib.util.spec_from_file_location("act_utils_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndexTests(unittest.TestCase):
    def test_success_index_fills_gap_and_is_independent_from_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "episode_0.hdf5",
                "episode_2.hdf5",
                "failure_0.hdf5",
                "episode_bad.hdf5",
            ):
                (root / name).touch()
            self.assertEqual(next_contiguous_index(root, "episode"), 1)
            self.assertEqual(next_contiguous_index(root, "failure"), 1)
            (root / "episode_1.hdf5").touch()
            self.assertEqual(next_contiguous_index(root, "episode"), 3)


class KeyboardTests(unittest.TestCase):
    def test_keyboard_discovery_filters_system_buttons(self):
        proc_text = """
I: Bus=0019
N: Name="Power Button"
H: Handlers=kbd event0

I: Bus=0011
N: Name="AT Translated Set 2 keyboard"
H: Handlers=sysrq kbd event4 leds

I: Bus=0003
N: Name="USB Mouse"
H: Handlers=mouse0 event7
"""
        self.assertEqual(
            parse_keyboard_devices(proc_text),
            [
                (
                    "AT Translated Set 2 keyboard",
                    Path("/dev/input/event4"),
                )
            ],
        )

    def test_key_down_up_and_repeat_are_distinct(self):
        monitor = KeyboardSafetyMonitor("/dev/null")
        monitor.process_event(1, 23, 1, event_time=1.0)
        monitor.process_event(1, 23, 2, event_time=1.1)
        monitor.process_event(1, 23, 1, event_time=1.2)
        self.assertTrue(monitor.is_pressed("i"))
        monitor.process_event(1, 23, 0, event_time=2.0)
        self.assertFalse(monitor.is_pressed("i"))
        self.assertEqual(
            monitor.drain_events(),
            [("down", "i", 1.0), ("up", "i", 2.0)],
        )

    def test_estop_key_is_global_event(self):
        monitor = KeyboardSafetyMonitor("/dev/null")
        monitor.process_event(1, 18, 1, event_time=3.0)
        self.assertTrue(monitor.is_pressed("e"))
        self.assertEqual(monitor.drain_events(), [("down", "e", 3.0)])

    def test_manual_gripper_keys_are_global_events(self):
        monitor = KeyboardSafetyMonitor("/dev/null")
        monitor.process_event(1, 22, 1)
        monitor.process_event(1, 32, 1)
        self.assertTrue(monitor.is_pressed("u"))
        self.assertTrue(monitor.is_pressed("d"))

    def test_start_closes_fd_when_initial_key_query_fails(self):
        monitor = KeyboardSafetyMonitor()
        with (
            mock.patch.object(
                monitor,
                "resolve_device",
                return_value=Path("/dev/input/event-test"),
            ),
            mock.patch("vr_real_runtime.os.open", return_value=37),
            mock.patch(
                "vr_real_runtime.fcntl.ioctl",
                side_effect=OSError("not an input device"),
            ),
            mock.patch("vr_real_runtime.os.close") as close,
        ):
            with self.assertRaisesRegex(OSError, "not an input device"):
                monitor.start()
        close.assert_called_once_with(37)
        self.assertIsNone(monitor._fd)


class DryRunBridgeTests(unittest.TestCase):
    def test_bridge_tracks_target_and_restarts_after_estop(self):
        bridge = DryRunHardwareBridge(
            control_frequency=500.0,
            smoothing_tau=0.01,
            max_speed=2.0,
        )
        try:
            bridge.start()
            bridge.set_target(
                np.full(7, 0.1),
                feedforward_torque=np.full(7, 0.2),
            )
            time.sleep(0.08)
            sent = bridge.get_sent_target()
            sent_torque = bridge.get_sent_feedforward_torque()
            timing = bridge.get_control_timing()
            state = bridge.get_state()[0]
            self.assertTrue(np.all(sent > 0.0))
            self.assertTrue(np.all(sent <= 0.1))
            self.assertTrue(np.all(state > 0.0))
            self.assertTrue(np.all(sent_torque > 0.0))
            self.assertTrue(np.all(sent_torque <= 0.2))
            self.assertGreater(timing["frequency_hz"], 0.0)
            self.assertLess(time.monotonic() - bridge.get_feedback_timestamp(), 0.05)

            bridge.emergency_stop()
            bridge.stop()
            self.assertFalse(bridge._running.is_set())
            bridge.start()
            self.assertTrue(bridge._running.is_set())
        finally:
            bridge.stop()

    def test_hardware_rate_limit_uses_elapsed_time_before_smoothing(self):
        last = np.zeros(7)
        target = np.ones(7)
        max_speed = np.full(7, 0.2)

        result = collector.HardwareMotorBridge._rate_limited_position(
            last,
            target,
            max_speed,
            dt=0.01,
            target_weight=0.3,
        )

        np.testing.assert_allclose(result, np.full(7, 0.002))

    def test_hardware_rate_limit_is_frequency_independent(self):
        target = np.ones(7)
        max_speed = np.full(7, 0.2)

        fast = np.zeros(7)
        for _ in range(100):
            fast = collector.HardwareMotorBridge._rate_limited_position(
                fast, target, max_speed, 0.01, 1.0
            )
        slow = np.zeros(7)
        for _ in range(20):
            slow = collector.HardwareMotorBridge._rate_limited_position(
                slow, target, max_speed, 0.05, 1.0
            )

        np.testing.assert_allclose(fast, np.full(7, 0.2))
        np.testing.assert_allclose(slow, fast)


class StereoCameraTests(unittest.TestCase):
    def test_side_by_side_bgr_frame_is_split_and_converted_to_rgb(self):
        camera = collector.StereoRGBCamera(
            "/dev/null",
            raw_width=6,
            raw_height=2,
            output_width=3,
            output_height=2,
            fps=30,
        )
        frame = np.empty((2, 6, 3), dtype=np.uint8)
        frame[:, :3] = [1, 2, 3]
        frame[:, 3:] = [4, 5, 6]

        images = camera._decode_frame(frame)

        self.assertEqual(set(images), {"stereo_left", "stereo_right"})
        self.assertTrue(np.all(images["stereo_left"] == [3, 2, 1]))
        self.assertTrue(np.all(images["stereo_right"] == [6, 5, 4]))

    def test_rgb_letterbox_preserves_aspect_ratio(self):
        source = np.full((2, 4, 3), [7, 8, 9], dtype=np.uint8)
        output = np.full((4, 4, 3), 255, dtype=np.uint8)

        collector._letterbox_rgb(source, output)

        self.assertTrue(np.all(output[0] == 0))
        self.assertTrue(np.all(output[3] == 0))
        self.assertTrue(np.all(output[1:3] == [7, 8, 9]))

    def test_real_stereo_is_the_default_headset_view(self):
        instance = object.__new__(collector.RealCollector)
        instance.a = SimpleNamespace(
            vr_view="real",
            camera_timeout=0.5,
            swap_vr_camera_eyes=False,
            show_wrist=False,
            teleop_debug=False,
        )
        left = np.full((2, 4, 3), [10, 20, 30], dtype=np.uint8)
        right = np.full((2, 4, 3), [40, 50, 60], dtype=np.uint8)
        instance.stereo_camera = mock.Mock()
        instance.stereo_camera.latest.return_value = (
            {"stereo_left": left, "stereo_right": right},
            time.monotonic(),
        )
        instance.img = np.zeros((480, 1280, 3), dtype=np.uint8)
        instance.hardware_armed = False
        instance.mode = instance.DISARMED

        instance.render(np.eye(4))

        self.assertTrue(np.all(instance.img[240, 320] == [10, 20, 30]))
        self.assertTrue(np.all(instance.img[240, 960] == [40, 50, 60]))
        self.assertTrue(np.all(instance.img[0] == 0))

    def test_invalid_stereo_frame_shape_is_rejected(self):
        camera = collector.StereoRGBCamera(
            "/dev/null",
            raw_width=6,
            raw_height=2,
            output_width=3,
            output_height=2,
            fps=30,
        )
        with self.assertRaisesRegex(RuntimeError, "Expected stereo frame"):
            camera._decode_frame(np.zeros((2, 3, 3), dtype=np.uint8))


def make_buffers(length=2):
    images = {
        "stereo_left": [
            np.full((4, 6, 3), index, dtype=np.uint8)
            for index in range(length)
        ],
        "stereo_right": [
            np.full((4, 6, 3), index + 10, dtype=np.uint8)
            for index in range(length)
        ],
    }
    return {
        "action": [np.full(14, index, dtype=np.float32) for index in range(length)],
        "sent_action": [
            np.full(14, index + 0.25, dtype=np.float32)
            for index in range(length)
        ],
        "qpos": [np.full(14, index + 1, dtype=np.float32) for index in range(length)],
        "qvel": [np.full(14, 0.1 * index, dtype=np.float32) for index in range(length)],
        "images": images,
        "camera_names": ["stereo_left", "stereo_right"],
        "image_source": "spca2100_v4l2_stereo_rgb",
        "camera_device": "/dev/v4l/by-id/test-stereo-index0",
        "camera_raw_resolution": [720, 2560],
        "camera_output_resolution": [360, 640],
        "future_qpos_delay": 0.5,
        "outcome_source": "operator_keyboard_y",
        "gravity_compensation_scale": 1.0,
        "initial_object_pose": np.arange(7, dtype=np.float32),
        "initial_motor_position": np.arange(7, dtype=np.float32),
        "calibration_q": np.arange(6, dtype=np.float32),
        "calibration_hand_pose": np.eye(4, dtype=np.float32),
        "calibration_tcp_pose": np.eye(4, dtype=np.float32),
        "started_at_utc": "2026-08-07T00:00:00+00:00",
        "finished_at_utc": "2026-08-07T00:00:01+00:00",
        "run_config": {"position_scale": 0.75},
        "summary": {
            "control_release_count": 1,
            "control_disabled_duration": 0.2,
            "vr_loss_count": 0,
            "vr_loss_duration": 0.0,
            "target_clamp_count": 0,
            "max_tracking_error_rad": 0.01,
        },
        "diagnostics": {
            "motor_command_q": [np.zeros(7) for _ in range(length)],
            "tracking_error": [np.zeros(6) for _ in range(length)],
            "control_enabled": [1 for _ in range(length)],
            "vr_tracking_valid": [1 for _ in range(length)],
            "target_clamped": [0 for _ in range(length)],
            "tcp_residual": [np.zeros(2) for _ in range(length)],
            "gravity_feedforward_raw": [
                np.zeros(6) for _ in range(length)
            ],
            "gravity_feedforward_command": [
                np.zeros(7) for _ in range(length)
            ],
            "gravity_feedforward_sent": [
                np.zeros(7) for _ in range(length)
            ],
            "motor_control_frequency_hz": [
                500.0 for _ in range(length)
            ],
            "motor_control_mean_dt": [
                0.002 for _ in range(length)
            ],
            "motor_control_max_dt": [
                0.003 for _ in range(length)
            ],
            "control_dt": [0.016 for _ in range(length)],
        },
        "timestamps": {
            name: [float(index) for index in range(length)]
            for name in (
                "record",
                "can",
                "action",
                "stereo_left",
                "stereo_right",
            )
        },
    }


class EpisodeStorageTests(unittest.TestCase):
    def test_atomic_episode_has_compatible_core_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = collector.save_real_episode(
                directory,
                "episode",
                0,
                make_buffers(),
                "success",
            )
            self.assertEqual(path.name, "episode_0.hdf5")
            with h5py.File(path, "r") as root:
                self.assertFalse(bool(root.attrs["sim"]))
                self.assertEqual(root.attrs["valid_length"], 2)
                self.assertEqual(root["/action"].shape, (750, 14))
                self.assertEqual(root["/requested_action"].shape, (750, 14))
                self.assertEqual(root["/sent_action"].shape, (750, 14))
                self.assertEqual(root["/future_qpos"].shape, (750, 14))
                self.assertEqual(root["/observations/qpos"].shape, (750, 14))
                self.assertEqual(root["/action"].id, root["/future_qpos"].id)
                np.testing.assert_allclose(root["/requested_action"][1], 1.0)
                np.testing.assert_allclose(root["/sent_action"][1], 1.25)
                np.testing.assert_allclose(root["/future_qpos"][0], 1.5)
                np.testing.assert_allclose(root["/future_qpos"][1], 2.0)
                self.assertEqual(
                    root.attrs["act_schema"],
                    "bimanual_joint_position_v1",
                )
                self.assertEqual(
                    list(root.attrs["active_arm_mask"]),
                    [1, 0],
                )
                self.assertEqual(root.attrs["controlled_arms"], "left")
                self.assertEqual(
                    list(root.attrs["joint_names"]),
                    list(collector.ACT_JOINT_NAMES),
                )
                self.assertEqual(
                    root.attrs["joint_order"],
                    "left_arm,left_gripper,right_arm,right_gripper",
                )
                self.assertEqual(
                    root.attrs["gripper_normalization"],
                    "closed=0.0,open=1.0",
                )
                self.assertEqual(
                    root.attrs["action_type"],
                    "absolute_joint_position",
                )
                self.assertEqual(root.attrs["action_source"], "future_qpos")
                self.assertFalse(bool(root.attrs["action_before_smoothing"]))
                self.assertTrue(
                    bool(root.attrs["requested_action_before_smoothing"])
                )
                self.assertEqual(root.attrs["future_qpos_delay_sec"], 0.5)
                self.assertEqual(
                    root.attrs["outcome_source"],
                    "operator_keyboard_y",
                )
                self.assertEqual(
                    root.attrs["image_source"],
                    "spca2100_v4l2_stereo_rgb",
                )
                self.assertEqual(
                    list(root.attrs["camera_names"]),
                    ["stereo_left", "stereo_right"],
                )
                self.assertEqual(
                    root.attrs["camera_layout"],
                    "side_by_side_left_right",
                )
                self.assertEqual(
                    list(root.attrs["camera_raw_resolution"]),
                    [720, 2560],
                )
                self.assertEqual(
                    list(root.attrs["camera_output_resolution"]),
                    [360, 640],
                )
                self.assertEqual(
                    root.attrs["gravity_compensation_source"],
                    "mujoco_static_bias",
                )
                self.assertEqual(
                    root.attrs["gravity_compensation_scale"],
                    1.0,
                )
                self.assertEqual(
                    root["/observations/images/stereo_left"].shape,
                    (2, 4, 6, 3),
                )
                self.assertEqual(
                    root["/diagnostics/motor_command_q"].shape,
                    (2, 7),
                )
                self.assertEqual(
                    root["/diagnostics/gravity_feedforward_sent"].shape,
                    (2, 7),
                )
                self.assertEqual(
                    root["/diagnostics/motor_control_frequency_hz"].shape,
                    (2,),
                )
                self.assertEqual(root["/timestamps/can"].shape, (2,))
                self.assertEqual(
                    json.loads(root.attrs["run_config_json"])["position_scale"],
                    0.75,
                )

            with self.assertRaises(FileExistsError):
                collector.save_real_episode(
                    directory,
                    "episode",
                    0,
                    make_buffers(),
                    "success",
                )

    def test_failed_write_leaves_no_final_or_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                collector,
                "_write_episode_file",
                side_effect=RuntimeError("injected write failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    collector.save_real_episode(
                        directory,
                        "failure",
                        0,
                        make_buffers(),
                        "timeout",
                    )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_episode_is_readable_by_real_act_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            collector.save_real_episode(
                directory,
                "episode",
                0,
                make_buffers(),
                "success",
            )
            act_utils = load_act_utils()
            norm_stats = {
                "action_mean": np.zeros(14, dtype=np.float32),
                "action_std": np.ones(14, dtype=np.float32),
                "qpos_mean": np.zeros(14, dtype=np.float32),
                "qpos_std": np.ones(14, dtype=np.float32),
            }
            dataset = act_utils.EpisodicDataset(
                [0],
                directory,
                ["stereo_left", "stereo_right"],
                norm_stats,
            )
            images, qpos, action, is_pad = dataset[0]
            self.assertEqual(tuple(images.shape), (2, 3, 4, 6))
            self.assertEqual(tuple(qpos.shape), (14,))
            self.assertEqual(tuple(action.shape), (750, 14))
            self.assertEqual(tuple(is_pad.shape), (750,))
            self.assertFalse(bool(dataset.is_sim))


class LoggingAndArgsTests(unittest.TestCase):
    def test_safety_jsonl_is_immediately_parseable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safety_events.jsonl"
            append_jsonl(path, {"event": "test", "value": np.int64(3)})
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["event"], "test")
            self.assertEqual(record["value"], "3")

    def test_safety_jsonl_retries_partial_writes(self):
        writes = []

        def partial_write(_fd, data):
            count = min(3, len(data))
            writes.append(bytes(data[:count]))
            return count

        with (
            mock.patch("vr_real_runtime.os.open", return_value=41),
            mock.patch(
                "vr_real_runtime.os.write",
                side_effect=partial_write,
            ),
            mock.patch("vr_real_runtime.os.fsync"),
            mock.patch("vr_real_runtime.os.close"),
        ):
            append_jsonl("/tmp/unused.jsonl", {"event": "partial"})
        self.assertEqual(
            b"".join(writes),
            b'{"event": "partial"}\n',
        )

    def test_dry_run_defaults_and_real_acknowledgement(self):
        args = collector.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.position_scale, 1.0)
        self.assertEqual(args.command_smoothing_tau, 0.02)
        self.assertEqual(args.command_lookahead, 0.08)
        self.assertEqual(args.ik_orientation_weight, 0.7)
        self.assertEqual(args.ik_posture_gain, 0.5)
        self.assertEqual(args.max_command_lead_deg, 6.0)
        self.assertEqual(
            tuple(args.motor_max_speed),
            collector.DEFAULT_MOTOR_MAX_SPEED,
        )
        self.assertEqual(args.arm_feedback_timeout, 2.0)
        self.assertEqual(args.can_timeout, collector.umc.CAN_TIMEOUT_SEC)
        self.assertEqual(args.vr_stale_timeout, 0.25)
        self.assertEqual(args.vr_position_filter_hz, 10.0)
        self.assertEqual(args.vr_rotation_filter_hz, 8.0)
        self.assertEqual(args.vr_position_deadband, 0.002)
        self.assertEqual(args.vr_rotation_deadband_deg, 0.5)
        self.assertEqual(
            args.camera_device,
            collector.DEFAULT_STEREO_CAMERA_DEVICE,
        )
        self.assertEqual(args.camera_raw_width, 2560)
        self.assertEqual(args.camera_raw_height, 720)
        self.assertEqual(args.camera_output_width, 640)
        self.assertEqual(args.camera_output_height, 360)
        self.assertEqual(args.vr_view, "real")
        self.assertFalse(args.swap_vr_camera_eyes)
        self.assertEqual(args.record_fps, 30)
        self.assertAlmostEqual(args.future_qpos_delay, 1.0 / 30.0)
        self.assertFalse(args.show_wrist)
        self.assertEqual(args.motor_smoothing, 1.0)
        self.assertEqual(args.gravity_compensation_scale, 1.0)
        self.assertEqual(
            args.ctrl_c_return_speed,
            collector.DEFAULT_CTRL_C_RETURN_SPEED,
        )
        high_kp = collector.parse_args(
            [
                "--dry-run",
                "--kp",
                "240", "10", "10", "10", "10", "10", "0.5",
            ]
        )
        self.assertEqual(high_kp.kp[0], 240.0)
        with self.assertRaises(SystemExit):
            collector.parse_args(
                ["--dry-run", "--kp"] + ["240.1"] * 7
            )
        high_kd = collector.parse_args(
            ["--dry-run", "--kd", "5", "5", "1.5", ".3", ".3", ".3", ".8"]
        )
        self.assertEqual(high_kd.kd[0], 5.0)
        with self.assertRaises(SystemExit):
            collector.parse_args(
                ["--dry-run", "--kd"] + ["5.1"] * 7
            )
        with self.assertRaises(SystemExit):
            collector.parse_args([])
        with self.assertRaises(SystemExit):
            collector.parse_args(["--dry-run", "--motor-enable"])
        with self.assertRaises(SystemExit):
            collector.parse_args(["--dry-run", "--record-fps", "0"])


def make_state_collector(*pressed):
    instance = object.__new__(collector.RealCollector)
    instance.a = SimpleNamespace(
        can_timeout=0.05,
        tracking_error_deg=10.0,
        tracking_error_duration=0.3,
        vr_stale_timeout=0.1,
        vr_loss_discard_time=0.5,
        vr_position_filter_hz=10.0,
        vr_rotation_filter_hz=8.0,
        vr_position_deadband=0.002,
        vr_rotation_deadband_deg=0.5,
        return_tolerance_deg=2.0,
        return_velocity_tolerance=0.05,
        return_settle_time=0.3,
        ctrl_c_return_speed=0.10,
        gravity_compensation_scale=1.0,
        disable_gripper=False,
        gripper_max_speed=0.12,
        gripper_open_distance=0.10,
        gripper_motor_open_pos=5.0,
    )
    instance.keyboard = FakeKeyboard(*pressed)
    instance.hardware_bridge = mock.Mock()
    instance.hardware_bridge._running = threading.Event()
    instance.hardware_bridge._running.set()
    instance.hardware_bridge.get_sent_target.return_value = np.zeros(7)
    instance.hardware_armed = True
    instance.hardware_estopped = False
    instance.mode = instance.READY_MODE
    instance.state = instance.READY
    instance.cal = True
    instance.calibrate_at = None
    instance.calibration_q = np.zeros(6)
    instance.home_q = np.zeros(6)
    instance.home_g = 0.0
    instance.q = np.zeros(6)
    instance.command_q = np.zeros(6)
    instance.latest_motor_q = np.zeros(7)
    instance.latest_motor_dq = np.zeros(7)
    instance.latest_can_timestamp = 100.0
    instance.require_enable_release = False
    instance.motion_active = False
    instance.needs_rebase = False
    instance.tracking_valid = True
    instance.tracking_loss_since = None
    instance.tracking_error_since = None
    instance.home_stable_since = None
    instance.return_stable_since = None
    instance.auto_home_return = False
    instance.episode_stats = None
    instance.latest_left_hand = None
    instance.hand_filter_input = None
    instance.hand_filter_output = None
    instance.latest_gravity_feedforward_command = np.zeros(7)
    instance.latest_gravity_feedforward_raw = np.zeros(6)
    instance._gravity_feedforward = mock.Mock(return_value=np.zeros(7))
    instance.latest_control_dt = 0.0
    instance.g = 0.0
    return instance


class StateMachineTests(unittest.TestCase):
    def test_vr_pose_filter_rejects_noise_and_smooths_real_motion(self):
        instance = make_state_collector()
        reference = np.eye(4)
        instance._reset_hand_filter(reference)

        noise = reference.copy()
        noise[:3, 3] = [0.001, 0.0, 0.0]
        noise[:3, :3] = collector.sim_collector.rotation_from_rotvec(
            np.deg2rad([0.0, 0.0, 0.25])
        )
        filtered_noise = instance._filter_hand_pose(noise, 1.0 / 60.0)
        np.testing.assert_allclose(filtered_noise, reference, atol=1e-12)

        motion = reference.copy()
        motion[:3, 3] = [0.01, 0.0, 0.0]
        motion[:3, :3] = collector.sim_collector.rotation_from_rotvec(
            np.deg2rad([0.0, 0.0, 5.0])
        )
        filtered_motion = instance._filter_hand_pose(motion, 1.0 / 60.0)
        self.assertGreater(filtered_motion[0, 3], 0.0)
        self.assertLess(filtered_motion[0, 3], motion[0, 3])
        filtered_angle = np.linalg.norm(
            collector.sim_collector.rotvec(filtered_motion[:3, :3])
        )
        self.assertGreater(filtered_angle, 0.0)
        self.assertLess(filtered_angle, np.deg2rad(5.0))

        instance._reset_hand_filter()
        self.assertIsNone(instance.hand_filter_input)
        self.assertIsNone(instance.hand_filter_output)

    def test_weighted_dls_uses_soft_posture_without_breaking_bounds(self):
        preferred = np.array([0.5, -0.5])
        velocity = collector.sim_collector.bounded_weighted_dls(
            np.zeros((1, 2)),
            np.zeros(1),
            np.eye(2),
            0.1,
            np.array([-0.2, -0.2]),
            np.array([0.2, 0.2]),
            preferred,
        )

        np.testing.assert_allclose(velocity, [0.2, -0.2])

    def test_gripper_feedback_does_not_overwrite_home_command(self):
        instance = make_state_collector()
        instance.g = 0.0
        instance.model_left_limits = np.tile([[-2.0, 2.0]], (6, 1))
        instance.lq = np.arange(6)
        instance.lv = np.arange(6)
        instance.lf = [6, 7]
        instance.lfv = 6
        instance.d = SimpleNamespace(qpos=np.zeros(8), qvel=np.zeros(7))
        instance.m = object()

        with mock.patch.object(collector.mujoco, "mj_forward"):
            instance._mirror_model_from_hardware(
                np.r_[np.zeros(6), 5.0],
                np.zeros(7),
            )

        self.assertEqual(instance.g, 0.0)
        np.testing.assert_allclose(
            instance.d.qpos[instance.lf],
            collector.sim_collector.GRIPPER_OPEN,
        )

    def test_j04_home_gravity_feedforward_is_not_clipped(self):
        instance = object.__new__(collector.RealCollector)
        instance.a = SimpleNamespace(gravity_compensation_scale=1.0)
        instance.m = object()
        instance.d = SimpleNamespace(qpos=np.zeros(1))
        instance.gravity_data = SimpleNamespace(
            qpos=np.zeros(1),
            qvel=np.zeros(1),
            qacc=np.zeros(1),
            qfrc_bias=np.array([0.0, 0.0, 0.0, 2.63, 0.0, 0.0]),
            qfrc_passive=np.zeros(6),
        )
        instance.lv = np.arange(6)
        instance.latest_gravity_feedforward_raw = np.zeros(6)
        instance.latest_gravity_feedforward_command = np.zeros(7)

        with mock.patch.object(collector.mujoco, "mj_forward"):
            command = instance._gravity_feedforward()

        self.assertAlmostEqual(command[3], 2.63)

    def test_i_release_freezes_and_next_press_rebases(self):
        instance = make_state_collector()
        instance.latest_motor_q = np.linspace(0.0, 0.6, 7)
        instance.last_q_velocity = np.ones(6)
        instance.motion_active = True
        feedforward = np.arange(7, dtype=float)
        instance._gravity_feedforward.return_value = feedforward
        instance._control_tick(np.eye(4), 100.0, 0.01)
        self.assertFalse(instance.motion_active)
        self.assertTrue(instance.needs_rebase)
        instance.hardware_bridge.set_target.assert_called_once()
        hold = instance.hardware_bridge.set_target.call_args.args[0]
        np.testing.assert_allclose(hold, instance.latest_motor_q)
        np.testing.assert_allclose(
            instance.hardware_bridge.set_target.call_args.kwargs[
                "feedforward_torque"
            ],
            feedforward,
        )
        np.testing.assert_allclose(instance.last_q_velocity, np.zeros(6))

        instance.keyboard.pressed.add("i")
        instance.teleop = mock.Mock()
        instance._send_smoothed_target = mock.Mock()

        def rebase(_left_hand):
            instance.needs_rebase = False
            return True

        instance._rebase_control = mock.Mock(side_effect=rebase)
        instance._control_tick(np.eye(4), 100.1, 0.01)
        instance._rebase_control.assert_called_once()
        instance.teleop.assert_called_once()
        instance._send_smoothed_target.assert_called_once()

    def test_command_lead_is_bounded_and_reverses_without_backlog(self):
        instance = make_state_collector("i")
        instance.a.command_smoothing_tau = 0.02
        instance.a.max_command_lead_deg = 6.0
        instance.a.joint_limit_margin = 0.0
        instance.left_limits = np.tile(
            np.array([[-2.0, 2.0]]),
            (6, 1),
        )
        instance.command_q = np.zeros(6)
        instance.latest_motor_q = np.zeros(7)
        instance._motor_gripper_target = mock.Mock(return_value=0.0)
        instance._gravity_feedforward = mock.Mock(return_value=np.zeros(7))

        instance._send_smoothed_target(np.ones(6), 0.02)
        lead = np.deg2rad(instance.a.max_command_lead_deg)
        positive_target = instance.command_q.copy()
        self.assertTrue(np.all(positive_target <= lead))

        instance._send_smoothed_target(-np.ones(6), 0.02)
        reversed_target = instance.command_q.copy()
        self.assertTrue(np.all(reversed_target < 0.0))
        self.assertTrue(np.all(np.abs(reversed_target) <= lead))

    def test_calibration_return_requires_i_and_r_together(self):
        instance = make_state_collector("r")
        instance.mode = instance.RETURN_REQUIRED
        instance.motion_active = True
        instance._send_smoothed_target = mock.Mock()
        instance._update_return_completion = mock.Mock()
        instance._control_tick(np.eye(4), 100.0, 0.01)
        instance._send_smoothed_target.assert_not_called()
        instance._update_return_completion.assert_not_called()

        instance.keyboard.pressed.add("i")
        instance._control_tick(np.eye(4), 100.1, 0.01)
        instance._send_smoothed_target.assert_called_once()
        instance._update_return_completion.assert_called_once_with(100.1)

    def test_y_or_n_return_moves_to_home_without_i_or_r(self):
        instance = make_state_collector()
        instance.mode = instance.RETURN_REQUIRED
        instance.auto_home_return = True
        instance.home_q = np.ones(6)
        instance.calibration_q = np.zeros(6)
        instance._send_smoothed_target = mock.Mock()
        instance._update_return_completion = mock.Mock()

        instance._control_tick(np.eye(4), 100.0, 0.01)

        instance._send_smoothed_target.assert_called_once_with(
            instance.home_q, 0.01
        )
        instance._update_return_completion.assert_called_once_with(100.0)

    def test_y_or_n_home_return_clears_calibration_and_waits_for_p(self):
        instance = make_state_collector()
        instance.mode = instance.RETURN_REQUIRED
        instance.auto_home_return = True
        instance.return_stable_since = 100.0
        instance.latest_motor_dq[:] = 1.0
        instance.calibration_hand_pose = np.eye(4)
        instance.calibration_tcp_pose = np.eye(4)
        instance.targets_complete_after_return = False
        instance._freeze_motion_target = mock.Mock()
        instance._reset_virtual_scene = mock.Mock()

        instance._update_return_completion(100.31)

        self.assertEqual(instance.mode, instance.ARMED)
        self.assertFalse(instance.cal)
        self.assertIsNone(instance.calibration_q)
        self.assertIsNone(instance.calibration_hand_pose)
        self.assertIsNone(instance.calibration_tcp_pose)
        self.assertFalse(instance.require_enable_release)
        self.assertFalse(instance.needs_rebase)

    def test_y_success_selects_automatic_home_return(self):
        instance = make_state_collector()
        instance.mode = instance.RECORDING
        instance.buf = {}
        instance.ep = 0
        instance.output_dir = Path("/unused")
        instance._close_episode_stats = mock.Mock()
        instance._freeze_motion_target = mock.Mock()
        instance.advance_failure_target = mock.Mock(return_value=False)

        with (
            mock.patch.object(collector, "save_real_episode"),
            mock.patch.object(collector, "next_contiguous_index", return_value=1),
        ):
            instance._finish_success()

        self.assertEqual(instance.mode, instance.RETURN_REQUIRED)
        self.assertTrue(instance.auto_home_return)

    def test_arm_enters_yaml_home_required_state(self):
        instance = make_state_collector()
        instance.hardware_armed = False
        instance._prepare_bridge_rearm = mock.Mock()
        instance._wait_for_arm_feedback = mock.Mock()
        instance._synchronize_control_from_feedback = mock.Mock()

        instance.arm_hardware()

        self.assertEqual(instance.mode, instance.HOME_REQUIRED)
        self.assertTrue(instance.hardware_armed)
        self.assertEqual(instance.g, instance.home_g)
        instance.hardware_bridge.start.assert_called_once()

    def test_startup_home_moves_automatically_without_i_or_r(self):
        instance = make_state_collector()
        instance.mode = instance.HOME_REQUIRED
        instance._send_smoothed_target = mock.Mock()
        instance._update_home_completion = mock.Mock()

        instance._control_tick(np.eye(4), 100.0, 0.01)

        instance._send_smoothed_target.assert_called_once_with(instance.home_q, 0.01)
        instance._update_home_completion.assert_called_once_with(100.0)

    def test_startup_home_requires_stable_position_before_calibration(self):
        instance = make_state_collector()
        instance.mode = instance.HOME_REQUIRED
        instance.latest_motor_dq[:] = 1.0
        instance._freeze_motion_target = mock.Mock()
        instance._update_home_completion(100.0)
        self.assertEqual(instance.mode, instance.HOME_REQUIRED)
        self.assertEqual(instance.home_stable_since, 100.0)

        instance._update_home_completion(100.31)
        self.assertEqual(instance.mode, instance.ARMED)
        self.assertFalse(instance.require_enable_release)
        self.assertIsNone(instance.home_stable_since)
        instance._freeze_motion_target.assert_called_once()

    def test_u_opens_and_d_closes_gripper_before_calibration(self):
        instance = make_state_collector("u")
        instance.mode = instance.ARMED
        instance.g = 0.02
        instance.a.command_smoothing_tau = 0.02
        instance.a.max_command_lead_deg = 6.0
        instance.a.joint_limit_margin = 0.0
        instance.left_limits = np.tile(np.array([[-2.0, 2.0]]), (6, 1))
        instance._gravity_feedforward = mock.Mock(return_value=np.zeros(7))

        instance._control_tick(np.eye(4), 100.0, 0.1)
        target = instance.hardware_bridge.set_target.call_args.args[0]
        self.assertEqual(collector.umc.GRIPPER_CAN_ID, 0x01)
        self.assertEqual(target[collector.umc.ARM_DOF], 5.0)

        instance.keyboard.pressed = {"d"}
        instance._control_tick(np.eye(4), 100.1, 0.1)
        target = instance.hardware_bridge.set_target.call_args.args[0]
        self.assertEqual(target[collector.umc.ARM_DOF], 0.0)
        self.assertEqual(instance.hardware_bridge.set_target.call_count, 2)

    def test_armed_waiting_for_p_keeps_home_and_gravity_compensation(self):
        instance = make_state_collector()
        instance.mode = instance.ARMED
        instance.home_q = np.ones(6)
        instance._send_smoothed_target = mock.Mock()

        instance._control_tick(np.eye(4), 100.0, 0.01)

        instance._send_smoothed_target.assert_called_once_with(
            instance.home_q, 0.01
        )

    def test_vr_pinch_distance_sets_absolute_gripper_target(self):
        instance = make_state_collector()

        instance.update_gripper(0.01, 0.05)
        self.assertAlmostEqual(
            instance.g,
            collector.sim_collector.GRIPPER_OPEN / 2.0,
        )
        instance.update_gripper(0.01, 0.10)
        self.assertEqual(instance._motor_gripper_target(), 5.0)

    def test_held_e_prevents_rearm(self):
        instance = make_state_collector("e")
        instance.hardware_armed = False
        instance.arm_hardware()
        instance.hardware_bridge.start.assert_not_called()
        self.assertFalse(instance.hardware_armed)

    def test_arm_accepts_fresh_feedback_from_all_active_motors(self):
        instance = make_state_collector()
        instance.a.arm_feedback_timeout = 0.2
        calls = []

        def fresh_feedback(require_running=True):
            calls.append(require_running)
            instance.latest_can_timestamp = (
                time.monotonic() + len(calls) * 1e-6
            )
            return instance.latest_motor_q, instance.latest_motor_dq

        instance._read_hardware_feedback = mock.Mock(
            side_effect=fresh_feedback
        )
        instance._wait_for_arm_feedback()
        self.assertEqual(calls, [False])

    def test_failed_rearm_from_fault_still_emergency_stops_bridge(self):
        instance = make_state_collector()
        instance.hardware_armed = False
        instance.hardware_estopped = True
        instance.mode = instance.FAULT
        instance._prepare_bridge_rearm = mock.Mock()
        instance._wait_for_arm_feedback = mock.Mock(
            side_effect=TimeoutError("no fresh feedback")
        )
        instance._log_safety_event = mock.Mock()

        def stop_bridge():
            instance.hardware_bridge._running.clear()

        instance.hardware_bridge.emergency_stop.side_effect = stop_bridge
        instance.arm_hardware()

        instance.hardware_bridge.start.assert_called_once()
        instance.hardware_bridge.emergency_stop.assert_called_once()
        self.assertFalse(instance.hardware_bridge._running.is_set())
        self.assertFalse(instance.hardware_armed)
        self.assertTrue(instance.hardware_estopped)
        self.assertEqual(instance.mode, instance.FAULT)

    def test_can_timeout_and_tracking_error_enter_fault(self):
        timed_out = make_state_collector()
        timed_out.latest_can_timestamp = 99.0
        timed_out._safety_fault = mock.Mock()
        timed_out._safety_checks(100.0)
        timed_out._safety_fault.assert_called_once()
        self.assertEqual(
            timed_out._safety_fault.call_args.args[0],
            "can_timeout",
        )

        tracking = make_state_collector()
        tracking.hardware_bridge.get_sent_target.return_value = np.ones(7)
        tracking.tracking_error_since = 99.5
        tracking._safety_fault = mock.Mock()
        tracking._safety_checks(100.0)
        tracking._safety_fault.assert_called_once()
        self.assertEqual(
            tracking._safety_fault.call_args.args[0],
            "tracking_error",
        )

    def test_vr_loss_under_and_over_threshold_then_recovers(self):
        instance = make_state_collector()
        instance.mode = instance.RECORDING
        instance.motion_active = True
        instance.episode_stats = {
            "vr_loss_count": 0,
            "vr_loss_duration": 0.0,
        }
        instance.tv = SimpleNamespace(left_hand_update_time=99.8)
        instance._freeze_motion_target = mock.Mock()
        instance._discard_episode = mock.Mock()

        instance._update_tracking(np.eye(4), np.eye(4), 100.0)
        instance._freeze_motion_target.assert_called_once()
        instance._discard_episode.assert_not_called()
        instance._update_tracking(np.eye(4), np.eye(4), 100.4)
        instance._discard_episode.assert_called_once_with(
            "vr_tracking_loss",
            require_return=True,
        )

        instance.tv.left_hand_update_time = 100.41
        instance._update_tracking(np.eye(4), np.eye(4), 100.42)
        self.assertTrue(instance.tracking_valid)
        self.assertTrue(instance.needs_rebase)

    def test_failed_calibration_returns_to_armed(self):
        instance = make_state_collector()
        instance.mode = instance.CALIBRATING
        instance.cal = False
        instance.calibrate_at = None
        with mock.patch.object(
            collector.sim_collector.Collector,
            "finish_calibration",
        ):
            instance.finish_calibration(np.eye(4), np.eye(4))
        self.assertEqual(instance.mode, instance.ARMED)

    def test_target_clamps_are_counted_as_transitions(self):
        instance = make_state_collector()
        instance.episode_stats = {"target_clamp_count": 0}
        instance.target_clamp_active = False
        for clamped in (False, True, True, False, True):
            instance.target_clamped = clamped
            instance._record_target_clamp_sample()
        self.assertEqual(instance.episode_stats["target_clamp_count"], 2)

    def test_y_saves_operator_success_and_requires_return(self):
        instance = make_state_collector()
        instance.mode = instance.RECORDING
        instance._finish_success = mock.Mock()

        self.assertTrue(instance._handle_terminal_key("y"))

        instance._finish_success.assert_called_once_with()

    def test_n_discards_operator_failure_and_requires_return(self):
        instance = make_state_collector()
        instance.mode = instance.RECORDING
        instance._discard_episode = mock.Mock()

        self.assertTrue(instance._handle_terminal_key("n"))

        instance._discard_episode.assert_called_once_with(
            "operator_failure",
            require_return=True,
            auto_home=True,
        )


class CleanupTests(unittest.TestCase):
    def test_ctrl_c_return_reaches_zero_pose_before_disable(self):
        instance = make_state_collector()
        instance.home_q = np.full(6, 0.5)
        instance.latest_motor_q = np.r_[np.full(6, 0.01), 2.0]
        instance.latest_motor_dq = np.zeros(7)
        instance._read_hardware_feedback = mock.Mock()

        def follow_target(target, **_kwargs):
            instance.latest_motor_q = np.asarray(target).copy()
            instance.latest_motor_dq[:] = 0.0

        instance.hardware_bridge.set_target.side_effect = follow_target
        with mock.patch.object(collector.time, "sleep"):
            instance._return_home_before_shutdown()

        final_target = instance.hardware_bridge.set_target.call_args.args[0]
        np.testing.assert_allclose(final_target[:6], collector.ZERO_POSE)
        self.assertEqual(final_target[6], 2.0)
        self.assertGreater(instance.hardware_bridge.set_target.call_count, 1)

    def test_connect_failure_still_stops_keyboard_and_closes_resources(self):
        instance = object.__new__(collector.RealCollector)
        instance.a = SimpleNamespace(
            keyboard_device=Path("/dev/input/test"),
            camera_start_timeout=5.0,
        )
        instance.mode = instance.DISARMED
        instance.hardware_armed = False
        instance.hardware_bridge = mock.Mock()
        instance.stereo_camera = mock.Mock()
        instance.stereo_camera.device = Path("/dev/video-test")
        instance.stereo_camera.raw_width = 2560
        instance.stereo_camera.raw_height = 720
        instance.stereo_camera.output_width = 640
        instance.stereo_camera.output_height = 360
        instance._close_resources = mock.Mock()
        instance.connect = mock.Mock(side_effect=RuntimeError("connect failed"))
        keyboard = mock.Mock()
        keyboard.device_name = "test"
        keyboard.device = Path("/dev/input/test")
        with (
            mock.patch.object(
                collector,
                "KeyboardSafetyMonitor",
                return_value=keyboard,
            ),
            self.assertRaisesRegex(RuntimeError, "connect failed"),
        ):
            instance.run()
        keyboard.stop.assert_called_once()
        instance.hardware_bridge.stop.assert_called_once()
        instance._close_resources.assert_called_once()

    def test_window_cleanup_failure_does_not_leak_other_resources(self):
        instance = object.__new__(collector.RealCollector)
        instance.a = SimpleNamespace(show_wrist=True)
        instance.ren = {"vr_center": mock.Mock(), "wrist": mock.Mock()}
        instance.video_renderer = mock.Mock()
        instance.video_writer = None
        process = mock.Mock()
        process.is_alive.return_value = True
        instance.tv = SimpleNamespace(process=process)
        instance.rl = mock.Mock()
        instance.rr = mock.Mock()
        instance.gl = mock.Mock()
        instance.sh = mock.Mock()

        with tempfile.TemporaryDirectory() as directory:
            instance.tmp = str(Path(directory) / "scene")
            Path(instance.tmp).mkdir()
            with mock.patch.object(
                collector.cv2,
                "destroyWindow",
                side_effect=collector.cv2.error("window not created"),
            ):
                instance._close_resources()

        process.terminate.assert_called_once()
        process.join.assert_called_once_with(timeout=2.0)
        instance.rl.free.assert_called_once()
        instance.rr.free.assert_called_once()
        instance.gl.free.assert_called_once()
        instance.sh.close.assert_called_once()
        instance.sh.unlink.assert_called_once()


if __name__ == "__main__":
    unittest.main()
