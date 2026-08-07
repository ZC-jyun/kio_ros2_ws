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
            bridge.set_target(np.full(7, 0.1))
            time.sleep(0.08)
            sent = bridge.get_sent_target()
            state = bridge.get_state()[0]
            self.assertTrue(np.all(sent > 0.0))
            self.assertTrue(np.all(sent <= 0.1))
            self.assertTrue(np.all(state > 0.0))
            self.assertLess(time.monotonic() - bridge.get_feedback_timestamp(), 0.05)

            bridge.emergency_stop()
            bridge.stop()
            self.assertFalse(bridge._running.is_set())
            bridge.start()
            self.assertTrue(bridge._running.is_set())
        finally:
            bridge.stop()


def make_buffers(length=2):
    images = {
        "vr_center": [
            np.full((4, 6, 3), index, dtype=np.uint8)
            for index in range(length)
        ],
        "wrist": [
            np.full((4, 6, 3), index + 10, dtype=np.uint8)
            for index in range(length)
        ],
    }
    return {
        "action": [np.full(14, index, dtype=np.float32) for index in range(length)],
        "qpos": [np.full(14, index + 1, dtype=np.float32) for index in range(length)],
        "qvel": [np.full(14, 0.1 * index, dtype=np.float32) for index in range(length)],
        "images": images,
        "camera_names": ["vr_center", "wrist"],
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
        },
        "timestamps": {
            name: [float(index) for index in range(length)]
            for name in ("record", "can", "action", "vr_center", "wrist")
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
                self.assertEqual(root["/observations/qpos"].shape, (750, 14))
                self.assertEqual(
                    root["/observations/images/vr_center"].shape,
                    (2, 4, 6, 3),
                )
                self.assertEqual(
                    root["/diagnostics/motor_command_q"].shape,
                    (2, 7),
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
                ["vr_center", "wrist"],
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
        self.assertEqual(args.position_scale, 0.75)
        self.assertEqual(args.command_smoothing_tau, 0.06)
        self.assertEqual(args.arm_feedback_timeout, 2.0)
        self.assertEqual(args.can_timeout, collector.umc.CAN_TIMEOUT_SEC)
        self.assertEqual(args.vr_stale_timeout, 0.25)
        with self.assertRaises(SystemExit):
            collector.parse_args([])
        with self.assertRaises(SystemExit):
            collector.parse_args(["--dry-run", "--motor-enable"])


def make_state_collector(*pressed):
    instance = object.__new__(collector.RealCollector)
    instance.a = SimpleNamespace(
        can_timeout=0.05,
        tracking_error_deg=10.0,
        tracking_error_duration=0.3,
        vr_stale_timeout=0.1,
        vr_loss_discard_time=0.5,
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
    instance.return_stable_since = None
    instance.episode_stats = None
    instance.latest_left_hand = None
    return instance


class StateMachineTests(unittest.TestCase):
    def test_i_release_freezes_and_next_press_rebases(self):
        instance = make_state_collector()
        instance.motion_active = True
        instance._control_tick(np.eye(4), 100.0, 0.01)
        self.assertFalse(instance.motion_active)
        self.assertTrue(instance.needs_rebase)
        instance.hardware_bridge.set_target.assert_called_once()

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


class CleanupTests(unittest.TestCase):
    def test_connect_failure_still_stops_keyboard_and_closes_resources(self):
        instance = object.__new__(collector.RealCollector)
        instance.a = SimpleNamespace(keyboard_device=Path("/dev/input/test"))
        instance.mode = instance.DISARMED
        instance.hardware_armed = False
        instance.hardware_bridge = mock.Mock()
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
