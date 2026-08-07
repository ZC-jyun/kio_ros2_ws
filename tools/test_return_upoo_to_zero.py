#!/usr/bin/env python3
"""Offline tests for the UPOO return-to-zero trajectory and preflight."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE_SRC = WORKSPACE / "src" / "kio_teleop_openarm"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from kio_teleop_openarm.lib import return_upoo_to_zero as homing
from kio_teleop_openarm.lib import upoo_motor_constants as umc


class ReturnToZeroTest(unittest.TestCase):
    def complete_zero_record(self):
        return {
            "schema_version": 1,
            "motors": {
                joint: {
                    "joint": joint,
                    "can_id": can_id,
                    "mst_id": mst_id,
                    "verify_passed": True,
                    "last_verify_position_rad": 0.0,
                }
                for joint, can_id, mst_id in umc.ARM_MOTOR_CONFIG
            },
        }

    def validate_record(self, data):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "zero.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return homing.validate_zero_calibration(path)

    def test_complete_zero_record_passes_without_direction_tests(self):
        data = self.complete_zero_record()
        checked = self.validate_record(data)
        self.assertTrue(checked["motors"]["J06"]["verify_passed"])

    def test_invalid_zero_record_is_rejected(self):
        data = self.complete_zero_record()
        data["motors"]["J03"]["last_verify_position_rad"] = 0.11
        data["motors"]["J05"]["can_id"] = 99
        with self.assertRaisesRegex(RuntimeError, "J03: last verified zero"):
            self.validate_record(data)

    def test_minimum_jerk_endpoints_and_monotonicity(self):
        samples = [homing.minimum_jerk_fraction(index, 100.0) for index in range(101)]
        self.assertEqual(samples[0], 0.0)
        self.assertEqual(samples[-1], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(samples, samples[1:])))
        self.assertAlmostEqual(homing.minimum_jerk_fraction(50.0, 100.0), 0.5)

    def test_duration_bounds_peak_joint_speed(self):
        start = (1.2, -0.4, 0.0, 0.2, -0.8, 0.1)
        speed = 0.1
        duration = homing.trajectory_duration(start, speed)
        self.assertAlmostEqual(
            duration, homing.MINIMUM_JERK_MAX_SLOPE * 1.2 / speed
        )
        for distance in map(abs, start):
            peak_speed = homing.MINIMUM_JERK_MAX_SLOPE * distance / duration
            self.assertLessEqual(peak_speed, speed + 1e-12)

    def test_trajectory_reaches_exact_zero(self):
        start = (0.5, -1.0, 0.1, -0.2, 0.0, 0.7)
        self.assertEqual(homing.trajectory_target(start, 0.0), start)
        self.assertEqual(homing.trajectory_target(start, 1.0), (0.0,) * 6)
        midpoint = homing.trajectory_target(start, 0.5)
        for initial, target in zip(start, midpoint):
            self.assertTrue(math.isclose(target, initial * 0.5))


    def feedback_with_j02_position(self, position):
        values = [0.0] * 6
        values[1] = position
        return homing.ArmFeedback(
            position=tuple(values),
            velocity=(0.0,) * 6,
            torque=(0.0,) * 6,
            status=(0,) * 6,
            timestamps=(0.0,) * 6,
        )

    def test_j02_zero_quantization_is_allowed_at_soft_limit(self):
        homing.validate_start(self.feedback_with_j02_position(0.004), 0.1)

    def test_j02_position_beyond_zero_tolerance_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, r"J02: q=\+0\.1100"):
            homing.validate_start(self.feedback_with_j02_position(0.11), 0.1)


if __name__ == "__main__":
    unittest.main()
