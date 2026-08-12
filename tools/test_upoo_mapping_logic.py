#!/usr/bin/env python3
"""Offline tests for UPOO motor configuration; never opens CAN hardware."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE_SRC = WORKSPACE / "src" / "kio_teleop_openarm"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from kio_teleop_openarm.lib import upoo_motor_constants as umc


class MotorMappingTest(unittest.TestCase):
    def test_declared_types_and_directions(self):
        self.assertEqual(
            [umc.ARM_MOTOR_TYPES[name] for name, _, _ in umc.ARM_MOTOR_CONFIG],
            ["DM4340_48V"] * 3 + ["DM4310_48V"] * 3,
        )
        self.assertEqual(umc.arm_direction_vector(), [1.0, -1.0, 1.0, 1.0, -1.0, 1.0])

    def test_openarm_reference_gains_keep_gripper_independent(self):
        self.assertEqual(umc.DEFAULT_KP, [240, 240, 120, 40, 24, 31, 0.5])
        self.assertEqual(umc.DEFAULT_KD, [5, 5, 1.5, 0.3, 0.3, 0.3, 0.8])
        self.assertEqual(umc.MAX_RUNTIME_KP, 240.0)
        self.assertEqual(umc.MAX_RUNTIME_KD, 5.0)

    def test_j04_feedforward_covers_horizontal_home_gravity(self):
        self.assertEqual(umc.MAX_FEEDFORWARD_TORQUE[3], 3.0)

    def test_position_velocity_and_torque_round_trip(self):
        for joint, _, _ in umc.ARM_MOTOR_CONFIG:
            for value in (-1.2, -0.05, 0.0, 0.05, 1.2):
                motor_value = umc.mujoco_to_motor(joint, value)
                self.assertAlmostEqual(umc.motor_to_mujoco(joint, motor_value), value)

    def test_all_arm_startup_limits_include_zero_tolerance(self):
        tolerance = umc.ZERO_VERIFY_TOLERANCE_RAD
        for joint, _, _ in umc.ARM_MOTOR_CONFIG:
            with self.subTest(joint=joint):
                soft_lo, soft_hi = umc.SOFT_POSITION_LIMITS[joint]
                startup_lo, startup_hi = umc.startup_position_limits(joint)
                self.assertAlmostEqual(startup_lo, soft_lo - tolerance)
                self.assertAlmostEqual(startup_hi, soft_hi + tolerance)
                self.assertGreaterEqual(soft_lo, startup_lo)
                self.assertLessEqual(soft_hi, startup_hi)

    def test_all_arm_startup_limits_reject_beyond_zero_tolerance(self):
        epsilon = 1e-6
        for joint, _, _ in umc.ARM_MOTOR_CONFIG:
            with self.subTest(joint=joint):
                startup_lo, startup_hi = umc.startup_position_limits(joint)
                self.assertFalse(startup_lo <= startup_lo - epsilon <= startup_hi)
                self.assertFalse(startup_lo <= startup_hi + epsilon <= startup_hi)

    def complete_record(self):
        data = json.loads(umc.CALIBRATION_RECORD.read_text(encoding="utf-8"))
        for joint, can_id, mst_id in umc.ARM_MOTOR_CONFIG:
            record = data["motors"][joint]
            record.update({
                "can_id": can_id,
                "mst_id": mst_id,
                "direction_test": "OK" if umc.JOINT_DIRECTION[joint] > 0 else "REVERSED",
                "verify_passed": True,
                "last_verify_position_rad": 0.0,
                "mapped_control_test_passed": True,
                "mapped_control_test_kp": umc.DEFAULT_KP[
                    [name for name, _, _ in umc.ARM_MOTOR_CONFIG].index(joint)],
                "mapped_control_test_kd": umc.DEFAULT_KD[
                    [name for name, _, _ in umc.ARM_MOTOR_CONFIG].index(joint)],
                "mapped_control_test_assist_torque": 0.0,
            })
        return data

    def validate_temp_record(
        self,
        data,
        require_control_tests=True,
        **runtime_gains,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return umc.validate_calibration_record(
                path, require_control_tests, **runtime_gains
            )

    def test_complete_record_passes(self):
        data = self.complete_record()
        self.assertIs(self.validate_temp_record(data)["motors"]["J05"]["verify_passed"], True)

    def test_control_record_must_match_selected_runtime_kp(self):
        data = self.complete_record()
        data["motors"]["Base_J01"]["mapped_control_test_kp"] = 200.0
        with self.assertRaisesRegex(RuntimeError, "runtime Kp"):
            self.validate_temp_record(data)

        runtime_kp = list(umc.DEFAULT_KP)
        runtime_kp[0] = 200.0
        result = self.validate_temp_record(
            data,
            expected_kp=runtime_kp,
            expected_kd=umc.DEFAULT_KD,
        )
        self.assertTrue(result["motors"]["Base_J01"]["mapped_control_test_passed"])

    def test_wrong_reversed_joint_is_rejected(self):
        data = self.complete_record()
        data["motors"]["J05"]["direction_test"] = "OK"
        with self.assertRaisesRegex(RuntimeError, "J05: direction"):
            self.validate_temp_record(data)

    def test_missing_zero_or_control_test_is_rejected(self):
        data = self.complete_record()
        data["motors"]["J03"]["verify_passed"] = False
        data["motors"]["J06"]["mapped_control_test_passed"] = False
        with self.assertRaisesRegex(RuntimeError, "J03: zero verification"):
            self.validate_temp_record(data)


if __name__ == "__main__":
    unittest.main()
