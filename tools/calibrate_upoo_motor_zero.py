#!/usr/bin/env python3
"""Safely set and verify the persistent zero of one UPOO arm motor.

The program only registers and commands the selected CAN ID. Other motors may
remain connected to the same CAN bus, provided no other program controls them.
It never scans or commands the other registered arm motors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE_SRC = WORKSPACE / "src" / "kio_teleop_openarm"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from kio_teleop_openarm.lib import upoo_motor_constants as umc


DEFAULT_RECORD_FILE = WORKSPACE / "data" / "hardware_calibration" / "left_arm_zero_results.json"
ZERO_TOLERANCE_RAD = umc.ZERO_VERIFY_TOLERANCE_RAD
FEEDBACK_TIMEOUT_SEC = 0.5
RAMP_RATE_HZ = 50.0
MIN_DIRECTION_MOTION_RAD = 0.01


@dataclass(frozen=True)
class MotorSpec:
    joint: str
    can_id: int
    mst_id: int


def motor_specs() -> dict[str, MotorSpec]:
    return {
        joint: MotorSpec(joint=joint, can_id=can_id, mst_id=mst_id)
        for joint, can_id, mst_id in umc.ARM_MOTOR_CONFIG
    }


def parse_args() -> argparse.Namespace:
    specs = motor_specs()
    parser = argparse.ArgumentParser(
        description="Set or verify the persistent zero of one connected UPOO arm motor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--joint", required=True, choices=sorted(specs))
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify a previously written zero without sending zero or position commands.",
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--kp", type=float, default=0.5, help="Optional direction-test position gain.")
    parser.add_argument("--kd", type=float, default=0.5, help="Optional direction-test velocity gain.")
    parser.add_argument("--test-angle", type=float, default=0.05)
    parser.add_argument("--test-speed", type=float, default=0.01, help="Direction-test ramp speed in rad/s.")
    parser.add_argument("--record-file", type=Path, default=DEFAULT_RECORD_FILE)
    args = parser.parse_args()
    if args.kp < 0 or args.kd < 0:
        parser.error("--kp and --kd must be non-negative")
    if not 0 < args.test_angle <= 0.05:
        parser.error("--test-angle must be in (0, 0.05] rad")
    if not 0 < args.test_speed <= 0.02:
        parser.error("--test-speed must be in (0, 0.02] rad/s")
    if args.channel < 0:
        parser.error("--channel must be non-negative")
    return args


def direction_test_target(joint: str, magnitude: float) -> float:
    lo, hi = umc.SOFT_POSITION_LIMITS[joint]
    if lo <= magnitude <= hi:
        return magnitude
    if lo <= -magnitude <= hi:
        return -magnitude
    raise ValueError(
        f"Neither +/-{magnitude:.3f} rad is inside {joint} soft limits [{lo}, {hi}]"
    )


def confirmation(expected: str, prompt: str) -> bool:
    print(prompt)
    return input(f"Type exactly '{expected}' to continue: ").strip() == expected


class SingleMotorSession:
    """A direct CAN session which knows about exactly one configured motor."""

    def __init__(self, spec: MotorSpec, channel: int):
        # Import only for a real hardware operation so --help stays hardware-free.
        from dmcan import dmcan_device_type
        from kio_teleop_openarm.lib.damiao import (
            Control_Mode,
            DM_Motor_Type,
            DmActData,
            Motor_Control,
        )

        self._control = Motor_Control(
            umc.NOM_BAUD,
            umc.DAT_BAUD,
            sn=umc.USB2CANFD_SN,
            data_ptr=[
                DmActData(
                    motorType=getattr(DM_Motor_Type, umc.ARM_MOTOR_TYPES[spec.joint]),
                    mode=Control_Mode.MIT_MODE,
                    can_id=spec.can_id,
                    mst_id=spec.mst_id,
                    channel=channel,
                )
            ],
            device_type=dmcan_device_type.USB2CANFD,
            auto_enable=False,
        )
        self._motor = self._control.getMotor(channel, spec.can_id)
        if self._motor is None:
            raise RuntimeError(f"Selected motor {spec.joint} was not registered")

    def read_feedback(self) -> tuple[float, float, float, int]:
        before = self._motor.last_time_
        self._control.refresh_motor_status(self._motor)
        deadline = time.monotonic() + FEEDBACK_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if self._motor.last_time_ > before:
                return (
                    float(self._motor.Get_Position()),
                    float(self._motor.Get_Velocity()),
                    float(self._motor.Get_tau()),
                    int(self._motor.Get_err()),
                )
            time.sleep(0.01)
        raise TimeoutError("No fresh CAN feedback from the selected motor")

    def set_zero(self) -> None:
        self._control.set_zero_position(self._motor)

    def enable_for_test(self) -> None:
        self._control.enable_all()

    def command(self, kp: float, kd: float, position: float) -> None:
        self._control.control_mit(self._motor, kp, kd, position, 0.0, 0.0)

    def ramp(self, kp: float, kd: float, start: float, target: float, speed: float) -> None:
        duration = abs(target - start) / speed
        steps = max(1, int(duration * RAMP_RATE_HZ))
        for step in range(steps + 1):
            position = start + (target - start) * step / steps
            self.command(kp, kd, position)
            time.sleep(1.0 / RAMP_RATE_HZ)

    def disable(self) -> None:
        self._control.disable_all()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_records(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "motors": {}}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("motors"), dict):
        raise ValueError(f"Invalid calibration record file: {path}")
    return data


def update_record(path: Path, spec: MotorSpec, values: dict[str, Any]) -> None:
    records = read_records(path)
    record = records["motors"].setdefault(spec.joint, asdict(spec))
    record.update(values)
    records["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, sort_keys=True)
        file.write("\n")


def print_feedback(label: str, sample: tuple[float, float, float, int]) -> None:
    position, velocity, torque, status = sample
    print(
        f"[{label}] q={position:+.4f} rad, dq={velocity:+.4f} rad/s, "
        f"tau={torque:+.4f}, status={status} ({umc.motor_status_label(status)})"
    )


def run_direction_test(session: SingleMotorSession, args: argparse.Namespace) -> str | None:
    test_target = direction_test_target(args.joint, args.test_angle)
    expected_direction = "positive" if test_target > 0 else "negative"
    if not confirmation(
        "TEST",
        "The selected motor will be enabled at low gain and move 0 -> "
        f"{test_target:+.3f} rad -> 0. Ensure the physical emergency stop is reachable.",
    ):
        print("[test] Skipped.")
        return None

    session.enable_for_test()
    try:
        session.command(args.kp, args.kd, 0.0)
        time.sleep(0.2)
        session.ramp(args.kp, args.kd, 0.0, test_target, args.test_speed)
        peak = session.read_feedback()
        print_feedback("test peak", peak)
        if umc.is_motor_fault(peak[3]):
            raise RuntimeError(
                f"Motor fault 0x{peak[3]:X} ({umc.motor_status_label(peak[3])})"
            )
        session.ramp(args.kp, args.kd, test_target, 0.0, args.test_speed)
        print_feedback("test return", session.read_feedback())
    finally:
        session.disable()
        print("[safety] Selected motor disabled after direction test.")

    if abs(peak[0]) < MIN_DIRECTION_MOTION_RAD:
        print(
            f"[test] Insufficient encoder motion ({peak[0]:+.4f} rad); "
            "recording direction as UNKNOWN."
        )
        return "UNKNOWN"

    while True:
        result = input(
            f"Did {test_target:+.3f} rad move in the MuJoCo {expected_direction} direction? "
            "Type OK, REVERSED, or UNKNOWN: "
        ).strip().upper()
        if result in {"OK", "REVERSED", "UNKNOWN"}:
            return result
        print("Please type OK, REVERSED, or UNKNOWN.")


def main() -> int:
    args = parse_args()
    spec = motor_specs()[args.joint]
    session: SingleMotorSession | None = None
    try:
        print(
            f"[safety] Selected only {spec.joint}: CAN ID 0x{spec.can_id:02X}, "
            f"master ID 0x{spec.mst_id:02X}, channel {args.channel}."
        )
        print("[safety] Stop every other CAN-control program before continuing.")
        session = SingleMotorSession(spec, args.channel)
        before = session.read_feedback()
        print_feedback("pre-operation", before)
        if umc.is_motor_fault(before[3]):
            raise RuntimeError(
                f"Selected motor reports fault 0x{before[3]:X} "
                f"({umc.motor_status_label(before[3])}); refusing to continue"
            )

        if args.verify:
            passed = abs(before[0]) <= ZERO_TOLERANCE_RAD
            print(
                f"[verify] {'PASS' if passed else 'FAIL'}: |q| {'<=' if passed else '>'} "
                f"{ZERO_TOLERANCE_RAD:.3f} rad"
            )
            update_record(
                args.record_file,
                spec,
                {"last_verify_at": utc_now(), "last_verify_position_rad": before[0], "verify_passed": passed},
            )
            return 0 if passed else 2

        if not confirmation(
            f"ZERO {spec.joint}",
            "With the motor still disabled, manually place this joint at the intended MuJoCo q=0 pose.",
        ):
            print("[zero] Cancelled. No zero command sent.")
            return 1

        session.set_zero()
        time.sleep(0.1)
        after = session.read_feedback()
        print_feedback("post-zero", after)
        if umc.is_motor_fault(after[3]) or abs(after[0]) > ZERO_TOLERANCE_RAD:
            raise RuntimeError(
                f"Zero verification failed: q={after[0]:+.4f} rad, error={after[3]}"
            )
        print(f"[zero] Verified persistent-zero command response within {ZERO_TOLERANCE_RAD:.3f} rad.")

        direction_result = run_direction_test(session, args)
        update_record(
            args.record_file,
            spec,
            {
                "zeroed_at": utc_now(),
                "post_zero_position_rad": after[0],
                "last_verify_at": utc_now(),
                "last_verify_position_rad": after[0],
                "verify_passed": True,
                "direction_test": direction_result,
                "test_angle_rad": direction_test_target(args.joint, args.test_angle)
                if direction_result is not None else None,
                "test_kp": args.kp if direction_result is not None else None,
                "test_kd": args.kd if direction_result is not None else None,
            },
        )
        print(f"[record] Updated {args.record_file}")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\n[abort] No further motion command will be sent.")
        return 130
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        if session is not None:
            try:
                session.disable()
                print("[safety] Selected motor disabled.")
            except Exception as exc:
                print(f"[safety] Failed to send selected-motor disable: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
