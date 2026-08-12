#!/usr/bin/env python3
"""Run a small, non-persistent direction test for one UPOO arm motor.

This tool never calls set_zero_position and never sends the 0xFE persistent
zero command. It only reads feedback, enables the selected motor after two
typed confirmations, commands a small test motion, then disables that motor.
"""

from __future__ import annotations

import argparse
import json
import signal
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
ZERO_START_TOLERANCE_RAD = umc.ZERO_VERIFY_TOLERANCE_RAD
FEEDBACK_TIMEOUT_SEC = 0.5
RAMP_RATE_HZ = 50.0
ENABLE_ONLY_DURATION_SEC = 2.0
MODE_SWITCH_SETTLE_SEC = 0.05
MIN_DIRECTION_MOTION_RAD = 0.01
MAX_TEST_KP = umc.MAX_RUNTIME_KP
DEFAULT_TARGET_HOLD_SEC = 1.0
DEFAULT_MAX_TORQUE_TEST_MOTION_RAD = 0.10


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
        description="Run a non-persistent, single-motor UPOO direction test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--joint", required=True, choices=sorted(specs))
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--kp", type=float, default=0.5, help="MIT position gain.")
    parser.add_argument("--kd", type=float, default=0.5, help="MIT velocity gain.")
    parser.add_argument("--test-angle", type=float, default=0.05)
    parser.add_argument("--test-speed", type=float, default=0.01, help="Ramp speed in rad/s.")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=DEFAULT_TARGET_HOLD_SEC,
        help="How long to hold the test target while sampling feedback.",
    )
    parser.add_argument("--record-file", type=Path, default=DEFAULT_RECORD_FILE)
    parser.add_argument(
        "--enable-only",
        action="store_true",
        help="Enable the selected motor briefly and check feedback without sending a motion command.",
    )
    parser.add_argument(
        "--torque-test",
        action="store_true",
        help="Hold the measured position with Kp=Kd=0 and apply only --tau-ff.",
    )
    parser.add_argument(
        "--tau-ff",
        type=float,
        default=0.0,
        help="Feed-forward torque for position or torque tests, limited to +/-0.5.",
    )
    parser.add_argument(
        "--max-torque-motion",
        type=float,
        default=DEFAULT_MAX_TORQUE_TEST_MOTION_RAD,
        help="Abort torque test when motion exceeds this displacement in rad.",
    )
    args = parser.parse_args()
    if args.channel < 0:
        parser.error("--channel must be non-negative")
    if not 0 <= args.kp <= MAX_TEST_KP:
        parser.error(f"--kp must be in [0, {MAX_TEST_KP:.1f}] for this test")
    if not 0 <= args.kd <= 1.0:
        parser.error("--kd must be in [0, 1.0] for this test")
    if not 0 < args.test_angle <= 0.5:
        parser.error("--test-angle must be in (0, 0.5] rad")
    if not 0 < args.test_speed <= 0.02:
        parser.error("--test-speed must be in (0, 0.02] rad/s")
    if not 0 < args.hold_seconds <= 5.0:
        parser.error("--hold-seconds must be in (0, 5.0] s")
    if abs(args.tau_ff) > 0.5:
        parser.error("--tau-ff must be in [-0.5, 0.5]")
    if not 0.02 <= args.max_torque_motion <= 0.5:
        parser.error("--max-torque-motion must be in [0.02, 0.5] rad")
    if args.torque_test and args.enable_only:
        parser.error("--torque-test cannot be combined with --enable-only")
    if args.enable_only and abs(args.tau_ff) > 1e-12:
        parser.error("--tau-ff cannot be combined with --enable-only")
    if not args.torque_test and abs(args.tau_ff) > 1e-12 and args.kp <= 0.0:
        parser.error("position-test --tau-ff requires --kp greater than zero")
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


def confirmation(expected: str, prompt: str) -> bool:
    print(prompt)
    return input(f"Type exactly '{expected}' to continue: ").strip() == expected


class SingleMotorSession:
    """A CAN session that only registers and commands one motor ID."""

    def __init__(self, spec: MotorSpec, channel: int):
        from dmcan import dmcan_device_type
        from kio_teleop_openarm.lib.damiao import (
            Control_Mode,
            Control_Mode_Code,
            DM_Motor_Type,
            DM_REG,
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
        self._mit_mode_code = Control_Mode_Code.MIT
        self._diagnostic_regs = (
            ("CTRL_MODE", DM_REG.CTRL_MODE),
            ("PMAX", DM_REG.PMAX),
            ("VMAX", DM_REG.VMAX),
            ("TMAX", DM_REG.TMAX),
        )

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

    def feedback_timestamp(self) -> float:
        return float(self._motor.last_time_)

    def latest_feedback(self) -> tuple[float, float, float, int]:
        return (
            float(self._motor.Get_Position()),
            float(self._motor.Get_Velocity()),
            float(self._motor.Get_tau()),
            int(self._motor.Get_err()),
        )

    def clear_errors(self) -> None:
        """Use the same selected-motor fault-clear sequence as scan_motors.py."""
        for _ in range(5):
            self._control.control_cmd(
                self._motor.GetCanId(),
                0xFB,
                self._motor.GetChannel(),
            )
            time.sleep(0.005)

    def enable(self) -> None:
        # The all-motor scan naturally gives each mode write time to settle
        # while it configures the other motors. Do that explicitly here.
        if not self._control.switchControlMode(self._motor, self._mit_mode_code):
            raise RuntimeError("Failed to send MIT control-mode command")
        time.sleep(MODE_SWITCH_SETTLE_SEC)
        diagnostics = {}
        for name, register in self._diagnostic_regs:
            value = self._control.read_motor_param(
                self._motor, register, timeout=FEEDBACK_TIMEOUT_SEC
            )
            diagnostics[name] = value
            if value is None:
                raise RuntimeError(f"No response while reading motor parameter {name}")
            print(f"[diagnostic] {name}={value}")
        mode_value = int(diagnostics["CTRL_MODE"])
        if mode_value != int(self._mit_mode_code):
            raise RuntimeError(
                "Motor did not confirm MIT mode: "
                f"CTRL_MODE={mode_value} (expected {int(self._mit_mode_code)})"
            )
        for name in ("PMAX", "VMAX", "TMAX"):
            if float(diagnostics[name]) <= 0.0:
                raise RuntimeError(f"Motor parameter {name} is invalid: {diagnostics[name]}")
        self._motor.limit_param = [
            float(diagnostics["PMAX"]),
            float(diagnostics["VMAX"]),
            float(diagnostics["TMAX"]),
        ]
        print(
            "[diagnostic] Applied MIT limits: "
            f"P={self._motor.get_limit_param()[0]:g}, "
            f"V={self._motor.get_limit_param()[1]:g}, "
            f"T={self._motor.get_limit_param()[2]:g}"
        )
        for _ in range(5):
            self._control.control_cmd(
                self._motor.GetCanId() + self._motor.GetMotorMode(),
                0xFC,
                self._motor.GetChannel(),
            )
            time.sleep(0.002)

    def disable(self) -> None:
        self._control.disable_all()

    def command(self, kp: float, kd: float, position: float, tau_ff: float = 0.0) -> None:
        self._control.control_mit(self._motor, kp, kd, position, 0.0, tau_ff)

    def ramp(self, kp: float, kd: float, start: float, target: float, speed: float, tau_ff: float = 0.0) -> None:
        duration = abs(target - start) / speed
        steps = max(1, int(duration * RAMP_RATE_HZ))
        for step in range(steps + 1):
            status = int(self._motor.Get_err())
            if umc.is_motor_fault(status):
                raise RuntimeError(
                    f"Motor fault 0x{status:X} ({umc.motor_status_label(status)}) during test"
                )
            position = start + (target - start) * step / steps
            self.command(kp, kd, position, tau_ff)
            time.sleep(1.0 / RAMP_RATE_HZ)

    def hold_and_observe(
        self,
        kp: float,
        kd: float,
        target: float,
        duration: float,
        reference: float,
        tau_ff: float = 0.0,
        motion_limit: float | None = None,
    ) -> tuple[tuple[float, float, float, int], float]:
        """Hold a target and return the sample with largest observed motion."""
        deadline = time.monotonic() + duration
        peak: tuple[float, float, float, int] | None = None
        max_motion = 0.0
        while time.monotonic() < deadline:
            status = int(self._motor.Get_err())
            if umc.is_motor_fault(status):
                raise RuntimeError(
                    f"Motor fault 0x{status:X} ({umc.motor_status_label(status)}) during hold"
                )
            self.command(kp, kd, target, tau_ff)
            time.sleep(1.0 / RAMP_RATE_HZ)
            sample = self.latest_feedback()
            motion = abs(sample[0] - reference)
            if peak is None or motion > max_motion:
                peak = sample
                max_motion = motion
            if motion_limit is not None and max_motion > motion_limit:
                self.command(0.0, 0.0, target, 0.0)
                raise RuntimeError(
                    f"Motion limit exceeded: {max_motion:.4f} rad "
                    f"> {motion_limit:.4f} rad; torque removed before disable"
                )
        if peak is None:
            peak = self.read_feedback()
            max_motion = abs(peak[0] - reference)
        return peak, max_motion


def print_feedback(label: str, sample: tuple[float, float, float, int]) -> None:
    position, velocity, torque, status = sample
    print(
        f"[{label}] q={position:+.4f} rad, dq={velocity:+.4f} rad/s, "
        f"tau={torque:+.4f}, status={status} ({umc.motor_status_label(status)})"
    )


def read_observation(test_target: float) -> str:
    expected_direction = "positive" if test_target > 0 else "negative"
    while True:
        result = input(
            f"Did {test_target:+.3f} rad move in the MuJoCo {expected_direction} direction? "
            "Type OK, REVERSED, or UNKNOWN: "
        ).strip().upper()
        if result in {"OK", "REVERSED", "UNKNOWN"}:
            return result
        print("Please type OK, REVERSED, or UNKNOWN.")


def monitor_enabled(session: SingleMotorSession, duration: float, since: float) -> None:
    """Listen for enabled feedback without transmitting a status request."""
    deadline = time.monotonic() + duration
    last_sample: tuple[float, float, float, int] | None = None
    received_enabled_feedback = False
    while time.monotonic() < deadline:
        last_sample = session.latest_feedback()
        if umc.is_motor_fault(last_sample[3]):
            raise RuntimeError(
                f"Motor fault 0x{last_sample[3]:X} "
                f"({umc.motor_status_label(last_sample[3])}) after enable"
            )
        received_enabled_feedback |= session.feedback_timestamp() > since
        time.sleep(0.05)
    if not received_enabled_feedback:
        raise TimeoutError("No unsolicited CAN feedback received after enable")
    if last_sample is not None:
        print_feedback("enable-only", last_sample)


def main() -> int:
    args = parse_args()
    spec = motor_specs()[args.joint]
    test_target = direction_test_target(args.joint, args.test_angle)
    session: SingleMotorSession | None = None
    previous_sigint_handler = None
    try:
        print(
            f"[safety] Selected only {spec.joint}: CAN ID 0x{spec.can_id:02X}, "
            f"master ID 0x{spec.mst_id:02X}, channel {args.channel}."
        )
        print("[safety] This tool never writes persistent zero or Flash data.")
        print("[safety] Stop every other CAN-control program before continuing.")
        session = SingleMotorSession(spec, args.channel)
        # damiao installs a handler that only clears its own event. Restore the
        # normal Python handler so Ctrl+C reaches our cleanup/finally path.
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        initial = session.read_feedback()
        print_feedback("pre-test", initial)
        if umc.is_motor_fault(initial[3]):
            raise RuntimeError(
                f"Selected motor reports fault 0x{initial[3]:X} "
                f"({umc.motor_status_label(initial[3])}); refusing to enable"
            )
        if abs(initial[0]) > ZERO_START_TOLERANCE_RAD:
            raise RuntimeError(
                f"Refusing test because |q|={abs(initial[0]):.4f} rad exceeds "
                f"{ZERO_START_TOLERANCE_RAD:.3f} rad; manually return to q=0 first"
            )

        if not confirmation(
            f"ARM {spec.joint}",
            "Confirm that the selected joint has clear travel, the physical emergency stop is reachable, "
            "and the robot is at rest.",
        ):
            print("[test] Cancelled before motor enable. No motion command sent.")
            return 1
        if args.enable_only:
            action_prompt = (
                "The selected motor will be enabled for "
                f"{ENABLE_ONLY_DURATION_SEC:.1f} seconds without any position or MIT motion command. "
                "No persistent data will be written."
            )
        elif args.torque_test:
            action_prompt = (
                "The selected motor will be enabled with Kp=0, Kd=0, "
                f"q_des=the measured start position, and tau_ff={args.tau_ff:+.3f}. "
                f"It will hold for {args.hold_seconds:.1f} s, then remove torque. "
                "No persistent data will be written."
            )
        else:
            action_prompt = (
                "The selected motor will be enabled with "
                f"Kp={args.kp:.1f}, Kd={args.kd:.1f}, tau_ff={args.tau_ff:+.3f} and move from the measured "
                f"start position to {test_target:+.3f} rad, hold the target for "
                f"{args.hold_seconds:.1f} s, then return. No persistent data will be written."
            )
        if not confirmation("TEST", action_prompt):
            print("[test] Cancelled before motor enable. No motion command sent.")
            return 1

        if args.enable_only:
            session.clear_errors()
            print("[safety] Sent selected-motor fault-clear commands before enable.")
        feedback_before_enable = session.feedback_timestamp()
        session.enable()
        peak = None
        try:
            if args.enable_only:
                monitor_enabled(session, ENABLE_ONLY_DURATION_SEC, feedback_before_enable)
            elif args.torque_test:
                start_position = initial[0]
                session.command(0.0, 0.0, start_position, 0.0)
                time.sleep(0.2)
                peak, max_motion = session.hold_and_observe(
                    0.0,
                    0.0,
                    start_position,
                    args.hold_seconds,
                    start_position,
                    args.tau_ff,
                    args.max_torque_motion,
                )
                print_feedback("torque peak", peak)
                print(f"[torque-test] max observed motion={max_motion:.4f} rad")
                session.command(0.0, 0.0, start_position, 0.0)
                time.sleep(0.2)
                print_feedback("torque removed", session.read_feedback())
            else:
                start_position = initial[0]
                session.command(args.kp, args.kd, start_position)
                time.sleep(0.2)
                session.ramp(args.kp, args.kd, start_position, test_target, args.test_speed, args.tau_ff)
                peak, max_motion = session.hold_and_observe(
                    args.kp, args.kd, test_target, args.hold_seconds, start_position,
                    args.tau_ff,
                )
                print_feedback("test peak", peak)
                print(f"[test] max observed motion={max_motion:.4f} rad")
                session.ramp(args.kp, args.kd, test_target, start_position, args.test_speed, 0.0)
                print_feedback("test return", session.read_feedback())
        finally:
            session.disable()
            operation = (
                "enable-only check"
                if args.enable_only
                else "torque test"
                if args.torque_test
                else "direction test"
            )
            print(f"[safety] Selected motor disabled after {operation}.")

        if args.enable_only:
            print("[result] Enable-only check passed.")
        elif args.torque_test:
            result = "MOTION_DETECTED" if max_motion >= MIN_DIRECTION_MOTION_RAD else "NO_MOTION"
            print(f"[result] Torque test: {result}")
            update_record(
                args.record_file,
                spec,
                {
                    "torque_test_at": utc_now(),
                    "torque_test_ff": args.tau_ff,
                    "torque_test_hold_seconds": args.hold_seconds,
                    "torque_test_max_motion_rad": max_motion,
                    "torque_test_result": result,
                },
            )
            print(f"[record] Updated {args.record_file}")
        else:
            if peak is None or max_motion < MIN_DIRECTION_MOTION_RAD:
                observation = "UNKNOWN"
                print(
                    "[result] Insufficient encoder motion; "
                    "recording direction as UNKNOWN."
                )
            else:
                observation = read_observation(test_target)
            update_record(
                args.record_file,
                spec,
                {
                    "direction_test_at": utc_now(),
                    "direction_test": observation,
                    "test_angle_rad": test_target,
                    "test_kp": args.kp,
                    "test_kd": args.kd,
                    "test_tau_ff": args.tau_ff,
                    "test_hold_seconds": args.hold_seconds,
                },
            )
            print(f"[result] Direction observation: {observation}")
            print(f"[record] Updated {args.record_file}")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\n[abort] No further motion command will be sent.")
        return 130
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)
        if session is not None:
            try:
                session.disable()
                print("[safety] Selected motor disabled.")
            except Exception as exc:
                print(f"[safety] Failed to send selected-motor disable: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
