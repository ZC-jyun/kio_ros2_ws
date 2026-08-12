#!/usr/bin/env python3
"""Return the six UPOO arm joints to their persistent encoder zero positions.

The gripper is never registered or commanded. The arm follows one synchronized
minimum-jerk path from its measured pose to MuJoCo q=0. Every exit path sends
motor-disable frames.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PACKAGE_SRC = Path(__file__).resolve().parents[2]
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from kio_teleop_openarm.lib import upoo_motor_constants as umc


ARM_JOINTS = tuple(name for name, _, _ in umc.ARM_MOTOR_CONFIG)
DEFAULT_MAX_SPEED_RAD_S = 0.10
DEFAULT_RATE_HZ = 100.0
DEFAULT_HOLD_SECONDS = 0.5
DEFAULT_TRACKING_ERROR_RAD = 0.25
DEFAULT_START_VELOCITY_RAD_S = 0.10
MAX_PRE_ENABLE_DRIFT_RAD = 0.02
FEEDBACK_TIMEOUT_SEC = 0.25
PARAMETER_TIMEOUT_SEC = 0.5
MINIMUM_JERK_MAX_SLOPE = 1.875
CONFIRMATION_TEXT = "RETURN TO ZERO"


@dataclass(frozen=True)
class ArmFeedback:
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    torque: tuple[float, ...]
    status: tuple[int, ...]
    timestamps: tuple[float, ...]


def validate_zero_calibration(path: Path) -> dict:
    """Require matching CAN IDs and a successful zero verification."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Calibration record not found: {path}")
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    motors = data.get("motors")
    if not isinstance(motors, dict):
        raise RuntimeError(f"Invalid calibration record: {path}")

    problems = []
    for joint, can_id, mst_id in umc.ARM_MOTOR_CONFIG:
        record = motors.get(joint, {})
        if record.get("can_id") != can_id or record.get("mst_id") != mst_id:
            problems.append(f"{joint}: CAN IDs are missing or do not match")
        if record.get("verify_passed") is not True:
            problems.append(f"{joint}: zero verification has not passed")
        position = record.get("last_verify_position_rad")
        if (
            isinstance(position, bool)
            or not isinstance(position, (int, float))
            or not math.isfinite(position)
            or abs(position) > umc.ZERO_VERIFY_TOLERANCE_RAD
        ):
            problems.append(f"{joint}: last verified zero position is invalid")
    if problems:
        raise RuntimeError("Zero calibration is incomplete:\n  - " + "\n  - ".join(problems))
    return data


def minimum_jerk_fraction(elapsed: float, duration: float) -> float:
    """Return a quintic blend in [0, 1] with zero endpoint velocity/acceleration."""
    if duration <= 0.0:
        return 1.0
    u = min(1.0, max(0.0, float(elapsed) / float(duration)))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def trajectory_duration(start: Sequence[float], max_speed: float) -> float:
    """Choose a duration which bounds every minimum-jerk joint speed."""
    if max_speed <= 0.0:
        raise ValueError("max_speed must be positive")
    max_distance = max((abs(float(value)) for value in start), default=0.0)
    return MINIMUM_JERK_MAX_SLOPE * max_distance / float(max_speed)


def trajectory_target(start: Sequence[float], fraction: float) -> tuple[float, ...]:
    remaining = 1.0 - min(1.0, max(0.0, float(fraction)))
    return tuple(float(value) * remaining for value in start)


def format_vector(values: Sequence[float]) -> str:
    return " ".join(
        f"{joint}={float(value):+.4f}" for joint, value in zip(ARM_JOINTS, values)
    )


class ArmMotorSession:
    """A direct CAN session which registers only the six arm motors."""

    def __init__(self, channel: int):
        from dmcan import dmcan_device_type
        from kio_teleop_openarm.lib.damiao import (
            Control_Mode,
            Control_Mode_Code,
            DM_Motor_Type,
            DM_REG,
            DmActData,
            Motor_Control,
        )

        # damiao installs a SIGINT handler which does not raise
        # KeyboardInterrupt. Restore normal CLI interruption before opening CAN.
        signal.signal(signal.SIGINT, signal.default_int_handler)

        self._control_mode_code = Control_Mode_Code.MIT
        self._control_mode = Control_Mode.MIT_MODE
        self._control_mode_register = DM_REG.CTRL_MODE
        self._registers = (
            ("PMAX", DM_REG.PMAX),
            ("VMAX", DM_REG.VMAX),
            ("TMAX", DM_REG.TMAX),
        )
        init_data = [
            DmActData(
                motorType=getattr(DM_Motor_Type, umc.ARM_MOTOR_TYPES[joint]),
                mode=Control_Mode.MIT_MODE,
                can_id=can_id,
                mst_id=mst_id,
                channel=channel,
            )
            for joint, can_id, mst_id in umc.ARM_MOTOR_CONFIG
        ]
        self._control = Motor_Control(
            umc.NOM_BAUD,
            umc.DAT_BAUD,
            sn=umc.USB2CANFD_SN,
            data_ptr=init_data,
            device_type=dmcan_device_type.USB2CANFD,
            auto_enable=False,
        )
        self._motors = []
        for joint, can_id, _ in umc.ARM_MOTOR_CONFIG:
            motor = self._control.getMotor(channel, can_id)
            if motor is None:
                raise RuntimeError(f"Motor {joint} was not registered")
            self._motors.append(motor)
        self._enabled = False

    def validate_parameters(self, switch_to_mit: bool) -> None:
        for index, (joint, _, _) in enumerate(umc.ARM_MOTOR_CONFIG):
            motor = self._motors[index]
            if switch_to_mit:
                if not self._control.switchControlMode(motor, self._control_mode_code):
                    raise RuntimeError(f"Failed to switch {joint} to MIT mode")
                time.sleep(0.05)

            mode = self._control.read_motor_param(
                motor, self._control_mode_register, timeout=PARAMETER_TIMEOUT_SEC
            )
            if mode is None or int(mode) != int(self._control_mode_code):
                raise RuntimeError(
                    f"Refusing to enable {joint}: CTRL_MODE={mode!r}, expected MIT=1"
                )

            actual = []
            for register_name, register in self._registers:
                value = self._control.read_motor_param(
                    motor, register, timeout=PARAMETER_TIMEOUT_SEC
                )
                if value is None:
                    raise RuntimeError(f"No {register_name} response from {joint}")
                actual.append(float(value))
            expected = umc.EXPECTED_MIT_LIMITS[umc.ARM_MOTOR_TYPES[joint]]
            if any(
                not math.isclose(value, wanted, rel_tol=0.0, abs_tol=1e-3)
                for value, wanted in zip(actual, expected)
            ):
                raise RuntimeError(
                    f"Refusing to enable {joint}: MIT limits {tuple(actual)} "
                    f"do not match expected {expected}"
                )
            motor.limit_param = actual
            print(
                f"[motor] {joint}: direction={int(umc.JOINT_DIRECTION[joint]):+d}, "
                f"MIT limits={tuple(actual)}"
            )

    def read_fresh_feedback(self, timeout: float = 0.5) -> ArmFeedback:
        previous = [float(motor.last_time_) for motor in self._motors]
        for motor in self._motors:
            self._control.refresh_motor_status(motor)
        pending = set(range(len(self._motors)))
        deadline = time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            pending = {
                index
                for index in pending
                if float(self._motors[index].last_time_) <= previous[index]
            }
            if pending:
                time.sleep(0.005)
        if pending:
            names = ", ".join(ARM_JOINTS[index] for index in sorted(pending))
            raise TimeoutError(f"No fresh CAN feedback from: {names}")
        return self.latest_feedback()

    def latest_feedback(self) -> ArmFeedback:
        position = []
        velocity = []
        torque = []
        status = []
        timestamps = []
        for index, motor in enumerate(self._motors):
            direction = float(umc.JOINT_DIRECTION[ARM_JOINTS[index]])
            position.append(direction * float(motor.Get_Position()))
            velocity.append(direction * float(motor.Get_Velocity()))
            torque.append(direction * float(motor.Get_tau()))
            status.append(int(motor.Get_err()))
            timestamps.append(float(motor.last_time_))
        numeric = position + velocity + torque + timestamps
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError("Motor feedback contains a non-finite value")
        return ArmFeedback(
            tuple(position), tuple(velocity), tuple(torque), tuple(status), tuple(timestamps)
        )

    def command_all(
        self, model_targets: Sequence[float], kp: Sequence[float], kd: Sequence[float]
    ) -> None:
        for index, motor in enumerate(self._motors):
            joint = ARM_JOINTS[index]
            motor_target = umc.mujoco_to_motor(joint, float(model_targets[index]))
            sent = self._control.control_mit(
                motor, float(kp[index]), float(kd[index]), motor_target, 0.0, 0.0
            )
            if not sent:
                raise RuntimeError(f"Failed to send MIT command to {joint}")

    def enable_at(
        self, model_positions: Sequence[float], kp: Sequence[float], kd: Sequence[float]
    ) -> None:
        # Load current-position targets while disabled, then enable and hold each
        # motor before proceeding to the next one.
        self.command_all(model_positions, kp, kd)
        for index, motor in enumerate(self._motors):
            joint = ARM_JOINTS[index]
            motor_target = umc.mujoco_to_motor(joint, float(model_positions[index]))
            for _ in range(5):
                self._control.control_cmd(
                    motor.GetCanId() + self._control_mode,
                    0xFC,
                    motor.GetChannel(),
                )
                time.sleep(0.002)
            for _ in range(3):
                sent = self._control.control_mit(
                    motor,
                    float(kp[index]),
                    float(kd[index]),
                    motor_target,
                    0.0,
                    0.0,
                )
                if not sent:
                    raise RuntimeError(f"Failed to hold {joint} after enable")
                time.sleep(0.002)
        self._enabled = True

    def assert_runtime_safe(
        self,
        target: Sequence[float],
        max_tracking_error: float,
        feedback_grace_until: float,
    ) -> ArmFeedback:
        feedback = self.latest_feedback()
        now = time.monotonic()
        for index, status in enumerate(feedback.status):
            if umc.is_motor_fault(status):
                raise RuntimeError(
                    f"{ARM_JOINTS[index]} fault 0x{status:X} "
                    f"({umc.motor_status_label(status)})"
                )
            if now >= feedback_grace_until:
                age = now - feedback.timestamps[index]
                if age > FEEDBACK_TIMEOUT_SEC:
                    raise TimeoutError(
                        f"CAN feedback timeout from {ARM_JOINTS[index]}: {age:.3f}s"
                    )
        error = max(
            abs(actual - wanted)
            for actual, wanted in zip(feedback.position, target)
        )
        if now >= feedback_grace_until and error > max_tracking_error:
            raise RuntimeError(
                f"Tracking error {error:.4f} rad exceeds {max_tracking_error:.4f} rad"
            )
        return feedback

    def disable(self) -> None:
        try:
            self._control.disable_all()
            time.sleep(0.05)
            if self._enabled:
                print("[safety] All six arm motors disabled.")
        finally:
            self._enabled = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Return all six UPOO arm joints to MuJoCo q=0.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED_RAD_S)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS)
    parser.add_argument(
        "--tolerance", type=float, default=umc.ZERO_VERIFY_TOLERANCE_RAD
    )
    parser.add_argument(
        "--max-tracking-error", type=float, default=DEFAULT_TRACKING_ERROR_RAD
    )
    parser.add_argument(
        "--max-start-velocity", type=float, default=DEFAULT_START_VELOCITY_RAD_S
    )
    parser.add_argument(
        "--kp", type=float, nargs=6, default=list(umc.DEFAULT_KP[: umc.ARM_DOF])
    )
    parser.add_argument(
        "--kd", type=float, nargs=6, default=list(umc.DEFAULT_KD[: umc.ARM_DOF])
    )
    parser.add_argument(
        "--calibration-record", type=Path, default=umc.CALIBRATION_RECORD
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and validate state, but never enable motors or send motion targets.",
    )
    args = parser.parse_args(argv)

    if args.channel < 0:
        parser.error("--channel must be non-negative")
    max_configured_speed = min(float(value) for value in umc.MAX_COMMAND_SPEED[:6])
    if not 0.0 < args.max_speed <= max_configured_speed:
        parser.error(f"--max-speed must be in (0, {max_configured_speed:g}]")
    if not 20.0 <= args.rate <= 500.0:
        parser.error("--rate must be in [20, 500] Hz")
    if not 0.0 <= args.hold_seconds <= 5.0:
        parser.error("--hold-seconds must be in [0, 5] seconds")
    if not 0.005 <= args.tolerance <= 0.10:
        parser.error("--tolerance must be in [0.005, 0.10] rad")
    if not 0.05 <= args.max_tracking_error <= 1.0:
        parser.error("--max-tracking-error must be in [0.05, 1.0] rad")
    if not 0.01 <= args.max_start_velocity <= 0.5:
        parser.error("--max-start-velocity must be in [0.01, 0.5] rad/s")
    if any(not 0.0 < value <= umc.MAX_RUNTIME_KP for value in args.kp):
        parser.error(
            f"every --kp value must be in (0, {umc.MAX_RUNTIME_KP:g}]"
        )
    if any(not 0.0 <= value <= umc.MAX_RUNTIME_KD for value in args.kd):
        parser.error(
            f"every --kd value must be in [0, {umc.MAX_RUNTIME_KD:g}]"
        )
    return args


def validate_start(feedback: ArmFeedback, max_velocity: float) -> None:
    problems = []
    for index, joint in enumerate(ARM_JOINTS):
        q = feedback.position[index]
        dq = feedback.velocity[index]
        status = feedback.status[index]
        lo, hi = umc.SOFT_POSITION_LIMITS[joint]
        if umc.is_motor_fault(status):
            problems.append(
                f"{joint}: fault 0x{status:X} ({umc.motor_status_label(status)})"
            )
        limit_tolerance = umc.ZERO_VERIFY_TOLERANCE_RAD
        if not lo - limit_tolerance <= q <= hi + limit_tolerance:
            problems.append(
                f"{joint}: q={q:+.4f} outside soft limits [{lo:+.4f}, {hi:+.4f}] "
                f"plus {limit_tolerance:.4f} rad zero tolerance"
            )
        if abs(dq) > max_velocity:
            problems.append(
                f"{joint}: |dq|={abs(dq):.4f} exceeds {max_velocity:.4f} rad/s"
            )
    if problems:
        raise RuntimeError("Refusing to enable:\n  - " + "\n  - ".join(problems))


def execute_return(args: argparse.Namespace) -> int:
    validate_zero_calibration(args.calibration_record)
    print(f"[preflight] Zero calibration record passed: {args.calibration_record}")
    print("[safety] Stop every other CAN-control program before continuing.")
    print("[safety] The gripper is not registered and will not be commanded.")

    session: ArmMotorSession | None = None
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        session = ArmMotorSession(args.channel)
        # damiao installs a handler which does not raise KeyboardInterrupt.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        session.validate_parameters(switch_to_mit=not args.dry_run)
        initial = session.read_fresh_feedback()
        validate_start(initial, args.max_start_velocity)
        print(f"[state] q:   {format_vector(initial.position)}")
        print(f"[state] dq:  {format_vector(initial.velocity)}")
        print(f"[state] tau: {format_vector(initial.torque)}")

        max_error = max(abs(value) for value in initial.position)
        if max_error <= args.tolerance:
            print(
                f"[result] Already at zero: max |q|={max_error:.4f} rad "
                f"<= {args.tolerance:.4f} rad. Motors were not enabled."
            )
            return 0

        duration = trajectory_duration(initial.position, args.max_speed)
        print(
            f"[plan] Synchronized minimum-jerk return: {duration:.2f}s, "
            f"max speed <= {args.max_speed:.3f} rad/s"
        )
        if args.dry_run:
            print("[dry-run] Preflight passed. Motors were not enabled.")
            return 0

        print(
            "[warning] Confirm the straight joint-space path to q=0 is collision-free, "
            "support the arm, clear the workspace, and keep the emergency stop reachable."
        )
        confirmed = input(
            f"Type exactly '{CONFIRMATION_TEXT}' to enable the arm: "
        ).strip()
        if confirmed != CONFIRMATION_TEXT:
            print("[abort] Confirmation did not match. Motors were not enabled.")
            return 1

        pre_enable = session.read_fresh_feedback()
        validate_start(pre_enable, args.max_start_velocity)
        drift = max(
            abs(current - shown)
            for current, shown in zip(pre_enable.position, initial.position)
        )
        if drift > MAX_PRE_ENABLE_DRIFT_RAD:
            raise RuntimeError(
                f"Arm moved {drift:.4f} rad after preflight; refusing to enable. "
                "Run the program again from the new pose."
            )
        initial = pre_enable
        duration = trajectory_duration(initial.position, args.max_speed)
        session.enable_at(initial.position, args.kp, args.kd)
        print("[motion] Motors enabled at the measured hold positions.")
        period = 1.0 / args.rate
        start_time = time.monotonic()
        next_tick = start_time
        next_report = start_time
        feedback_grace_until = start_time + 0.5

        while True:
            now = time.monotonic()
            elapsed = now - start_time
            fraction = minimum_jerk_fraction(elapsed, duration)
            target = trajectory_target(initial.position, fraction)
            session.command_all(target, args.kp, args.kd)
            feedback = session.assert_runtime_safe(
                target, args.max_tracking_error, feedback_grace_until
            )
            if now >= next_report:
                print(
                    f"[motion] {min(elapsed, duration):6.2f}/{duration:.2f}s "
                    f"q: {format_vector(feedback.position)}"
                )
                next_report = now + 1.0
            if elapsed >= duration:
                break
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))

        hold_deadline = time.monotonic() + args.hold_seconds
        zero_target = (0.0,) * umc.ARM_DOF
        while time.monotonic() < hold_deadline:
            session.command_all(zero_target, args.kp, args.kd)
            session.assert_runtime_safe(
                zero_target, args.max_tracking_error, feedback_grace_until
            )
            time.sleep(period)

        final = session.read_fresh_feedback()
        final_error = max(abs(value) for value in final.position)
        print(f"[final] q: {format_vector(final.position)}")
        if final_error > args.tolerance:
            print(
                f"[result] FAIL: max |q|={final_error:.4f} rad exceeds "
                f"{args.tolerance:.4f} rad",
                file=sys.stderr,
            )
            return 2
        print(
            f"[result] PASS: all arm joints are within {args.tolerance:.4f} rad of zero."
        )
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\n[abort] Interrupted; disabling all registered arm motors.", file=sys.stderr)
        return 130
    finally:
        if session is not None:
            try:
                session.disable()
            except Exception as exc:
                print(f"[safety] Motor disable failed: {exc}", file=sys.stderr)
        signal.signal(signal.SIGINT, previous_sigint_handler)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return execute_return(parse_args(argv))
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
