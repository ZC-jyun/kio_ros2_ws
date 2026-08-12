#!/usr/bin/env python3
"""Safely test one UPOO motor using MuJoCo joint coordinates."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE_SRC = WORKSPACE / "src" / "kio_teleop_openarm"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from kio_teleop_openarm.lib import upoo_motor_constants as umc
from test_upoo_motor_direction import (
    MIN_DIRECTION_MOTION_RAD,
    RAMP_RATE_HZ,
    SingleMotorSession,
    confirmation,
    motor_specs,
    print_feedback,
    update_record,
    utc_now,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test one motor after applying the recorded MuJoCo/motor direction map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--joint", required=True, choices=sorted(motor_specs()))
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--test-angle", type=float, default=0.02)
    parser.add_argument("--test-speed", type=float, default=0.005)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument(
        "--return-settle-seconds",
        type=float,
        default=3.0,
        help="Zero-feedforward hold at the start pose before judging return error.",
    )
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument(
        "--assist-torque", type=float, default=0.0,
        help="Positive MuJoCo-frame torque magnitude applied toward the selected target.",
    )
    parser.add_argument(
        "--return-assist-torque", type=float, default=0.0,
        help="Positive torque magnitude applied toward the start during return.",
    )
    parser.add_argument("--max-motion", type=float, default=1.0)
    parser.add_argument("--record-file", type=Path, default=umc.CALIBRATION_RECORD)
    args = parser.parse_args()
    joint_index = [name for name, _, _ in umc.ARM_MOTOR_CONFIG].index(args.joint)
    if args.kp is None:
        args.kp = umc.DEFAULT_KP[joint_index]
    if args.kd is None:
        args.kd = umc.DEFAULT_KD[joint_index]
    if args.channel < 0:
        parser.error("--channel must be non-negative")
    if not 0 < args.test_angle <= 1.0:
        parser.error("--test-angle must be in (0, 1.0] rad")
    if not 0 < args.test_speed <= 0.02:
        parser.error("--test-speed must be in (0, 0.02] rad/s")
    if not 0 < args.hold_seconds <= 2.0:
        parser.error("--hold-seconds must be in (0, 2.0] s")
    if not 0 < args.return_settle_seconds <= 5.0:
        parser.error("--return-settle-seconds must be in (0, 5.0] s")
    if not 0 < args.kp <= umc.MAX_RUNTIME_KP or not 0 <= args.kd <= umc.MAX_RUNTIME_KD:
        parser.error(
            f"--kp must be in (0, {umc.MAX_RUNTIME_KP:g}] and "
            f"--kd in [0, {umc.MAX_RUNTIME_KD:g}]")
    if not 0 <= args.assist_torque <= 0.5:
        parser.error("--assist-torque must be in [0, 0.5]")
    if not 0 <= args.return_assist_torque <= 0.5:
        parser.error("--return-assist-torque must be in [0, 0.5]")
    if not 0.05 <= args.max_motion <= 1.0:
        parser.error("--max-motion must be in [0.05, 1.0] rad")
    if args.max_motion < args.test_angle:
        parser.error("--max-motion must be greater than or equal to --test-angle")
    return args


def require_selected_joint_calibration(path: Path, joint: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    record = data.get("motors", {}).get(joint, {})
    expected = "OK" if umc.JOINT_DIRECTION[joint] > 0 else "REVERSED"
    if record.get("direction_test") != expected:
        raise RuntimeError(
            f"{joint} direction record is {record.get('direction_test')!r}; expected {expected!r}")
    if record.get("verify_passed") is not True:
        raise RuntimeError(f"{joint} zero verification has not passed")
    q_verify = record.get("last_verify_position_rad")
    if not isinstance(q_verify, (int, float)) or abs(q_verify) > umc.ZERO_VERIFY_TOLERANCE_RAD:
        raise RuntimeError(f"{joint} last verified zero position is invalid: {q_verify!r}")


def choose_target(joint: str, start: float, magnitude: float):
    lo, hi = umc.SOFT_POSITION_LIMITS[joint]
    for delta in (magnitude, -magnitude):
        target = start + delta
        if lo <= target <= hi:
            return target, delta
    raise RuntimeError(
        f"No +/-{magnitude:.3f} rad target from {start:+.4f} inside [{lo:+.3f}, {hi:+.3f}]")


def model_feedback(raw_sample, direction):
    q, dq, tau, status = raw_sample
    return direction * q, direction * dq, direction * tau, status


def main():
    args = parse_args()
    spec = motor_specs()[args.joint]
    require_selected_joint_calibration(args.record_file, args.joint)
    direction = float(umc.JOINT_DIRECTION[args.joint])
    session = None
    previous_sigint_handler = None
    try:
        print(
            f"[mapping] {args.joint}: {umc.ARM_MOTOR_TYPES[args.joint]}, "
            f"CAN ID 0x{spec.can_id:02X}, direction={int(direction):+d}")
        print("[safety] Only the selected motor will be enabled. Stop every other CAN controller.")
        session = SingleMotorSession(spec, args.channel)
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        raw_initial = session.read_feedback()
        initial = model_feedback(raw_initial, direction)
        print_feedback("pre-test MuJoCo", initial)
        if umc.is_motor_fault(initial[3]):
            raise RuntimeError(f"Motor fault 0x{initial[3]:X}")
        start_model = initial[0]
        lo, hi = umc.SOFT_POSITION_LIMITS[args.joint]
        if not lo <= start_model <= hi:
            raise RuntimeError(
                f"Mapped start {start_model:+.4f} is outside soft limits [{lo:+.4f}, {hi:+.4f}]")
        target_model, delta_model = choose_target(args.joint, start_model, args.test_angle)
        start_motor = direction * start_model
        target_motor = direction * target_model
        target_sign = 1.0 if delta_model > 0 else -1.0
        tau_model = target_sign * args.assist_torque
        tau_motor = direction * tau_model
        return_tau_model = -target_sign * args.return_assist_torque
        return_tau_motor = direction * return_tau_model
        prompt = (
            f"The selected joint will move in MuJoCo coordinates {start_model:+.4f} -> "
            f"{target_model:+.4f} rad and return. Motor target={target_motor:+.4f}, "
            f"motor tau_ff={tau_motor:+.3f}, return tau_ff={return_tau_motor:+.3f}. "
            "Keep the emergency stop reachable.")
        if not confirmation(f"TEST {args.joint}", prompt):
            print("[test] Cancelled before motor enable.")
            return 1
        update_record(args.record_file, spec, {
            "mapped_control_test_at": utc_now(),
            "mapped_control_test_passed": False,
            "mapped_control_test_visual": "PENDING",
            "mapped_control_test_kp": args.kp,
            "mapped_control_test_kd": args.kd,
            "mapped_control_test_assist_torque": args.assist_torque,
            "mapped_control_test_return_assist_torque": args.return_assist_torque,
        })
        session.enable()
        try:
            session.command(args.kp, args.kd, start_motor, 0.0)
            time.sleep(0.2)
            session.ramp(
                args.kp, args.kd, start_motor, target_motor,
                args.test_speed, tau_motor)
            raw_peak, max_motion = session.hold_and_observe(
                args.kp, args.kd, target_motor, args.hold_seconds,
                start_motor, tau_motor, args.max_motion)
            peak = model_feedback(raw_peak, direction)
            print_feedback("test peak MuJoCo", peak)
            print(f"[test] max observed motion={max_motion:.4f} rad")
            session.ramp(
                args.kp, args.kd, target_motor, start_motor,
                args.test_speed, return_tau_motor)
            print(
                f"[test] Settling at start for {args.return_settle_seconds:.1f}s "
                "with zero feedforward torque"
            )
            settle_deadline = time.monotonic() + args.return_settle_seconds
            while time.monotonic() < settle_deadline:
                session.command(
                    args.kp, args.kd, start_motor, 0.0
                )
                time.sleep(1.0 / RAMP_RATE_HZ)
                raw_sample = session.latest_feedback()
                status = int(raw_sample[3])
                if umc.is_motor_fault(status):
                    raise RuntimeError(
                        f"Motor fault 0x{status:X} during return settle"
                    )
                if time.monotonic() - session.feedback_timestamp() > 0.5:
                    raise TimeoutError("CAN feedback stale during return settle")
                return_motion = abs(raw_sample[0] - start_motor)
                if return_motion > args.max_motion:
                    session.command(0.0, 0.0, start_motor, 0.0)
                    raise RuntimeError(
                        f"Return motion limit exceeded: {return_motion:.4f} rad"
                    )
            final = model_feedback(session.read_feedback(), direction)
            print_feedback("test return MuJoCo", final)
        finally:
            session.disable()
            print("[safety] Selected motor disabled after mapped control test.")

        observed_delta = peak[0] - start_model
        tracking_ratio = abs(observed_delta) / abs(delta_model)
        return_error = abs(final[0] - start_model)
        automatic_pass = (
            max_motion >= MIN_DIRECTION_MOTION_RAD
            and observed_delta * delta_model > 0.0
            and tracking_ratio >= 0.50
            and return_error <= 0.01
        )
        visual = "UNKNOWN"
        if automatic_pass:
            visual = input(
                f"Did the physical joint match MuJoCo motion toward {target_model:+.4f} rad? "
                "Type OK, REVERSED, or UNKNOWN: ").strip().upper()
            if visual not in {"OK", "REVERSED", "UNKNOWN"}:
                visual = "UNKNOWN"
        passed = automatic_pass and visual == "OK"
        update_record(args.record_file, spec, {
            "mapped_control_test_at": utc_now(),
            "mapped_control_test_passed": passed,
            "mapped_control_test_visual": visual,
            "mapped_control_test_return_settle_seconds": args.return_settle_seconds,
            "mapped_control_test_start_rad": start_model,
            "mapped_control_test_target_rad": target_model,
            "mapped_control_test_peak_rad": peak[0],
            "mapped_control_test_return_rad": final[0],
            "mapped_control_test_tracking_ratio": tracking_ratio,
            "mapped_control_test_return_error_rad": return_error,
            "mapped_control_test_kp": args.kp,
            "mapped_control_test_kd": args.kd,
            "mapped_control_test_assist_torque": args.assist_torque,
            "mapped_control_test_return_assist_torque": args.return_assist_torque,
        })
        print(f"[result] MAPPED CONTROL TEST {'PASS' if passed else 'FAIL'}")
        print(f"[record] Updated {args.record_file}")
        return 0 if passed else 2
    except (KeyboardInterrupt, EOFError):
        print("\n[abort] No further motor command will be sent.")
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
                print(f"[safety] Disable failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
