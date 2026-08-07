#!/usr/bin/env python3
"""Check whether recorded UPOO calibration is sufficient for hardware VR."""

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE_SRC = WORKSPACE / "src" / "kio_teleop_openarm"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from kio_teleop_openarm.lib import upoo_motor_constants as umc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-file", type=Path, default=umc.CALIBRATION_RECORD)
    parser.add_argument("--require-control-tests", action="store_true")
    args = parser.parse_args()

    try:
        data = umc.validate_calibration_record(
            args.record_file, require_control_tests=args.require_control_tests)
        result = "READY"
        exit_code = 0
    except Exception as exc:
        data = None
        result = str(exc)
        exit_code = 2

    if data is None:
        import json
        if args.record_file.exists():
            data = json.loads(args.record_file.read_text(encoding="utf-8"))
        else:
            data = {"motors": {}}
    motors = data.get("motors", {})
    print("joint       motor         sign  direction  zero  mapped-control")
    print("----------- ------------- ----- ---------- ----- --------------")
    for joint, _, _ in umc.ARM_MOTOR_CONFIG:
        record = motors.get(joint, {})
        print(
            f"{joint:<11} {umc.ARM_MOTOR_TYPES[joint]:<13} "
            f"{int(umc.JOINT_DIRECTION[joint]):+d}    "
            f"{str(record.get('direction_test')):<10} "
            f"{str(record.get('verify_passed')):<5} "
            f"{str(record.get('mapped_control_test_passed')):<14}"
        )
    print(f"[preflight] {result}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
