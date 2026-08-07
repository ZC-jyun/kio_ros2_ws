#!/usr/bin/env python3
"""Scan CAN bus for online DM-series motors via USB2CANFD.

Usage:
    python3 tools/scan_motors.py [--timeout 2.0]
"""

import argparse
import sys
import time

sys.path.insert(0, "src/kio_teleop_openarm/kio_teleop_openarm/lib")

from damiao import Motor_Control, DmActData, DM_Motor_Type, Control_Mode
from dmcan import dmcan_device_type
import upoo_motor_constants as umc

MOTOR_NAMES = {can_id: name for name, can_id, _mst in umc.ARM_MOTOR_CONFIG}
MOTOR_NAMES[umc.GRIPPER_CAN_ID] = "gripper"


def build_init_data():
    data = []
    for name, can_id, mst_id in umc.ARM_MOTOR_CONFIG:
        data.append(DmActData(
            motorType=getattr(DM_Motor_Type, umc.ARM_MOTOR_TYPES[name]),
            mode=Control_Mode.MIT_MODE,
            can_id=can_id, mst_id=mst_id,
        ))
    data.append(DmActData(
        motorType=getattr(DM_Motor_Type, umc.GRIPPER_MOTOR_TYPE),
        mode=Control_Mode.MIT_MODE,
        can_id=umc.GRIPPER_CAN_ID, mst_id=umc.GRIPPER_MST_ID,
    ))
    return data


def scan(timeout: float = 2.0):
    """Return {can_id: (position, velocity, error_code)} for online motors."""
    init_data = build_init_data()

    control = Motor_Control(
        umc.NOM_BAUD, umc.DAT_BAUD,
        sn=umc.USB2CANFD_SN,
        data_ptr=init_data,
        device_type=dmcan_device_type.USB2CANFD,
        auto_enable=False,
    )

    # Clear errors and enable motors (they won't send status frames otherwise)
    can_ids = [cid for _, cid, _ in umc.ARM_MOTOR_CONFIG] + [umc.GRIPPER_CAN_ID]
    for _ in range(5):
        for cid in can_ids:
            control.control_cmd(cid, 0xFB, 0)
        time.sleep(0.005)
    control.enable_all()
    print("Motors enabled, listening...")

    print(f"Listening for {timeout:.1f}s ...")
    time.sleep(timeout)

    results = {}
    for can_id in MOTOR_NAMES:
        motor = control.getMotor(can_id)
        if motor is None:
            continue
        dt = motor.getTimeInterval()
        if dt > 0.0 and dt < 5.0:
            results[can_id] = (motor.state_q, motor.state_dq, motor.state_err)

    # Do NOT call control.close() — it triggers libusb pthread_mutex_lock crash.
    # Let the OS clean up on process exit.
    return results


def main():
    parser = argparse.ArgumentParser(description="Scan DM motor bus")
    parser.add_argument("--timeout", type=float, default=2.0, help="Scan duration in seconds")
    args = parser.parse_args()

    print(f"Scanning for motors (timeout={args.timeout}s)...")

    for attempt in range(3):
        try:
            results = scan(args.timeout)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print("Waiting 2s before retry...")
                time.sleep(2)
    else:
        print("All attempts failed.")
        sys.exit(1)

    print()
    print(f"{'CAN ID':>8}  {'Name':<14}  {'Position':>10}  {'Velocity':>10}  {'Error':>6}  {'Status'}")
    print("-" * 70)

    for can_id in sorted(MOTOR_NAMES):
        name = MOTOR_NAMES[can_id]
        if can_id in results:
            pos, vel, err = results[can_id]
            print(f"  0x{can_id:02X}    {name:<14}  {pos:>10.4f}  {vel:>10.4f}  {err:>6}  ONLINE")
        else:
            print(f"  0x{can_id:02X}    {name:<14}  {'N/A':>10}  {'N/A':>10}  {'N/A':>6}  OFFLINE")

    online = len(results)
    total = len(MOTOR_NAMES)
    print()
    print(f"{online}/{total} motors online")


if __name__ == "__main__":
    main()
