"""Motor and MuJoCo-joint mapping for the UPOO 6-DOF arm."""

import json
from pathlib import Path

# ── CAN IDs ────────────────────────────────────────────────────
# Each arm joint: (joint_name, send_can_id, recv_mst_id)
# CAN IDs 0x02-0x07 for arm joints (0x01 reserved for gripper)
ARM_MOTOR_CONFIG = [
    ("Base_J01", 0x02, 0x12),
    ("J02",      0x03, 0x13),
    ("J03",      0x04, 0x14),
    ("J04",      0x05, 0x15),
    ("J05",      0x06, 0x16),
    ("J06",      0x07, 0x17),
]

# Gripper motor (separate for clarity)
GRIPPER_CAN_ID = 0x01
GRIPPER_MST_ID = 0x11

ARM_DOF = len(ARM_MOTOR_CONFIG)   # 6
NUM_MOTORS = ARM_DOF + 1          # 7 (arm + gripper)

# ── Motor type, direction & control mode ───────────────────────
# motor_value = JOINT_DIRECTION[joint] * mujoco_value
ARM_MOTOR_TYPES = {
    "Base_J01": "DM4340_48V",
    "J02":      "DM4340_48V",
    "J03":      "DM4340_48V",
    "J04":      "DM4310_48V",
    "J05":      "DM4310_48V",
    "J06":      "DM4310_48V",
}
GRIPPER_MOTOR_TYPE = "DM4310_48V"
JOINT_DIRECTION = {
    "Base_J01": 1.0,
    "J02":     -1.0,
    "J03":      1.0,
    "J04":      1.0,
    "J05":     -1.0,
    "J06":      1.0,
}
EXPECTED_MIT_LIMITS = {
    "DM4340_48V": (12.5, 20.0, 28.0),
    "DM4310_48V": (12.5, 50.0, 10.0),
}
CONTROL_MODE = "MIT_MODE"

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CALIBRATION_RECORD = WORKSPACE_ROOT / "data" / "hardware_calibration" / "left_arm_zero_results.json"
ZERO_VERIFY_TOLERANCE_RAD = 0.10

# ── MIT control gains per joint ────────────────────────────────
# [Base_J01, J02, J03, J04, J05, J06, gripper]
DEFAULT_KP = [240.0, 240.0, 120.0, 40.0, 24.0, 31.0, 0.5]
DEFAULT_KD = [5.0, 5.0, 1.5, 0.3, 0.3, 0.3, 0.8]
# The wire protocol can encode Kp up to 500, but the robot application uses
# a deliberately lower bound to limit stiffness and impact energy.
MAX_RUNTIME_KP = 240.0
MAX_RUNTIME_KD = 5.0

# ── Soft position limits (radians) ─────────────────────────────
# Slightly tighter than MuJoCo joint ranges for safety margin.
# Gripper: mechanical range in meters (lead-screw or direct drive).
SOFT_POSITION_LIMITS = {
    "Base_J01": (-1.745,  1.745),   # -100° ~ 100°
    "J02":      (-3.00,  0.00),
    "J03":      (-1.50,  1.50),
    "J04":      (-0.70,  2.50),
    "J05":      (-1.50,  1.60),
    "J06":      (-1.50,  1.50),
    "gripper":  ( 0.00,  5.00),
}

# ── Hardware limits (from MuJoCo XML joint ranges) ─────────────
# These are the absolute joint ranges from the MuJoCo model.
HARD_POSITION_LIMITS = {
    "Base_J01": (-2.82,  2.82),
    "J02":      (-3.14,  0.00),
    "J03":      (-1.57,  1.57),
    "J04":      (-0.78,  2.60),
    "J05":      (-1.57,  1.60),
    "J06":      (-1.57,  1.57),
    "gripper":  ( 0.00,  5.00),
}

# ── USB2CANFD device ───────────────────────────────────────────
USB2CANFD_SN = "2EBF423413AA04B9E80688FE6504D508"
NOM_BAUD = 1_000_000      # CAN nominal baud rate
DAT_BAUD = 5_000_000      # CANFD data baud rate

# ── Motor control parameters ────────────────────────────────────
MOTOR_CTRL_FREQ = 1000.0   # Hz
MOTOR_SMOOTHING = 1.0      # target weight (1=immediate, 0=no motion)
CAN_TIMEOUT_SEC = 0.2      # trigger estop if no CAN frame within this window
IK_DIVERGENCE_THRESH = 0.5 # rad, skip motor update if IK dq exceeds this
MAX_SLEW_DT_SEC = 0.05     # cap one delayed loop's allowed position/torque step

# Maximum command slew rates in joint units per second. These deliberately
# conservative values are intended for the first hardware teleoperation tests.
# [Base_J01, J02, J03, J04, J05, J06, gripper]
MAX_COMMAND_SPEED = [0.25, 0.20, 0.20, 0.30, 0.30, 0.30, 0.50]

# Joint-coordinate gravity feedforward safety bounds. J04 needs 2.63 Nm to
# hold the configured horizontal Home pose; the gripper receives none.
MAX_FEEDFORWARD_TORQUE = [2.8, 2.8, 2.8, 3.0, 1.0, 1.0, 0.0]
MAX_FEEDFORWARD_TORQUE_SLEW = [5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 0.0]

# Motor feedback status nibble. Values 0 and 1 describe operating state;
# actual faults occupy 0x8 through 0xE in the DM protocol.
MOTOR_STATUS_DISABLED = 0x0
MOTOR_STATUS_ENABLED = 0x1
MOTOR_FAULT_CODES = {
    0x8: "overvoltage",
    0x9: "undervoltage",
    0xA: "overcurrent",
    0xB: "MOS overtemperature",
    0xC: "motor coil overtemperature",
    0xD: "communication lost",
    0xE: "overload",
}


def is_motor_fault(status):
    return int(status) in MOTOR_FAULT_CODES


def motor_status_label(status):
    status = int(status)
    if status == MOTOR_STATUS_DISABLED:
        return "disabled"
    if status == MOTOR_STATUS_ENABLED:
        return "enabled"
    return MOTOR_FAULT_CODES.get(status, f"unknown 0x{status:X}")



def arm_direction_vector():
    return [JOINT_DIRECTION[name] for name, _, _ in ARM_MOTOR_CONFIG]


def mujoco_to_motor(joint_name, value):
    return JOINT_DIRECTION[joint_name] * value


def motor_to_mujoco(joint_name, value):
    return JOINT_DIRECTION[joint_name] * value


def startup_position_limits(joint_name):
    """Return pre-enable limits, including zero-calibration tolerance."""
    lo, hi = SOFT_POSITION_LIMITS[joint_name]
    if joint_name in JOINT_DIRECTION:
        tolerance = ZERO_VERIFY_TOLERANCE_RAD
        return lo - tolerance, hi + tolerance
    return lo, hi


def validate_calibration_record(
    path=CALIBRATION_RECORD,
    require_control_tests=False,
    expected_kp=None,
    expected_kd=None,
):
    """Validate direction, zero and optional mapped-control evidence."""
    runtime_kp = DEFAULT_KP if expected_kp is None else list(expected_kp)
    runtime_kd = DEFAULT_KD if expected_kd is None else list(expected_kd)
    if len(runtime_kp) < ARM_DOF or len(runtime_kd) < ARM_DOF:
        raise ValueError(
            f"expected_kp/expected_kd must contain at least {ARM_DOF} values"
        )
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Calibration record not found: {path}")
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    motors = data.get("motors")
    if not isinstance(motors, dict):
        raise RuntimeError(f"Invalid calibration record: {path}")

    problems = []
    for joint_name, can_id, mst_id in ARM_MOTOR_CONFIG:
        record = motors.get(joint_name, {})
        expected_direction = "OK" if JOINT_DIRECTION[joint_name] > 0 else "REVERSED"
        if record.get("can_id") != can_id or record.get("mst_id") != mst_id:
            problems.append(f"{joint_name}: CAN IDs are missing or do not match configuration")
        if record.get("direction_test") != expected_direction:
            problems.append(
                f"{joint_name}: direction={record.get('direction_test')!r}, "
                f"expected {expected_direction!r}"
            )
        if record.get("verify_passed") is not True:
            problems.append(f"{joint_name}: zero verification has not passed")
        verify_position = record.get("last_verify_position_rad")
        if not isinstance(verify_position, (int, float)) or abs(verify_position) > ZERO_VERIFY_TOLERANCE_RAD:
            problems.append(f"{joint_name}: last verified zero position is invalid")
        if require_control_tests:
            control_passed = record.get("mapped_control_test_passed") is True
            if not control_passed:
                problems.append(f"{joint_name}: mapped control test has not passed")
            else:
                joint_index = [name for name, _, _ in ARM_MOTOR_CONFIG].index(joint_name)
                if record.get("mapped_control_test_kp") != runtime_kp[joint_index]:
                    problems.append(f"{joint_name}: mapped control test Kp does not match runtime Kp")
                if record.get("mapped_control_test_kd") != runtime_kd[joint_index]:
                    problems.append(f"{joint_name}: mapped control test Kd does not match runtime Kd")
                if record.get("mapped_control_test_assist_torque") != 0.0:
                    problems.append(f"{joint_name}: mapped control test used unsupported assist torque")
    if problems:
        raise RuntimeError("Hardware calibration is incomplete:\n  - " + "\n  - ".join(problems))
    return data
