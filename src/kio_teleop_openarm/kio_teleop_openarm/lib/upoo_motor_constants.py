"""
Motor-to-joint mapping constants for the UPOO 6-DOF robotic arm.

All motors are DM4310_48V, controlled via USB2CANFD in MIT mode.
DM motor position units = radians (1:1 mapping, no gear ratio).
"""

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

# ── Motor type & control mode ───────────────────────────────────
MOTOR_TYPE = "DM4310_48V"         # all motors are DM4310_48V
CONTROL_MODE = "MIT_MODE"

# ── MIT control gains per joint ────────────────────────────────
# [Base_J01, J02, J03, J04, J05, J06, gripper]
DEFAULT_KP = [1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.5]
DEFAULT_KD = [0.3, 0.5, 0.5, 0.5, 0.3, 0.3, 0.8]

# ── Soft position limits (radians) ─────────────────────────────
# Slightly tighter than MuJoCo joint ranges for safety margin.
# Gripper: mechanical range in meters (lead-screw or direct drive).
SOFT_POSITION_LIMITS = {
    "Base_J01": (-1.745,  1.745),   # -100° ~ 100°
    "J02":      (-1.00,  1.50),
    "J03":      (-1.50,  1.50),
    "J04":      (-0.70,  2.50),
    "J05":      (-1.50,  1.50),
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
    "J05":      (-1.57,  1.57),
    "J06":      (-1.57,  1.57),
    "gripper":  ( 0.00,  5.00),
}

# ── USB2CANFD device ───────────────────────────────────────────
USB2CANFD_SN = "2EBF423413AA04B9E80688FE6504D508"
NOM_BAUD = 1_000_000      # CAN nominal baud rate
DAT_BAUD = 5_000_000      # CANFD data baud rate

# ── Motor control parameters ────────────────────────────────────
MOTOR_CTRL_FREQ = 1000.0   # Hz
MOTOR_SMOOTHING = 0.3      # exponential smoothing (0=instant, 1=no change)
CAN_TIMEOUT_SEC = 0.2      # trigger estop if no CAN frame within this window
IK_DIVERGENCE_THRESH = 0.5 # rad, skip motor update if IK dq exceeds this
