"""Shared scene constants for VR-collected left-arm basket-drop data."""
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
UPOO_SOURCE = Path("/home/kiorobot/kio_robot_zzc/kio_upoo-main")
if str(UPOO_SOURCE) not in sys.path:
    sys.path.insert(0, str(UPOO_SOURCE))

from collect_act_grasp_data import build_scene_xml, prep_arm_xml

BASKET_X, BASKET_Y, BASKET_BASE_Z = -0.500, -0.200, 0.370
BASKET_INNER_HALF, BASKET_WALL, BASKET_DEPTH, CUP_HALF = 0.060, 0.005, 0.060, 0.020
CUP_X_RANGE = (-0.380, -0.240)
CUP_Y_RANGE = (-0.300, -0.140)
CUP_GRID_X = tuple(round(-0.380 + 0.020 * index, 3) for index in range(8))
CUP_GRID_Y = tuple(round(-0.280 + 0.020 * index, 3) for index in range(8))
CUP_GRID_SIZE = len(CUP_GRID_X) * len(CUP_GRID_Y)
GRASP_HOLD_SECONDS, BASKET_HOLD_SECONDS = 0.10, 0.50
VR_CENTER_CAMERA_XML = '<camera name="vr_center" pos="0.120 0.000 1.080" xyaxes="0 1 0 -0.768221 0 0.640184" fovy="45" resolution="640 480"/>'
WRIST_CAMERA_TARGET_XML = '<body name="upoo_left_wrist_camera_target" pos="0 0 0.080"/>'
WRIST_CAMERA_XML = '<camera name="wrist" pos="-0.060 -0.030 -0.060" fovy="60" mode="targetbody" target="upoo_left_wrist_camera_target" resolution="640 480"/>'

BASKET_OUTER_HALF = BASKET_INNER_HALF + BASKET_WALL
BASKET_WALL_CENTER = BASKET_INNER_HALF + BASKET_WALL / 2
BASKET_BOTTOM_HALF_HEIGHT = BASKET_WALL / 2
BASKET_WALL_HALF_HEIGHT = (BASKET_DEPTH - BASKET_WALL) / 2
BASKET_WALL_CENTER_Z = BASKET_WALL + BASKET_WALL_HALF_HEIGHT

BASKET_XML = f"""
    <!-- Fixed left-arm drop basket: 12 cm inner square, 6 cm deep. -->
    <body name="basket" pos="{BASKET_X:.3f} {BASKET_Y:.3f} {BASKET_BASE_Z:.3f}">
      <geom name="basket_bottom" type="box" pos="0 0 {BASKET_BOTTOM_HALF_HEIGHT:.4f}" size="{BASKET_OUTER_HALF:.4f} {BASKET_OUTER_HALF:.4f} {BASKET_BOTTOM_HALF_HEIGHT:.4f}" rgba="0.12 0.42 0.85 1" friction="1.0 0.1 0.1"/>
      <geom name="basket_left_wall" type="box" pos="-{BASKET_WALL_CENTER:.4f} 0 {BASKET_WALL_CENTER_Z:.4f}" size="{BASKET_BOTTOM_HALF_HEIGHT:.4f} {BASKET_OUTER_HALF:.4f} {BASKET_WALL_HALF_HEIGHT:.4f}" rgba="0.12 0.42 0.85 1" friction="1.0 0.1 0.1"/>
      <geom name="basket_right_wall" type="box" pos="{BASKET_WALL_CENTER:.4f} 0 {BASKET_WALL_CENTER_Z:.4f}" size="{BASKET_BOTTOM_HALF_HEIGHT:.4f} {BASKET_OUTER_HALF:.4f} {BASKET_WALL_HALF_HEIGHT:.4f}" rgba="0.12 0.42 0.85 1" friction="1.0 0.1 0.1"/>
      <geom name="basket_front_wall" type="box" pos="0 -{BASKET_WALL_CENTER:.4f} {BASKET_WALL_CENTER_Z:.4f}" size="{BASKET_INNER_HALF:.4f} {BASKET_BOTTOM_HALF_HEIGHT:.4f} {BASKET_WALL_HALF_HEIGHT:.4f}" rgba="0.12 0.42 0.85 1" friction="1.0 0.1 0.1"/>
      <geom name="basket_back_wall" type="box" pos="0 {BASKET_WALL_CENTER:.4f} {BASKET_WALL_CENTER_Z:.4f}" size="{BASKET_INNER_HALF:.4f} {BASKET_BOTTOM_HALF_HEIGHT:.4f} {BASKET_WALL_HALF_HEIGHT:.4f}" rgba="0.12 0.42 0.85 1" friction="1.0 0.1 0.1"/>
    </body>
"""

_ORIGINAL_TOP_CAMERA = '<camera name="top" pos="-0.35 0.18 0.72" xyaxes="1 0 0 0 1 0" fovy="45" resolution="640 480"/>'
_ORIGINAL_ANGLE_CAMERA = '<camera name="angle" pos="-0.15 0.40 0.60" xyaxes="0.894 -0.447 0 0.229 0.458 -0.858" fovy="50" resolution="640 480"/>'
_LEFT_WRIST_MOTOR_BODY = '<body name="upoo_left_Link_06" pos="-0 -0.0455 0">'
_LEFT_HAND_BODY = '<body name="upoo_left_openarm_v1_hand" pos="0 -0.025 0.1001" quat="1 0 0 0">'


def cup_grid_xy(index):
    """Return one collector-grid point, with X changing before Y."""
    grid_index = int(index) % CUP_GRID_SIZE
    row, column = divmod(grid_index, len(CUP_GRID_X))
    return CUP_GRID_X[column], CUP_GRID_Y[row]


def sample_cup_xy(rng=None, y_range=None):
    """Sample the collector's reachable left-arm object region."""
    generator = rng if rng is not None else __import__("numpy").random
    return generator.uniform(*CUP_X_RANGE), generator.uniform(*(CUP_Y_RANGE if y_range is None else y_range))


def scene_xml(cup_x, cup_y):
    """Build the collector scene with fixed-center and wrist-mounted cameras."""
    arm = prep_arm_xml()
    arm = arm.replace(_ORIGINAL_TOP_CAMERA, VR_CENTER_CAMERA_XML)
    arm = arm.replace(_ORIGINAL_ANGLE_CAMERA, "")
    if _LEFT_WRIST_MOTOR_BODY not in arm or _LEFT_HAND_BODY not in arm:
        raise RuntimeError("Could not find the left wrist motor or hand body for the wrist camera")
    arm = arm.replace(_LEFT_WRIST_MOTOR_BODY, _LEFT_WRIST_MOTOR_BODY + WRIST_CAMERA_XML, 1)
    arm = arm.replace(_LEFT_HAND_BODY, _LEFT_HAND_BODY + WRIST_CAMERA_TARGET_XML, 1)
    return build_scene_xml(cup_x, cup_y, arm.replace("</worldbody>", BASKET_XML + "</worldbody>"))
