#!/usr/bin/env python3
"""Hand-eye calibration: solve T_robotbase_camera using AX=XB.

Requires a running ROS2 system (for /joint_state) and the SPCA2100 camera.

Usage:
  python tools/hand_eye_calibrate.py --output config/hand_eye.yaml
"""
import time
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import mujoco

# Add project root for openarm_mujoco access
_PROJ = Path(__file__).resolve().parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

CHESSBOARD = (7, 4)
SQUARE_SIZE = 0.025  # 25mm

# Joint constants — must match controller.py
LEFT_ARM_JOINTS = [
    "upoo_left_Base_J01", "upoo_left_J02", "upoo_left_J03",
    "upoo_left_J04", "upoo_left_J05", "upoo_left_J06",
]
RIGHT_ARM_JOINTS = [
    "upoo_right_Base_J01", "upoo_right_J02", "upoo_right_J03",
    "upoo_right_J04", "upoo_right_J05", "upoo_right_J06",
]
FINGER_JOINTS = [
    "upoo_left_finger_left_joint", "upoo_left_finger_right_joint",
    "upoo_right_finger_left_joint", "upoo_right_finger_right_joint",
]
ALL_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + FINGER_JOINTS
LEFT_EE_BODY = "upoo_left_Link_06"


class HandEyeCollector(Node):
    """ROS2 node that collects hand-eye calibration data."""

    def __init__(self, calib_npz_path):
        super().__init__("hand_eye_collector")
        self._calib = np.load(calib_npz_path)
        self.K = self._calib["K_left"]

        # Load MuJoCo kinematic model
        import openarm_mujoco
        xml_path = openarm_mujoco.openarm_upoo_bimanual_xml()
        self.get_logger().info(f"Loading model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Joint qpos indices
        self.joint_qpos = {}
        for name in ALL_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_qpos[name] = self.model.jnt_qposadr[jid]

        self.left_ee_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_BODY)

        # Camera
        self.cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Subscribe to joint states
        self._latest_qpos = None
        self._js_sub = self.create_subscription(
            JointState, "/joint_state", self._js_cb, 10)

        # Collected data
        self.poses_ee_in_base = []  # [(R_3x3, t_3x1)]
        self.poses_cb_in_cam = []   # [(R_3x3, t_3x1)]

    def _js_cb(self, msg: JointState):
        qpos = {}
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_qpos:
                qpos[name] = float(pos)
        if len(qpos) >= len(ALL_JOINTS):
            self._latest_qpos = qpos

    def get_ee_pose(self) -> np.ndarray | None:
        """Get left end-effector pose in world frame (4x4) via MuJoCo FK."""
        if self._latest_qpos is None:
            return None
        for name, adr in self.joint_qpos.items():
            self.data.qpos[adr] = self._latest_qpos.get(name, 0.0)
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.xpos[self.left_ee_body].copy()
        quat = self.data.xquat[self.left_ee_body].copy()  # [w, x, y, z]
        # Convert to rotation matrix via scipy
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def get_camera_image(self):
        """Capture one frame from SPCA2100, return left eye RGB."""
        for _ in range(3):  # skip buffered frames
            self.cap.read()
        ret, frame = self.cap.read()
        if not ret:
            return None
        h, w = frame.shape[:2]
        mid = w // 2
        left = cv2.resize(frame[:, :mid], (640, 480))
        return left

    def capture_one(self) -> bool:
        """Capture one calibration pose: chessboard in camera + joint angles."""
        left = self.get_camera_image()
        if left is None:
            self.get_logger().warn("Camera read failed")
            return False

        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret:
            self.get_logger().warn("Chessboard not detected")
            return False

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
        objp[:, :2] = (np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]]
                       .T.reshape(-1, 2) * SQUARE_SIZE)

        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners, self.K, None)
        R_cb2cam, _ = cv2.Rodrigues(rvec)
        t_cb2cam = tvec.flatten()

        T_ee = self.get_ee_pose()
        if T_ee is None:
            self.get_logger().warn("No joint state received yet")
            return False

        self.poses_cb_in_cam.append((R_cb2cam, t_cb2cam))
        self.poses_ee_in_base.append((T_ee[:3, :3].copy(), T_ee[:3, 3].copy()))
        self.get_logger().info(f"Captured pose {len(self.poses_ee_in_base)}")
        return True

    def solve(self) -> dict | None:
        """Run hand-eye calibration (AX=XB)."""
        n = len(self.poses_ee_in_base)
        if n < 5:
            self.get_logger().error(f"Need >=5 poses, have {n}")
            return None

        R_gripper2base, t_gripper2base = [], []
        R_target2cam, t_target2cam = [], []

        for i in range(n - 1):
            R0, t0 = self.poses_ee_in_base[i]
            R1, t1 = self.poses_ee_in_base[i + 1]
            R_A = R0.T @ R1
            t_A = R0.T @ (t1 - t0)
            R_gripper2base.append(R_A)
            t_gripper2base.append(t_A)

            RB0, tB0 = self.poses_cb_in_cam[i]
            RB1, tB1 = self.poses_cb_in_cam[i + 1]
            R_B = RB0.T @ RB1
            t_B = RB0.T @ (tB1 - tB0)
            R_target2cam.append(R_B)
            t_target2cam.append(t_B)

        methods = [
            ("Tsai", cv2.CALIB_HAND_EYE_TSAI),
            ("Park", cv2.CALIB_HAND_EYE_PARK),
            ("Daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS),
        ]

        results = {}
        for name, method in methods:
            try:
                R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=method)
                T = np.eye(4)
                T[:3, :3] = R_cam2ee
                T[:3, 3] = t_cam2ee.flatten()
                results[name] = T
                self.get_logger().info(
                    f"{name}: R={R_cam2ee.tolist()}, t={t_cam2ee.flatten().tolist()}")
            except Exception as e:
                self.get_logger().error(f"{name} failed: {e}")

        return results

    def verify(self, T_cam2ee):
        """Verify calibration: compute chessboard origin in base frame for all poses."""
        errors = []
        for (R_ee, t_ee), (R_cb, t_cb) in zip(
                self.poses_ee_in_base, self.poses_cb_in_cam):
            p_cam = np.array([0.0, 0.0, 0.0])
            p_cam = R_cb @ p_cam + t_cb
            p_base = R_ee @ (T_cam2ee[:3, :3] @ p_cam + T_cam2ee[:3, 3]) + t_ee
            errors.append(p_base)

        all_p = np.array(errors)
        std = np.std(all_p, axis=0) * 1000
        self.get_logger().info(f"Reprojection std: {std} mm")
        return np.max(std) < 10.0

    def close(self):
        if hasattr(self, 'cap'):
            self.cap.release()


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="Hand-eye calibration AX=XB")
    parser.add_argument("--calib", default="config/stereo_calib.npz",
                        help="Path to stereo calibration .npz")
    parser.add_argument("--output", default="config/hand_eye.yaml",
                        help="Output YAML path")
    parser.add_argument("--poses", type=int, default=15,
                        help="Number of calibration poses to collect")
    args_cli = parser.parse_args()

    rclpy.init(args=None)

    calib_path = args_cli.calib
    if not os.path.isabs(calib_path):
        calib_path = str(Path(__file__).resolve().parent.parent / calib_path)
    if not os.path.exists(calib_path):
        print(f"[hand-eye] Calibration file not found: {calib_path}")
        print("[hand-eye] Run tools/camera_calibrate.py first.")
        return

    node = HandEyeCollector(calib_path)
    print(f"\n{'='*60}")
    print("Hand-Eye Calibration (AX=XB)")
    print(f"{'='*60}")
    print("Place chessboard at a FIXED position on the table.")
    print("Move the robot arm to 15-20 different poses.")
    print("Press ENTER at each pose to capture.")
    print(f"Target: {args_cli.poses} poses\n")

    try:
        captured = 0
        while captured < args_cli.poses and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            input(f"[{captured}/{args_cli.poses}] Press ENTER to capture...")
            if node.capture_one():
                captured += 1

        if captured < 5:
            print("[hand-eye] Not enough poses collected.")
            return

        # Solve
        results = node.solve()
        if results is None:
            return

        # Verify with first method and save
        best_name = list(results.keys())[0]
        best_T = results[best_name]
        if not node.verify(best_T):
            print("[hand-eye] WARNING: Reprojection error > 10mm, consider more poses.")

        # Save
        import yaml
        output_path = args_cli.output
        if not os.path.isabs(output_path):
            output_path = str(Path(__file__).resolve().parent.parent / output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "T_cam2ee": best_T.tolist(),
            "method": best_name,
            "num_poses": captured,
            "all_results": {k: v.tolist() for k, v in results.items()},
            "description": "Camera in End-Effector frame. "
                           "T_robotbase_camera = T_ee2base @ T_cam2ee",
        }
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"\n[hand-eye] Saved to {output_path}")
        print(f"[hand-eye] Method: {best_name}")
        print(f"[hand-eye] T_cam2ee =\n{best_T}")

    except KeyboardInterrupt:
        print("\n[hand-eye] Interrupted.")
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
