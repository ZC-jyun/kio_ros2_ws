#!/usr/bin/env python3
"""vr_bridge node — VR data acquisition, publishes poses/landmarks/pinch, pushes stereo images to headset."""

import os
import sys
import time
import numpy as np
from pathlib import Path
from multiprocessing import Event, Queue, shared_memory

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Image
from pytransform3d import rotations

# Path: teleop_deploy & television modules
_DEPLOY_DIR = Path("/home/kiorobot/kio_robot_zzc/openarm-main/teleop_deploy")
_TELEVISION_DIR = _DEPLOY_DIR / "television"
for _p in (str(_DEPLOY_DIR), str(_TELEVISION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from TeleVision import OpenTeleVision
from constants_vuer import grd_yup2grd_zup
from motion_utils import mat_update, fast_mat_inv


def safe_get_landmarks(tv, side: str):
    candidates = [f"{side}_landmarks", f"{side}_hand_landmarks", f"{side}HandLandmarks"]
    for name in candidates:
        if hasattr(tv, name):
            try:
                arr = np.asarray(getattr(tv, name), dtype=np.float32).reshape(-1, 3)
                if arr.shape[0] >= 10 and np.isfinite(arr).all() and np.any(arr != 0):
                    return arr
            except Exception:
                pass
    return None


class AbsoluteVuerPreprocessor:
    def __init__(self):
        self.vuer_head_mat = np.eye(4, dtype=np.float32)
        self.vuer_right_wrist_mat = np.eye(4, dtype=np.float32)
        self.vuer_left_wrist_mat = np.eye(4, dtype=np.float32)

    def process(self, tv):
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
        self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())
        t_vuer_head = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        t_vuer_right = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        t_vuer_left = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        return (t_vuer_head.astype(np.float32), t_vuer_left.astype(np.float32),
                t_vuer_right.astype(np.float32))


def mat4_to_pose_stamped(mat4, frame_id="vuer"):
    """Convert 4x4 transform to PoseStamped."""
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()
    msg.pose.position = Point(x=float(mat4[0, 3]), y=float(mat4[1, 3]), z=float(mat4[2, 3]))
    quat = rotations.quaternion_from_matrix(mat4[:3, :3].astype(np.float64))
    msg.pose.orientation = Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])
    return msg


class VrBridgeNode(Node):
    def __init__(self):
        super().__init__("vr_bridge")

        # Declare parameters
        self.declare_parameter("resolution", [960, 1280])
        self.declare_parameter("ngrok", False)
        self.declare_parameter("cert_file", "./cert.pem")
        self.declare_parameter("key_file", "./key.pem")
        self.declare_parameter("diag_interval", 5.0)

        resolution = self.get_parameter("resolution").value
        ngrok = self.get_parameter("ngrok").value
        cert_file = self.get_parameter("cert_file").value
        key_file = self.get_parameter("key_file").value
        self.diag_interval = self.get_parameter("diag_interval").value

        resolution = tuple(resolution)
        self.resolution_cropped = resolution
        img_shape = (resolution[0], 2 * resolution[1], 3)
        self.shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(img_shape)) * np.uint8().itemsize)
        self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
        self.image_queue = Queue()
        toggle_streaming = Event()

        self.tv = OpenTeleVision(
            self.resolution_cropped, self.shm.name, self.image_queue,
            toggle_streaming, ngrok=ngrok, cert_file=cert_file, key_file=key_file)
        self.processor = AbsoluteVuerPreprocessor()

        # Publishers
        self.head_pub = self.create_publisher(PoseStamped, "/vr/head_pose", 10)
        self.left_hand_pub = self.create_publisher(PoseStamped, "/vr/left_hand_pose", 10)
        self.right_hand_pub = self.create_publisher(PoseStamped, "/vr/right_hand_pose", 10)
        self.left_lm_pub = self.create_publisher(Float32MultiArray, "/vr/landmarks_left", 10)
        self.right_lm_pub = self.create_publisher(Float32MultiArray, "/vr/landmarks_right", 10)
        self.left_pinch_pub = self.create_publisher(Float32, "/vr/left_pinch", 10)
        self.right_pinch_pub = self.create_publisher(Float32, "/vr/right_pinch", 10)

        # Subscriber: stereo image from simulator
        self.stereo_sub = self.create_subscription(
            Image, "/stereo_image", self._stereo_cb, 10)

        self._last_diag_ts = 0.0
        self._step_diag_last = 0.0

        # Timer: publish VR data at ~60 Hz
        self._timer = self.create_timer(1.0 / 60.0, self._step)
        self.get_logger().info("vr_bridge started")

    def _step(self):
        t_vuer_head, t_vuer_left, t_vuer_right = self.processor.process(self.tv)

        now = time.time()
        if now - self._step_diag_last > self.diag_interval:
            self._step_diag_last = now
            raw_head = self.tv.head_matrix
            raw_left = self.tv.left_hand
            raw_right = self.tv.right_hand
            self.get_logger().info(
                f"raw_head max={raw_head.max():.4f} "
                f"raw_left max={raw_left.max():.4f} raw_right max={raw_right.max():.4f} "
                f"t_head pos=({t_vuer_head[0,3]:.3f},{t_vuer_head[1,3]:.3f},{t_vuer_head[2,3]:.3f}) "
                f"pinch L={self.tv.left_pinch:.2f} R={self.tv.right_pinch:.2f}")

        # Publish poses
        stamp = self.get_clock().now().to_msg()
        for pub, mat4, fid in [
            (self.head_pub, t_vuer_head, "vuer_head"),
            (self.left_hand_pub, t_vuer_left, "vuer_left_hand"),
            (self.right_hand_pub, t_vuer_right, "vuer_right_hand"),
        ]:
            msg = mat4_to_pose_stamped(mat4, fid)
            msg.header.stamp = stamp
            pub.publish(msg)

        # Publish landmarks
        for pub, side in [(self.left_lm_pub, "left"), (self.right_lm_pub, "right")]:
            lm = safe_get_landmarks(self.tv, side)
            if lm is not None:
                msg = Float32MultiArray()
                msg.data = lm.flatten().tolist()
                pub.publish(msg)

        # Publish pinch
        self.left_pinch_pub.publish(Float32(data=float(self.tv.left_pinch)))
        self.right_pinch_pub.publish(Float32(data=float(self.tv.right_pinch)))

    def _stereo_cb(self, msg: Image):
        """Receive stereo image from simulator and write to VR shared memory."""
        try:
            h, w = msg.height, msg.width
            channels = 3  # rgb8
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, channels)
            if img.shape == self.img_array.shape:
                self.img_array[:] = img
        except Exception as e:
            self.get_logger().warn(f"stereo_cb error: {e}")

    def shutdown(self):
        try:
            self.shm.close()
            self.shm.unlink()
        except FileNotFoundError:
            pass
        if hasattr(self.tv, 'process') and self.tv.process is not None:
            try:
                self.tv.process.kill()
                self.tv.process.join(timeout=2.0)
            except Exception:
                pass
        if hasattr(self.tv, 'webrtc_process') and self.tv.webrtc_process is not None:
            try:
                self.tv.webrtc_process.kill()
                self.tv.webrtc_process.join(timeout=2.0)
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = VrBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
