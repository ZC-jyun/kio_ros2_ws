#!/usr/bin/env python3
"""camera_node — SPCA2100 stereo UVC camera driver.

Publishes left/right stereo images and combined side-by-side image.
No cv_bridge dependency — constructs sensor_msgs/Image from numpy directly.
"""
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
import numpy as np


def _cv2_to_imgmsg(cv_image, encoding="bgr8"):
    """Convert OpenCV/numpy image to sensor_msgs/Image without cv_bridge."""
    if encoding == "bgr8":
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        enc = "rgb8"
    else:
        img = cv_image
        enc = encoding
    msg = Image()
    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.encoding = enc
    msg.is_bigendian = False
    msg.step = img.shape[1] * img.shape[2] if img.ndim == 3 else img.shape[1]
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("capture_width", 2560)
        self.declare_parameter("capture_height", 640)
        self.declare_parameter("eye_width", 640)
        self.declare_parameter("eye_height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("publish_rate", 15.0)

        device = self.get_parameter("device").value
        capture_width = self.get_parameter("capture_width").value
        capture_height = self.get_parameter("capture_height").value
        self.eye_width = self.get_parameter("eye_width").value
        self.eye_height = self.get_parameter("eye_height").value
        fps = self.get_parameter("fps").value
        publish_rate = self.get_parameter("publish_rate").value

        self._lock = threading.Lock()
        self._latest_frame = None
        self._stop = threading.Event()

        self.left_pub = self.create_publisher(Image, "/camera/left_image", 10)
        self.right_pub = self.create_publisher(Image, "/camera/right_image", 10)
        self.stereo_pub = self.create_publisher(Image, "/camera/stereo_image", 10)

        try:
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                self.get_logger().error(f"Failed to open {device}")
                raise RuntimeError(f"Cannot open camera device: {device}")
            self.get_logger().info(
                f"Camera opened: {device} {capture_width}x{capture_height} @ {fps}fps")
        except Exception as e:
            self.get_logger().fatal(f"Camera init failed: {e}")
            raise

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        self._timer = self.create_timer(1.0 / publish_rate, self._publish)
        self.get_logger().info("camera_node started")

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame
            else:
                import time
                time.sleep(0.005)

    def _publish(self):
        with self._lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()

        h, w = frame.shape[:2]
        mid = w // 2
        left_raw = frame[:, :mid]
        right_raw = frame[:, mid:]

        left = cv2.resize(left_raw, (self.eye_width, self.eye_height))
        right = cv2.resize(right_raw, (self.eye_width, self.eye_height))

        stamp = self.get_clock().now().to_msg()

        left_msg = _cv2_to_imgmsg(left, encoding="bgr8")
        left_msg.header = Header(stamp=stamp, frame_id="camera_left")
        self.left_pub.publish(left_msg)

        right_msg = _cv2_to_imgmsg(right, encoding="bgr8")
        right_msg.header = Header(stamp=stamp, frame_id="camera_right")
        self.right_pub.publish(right_msg)

        stereo = np.hstack([left, right])
        stereo_msg = _cv2_to_imgmsg(stereo, encoding="bgr8")
        stereo_msg.header = Header(stamp=stamp, frame_id="camera_stereo")
        self.stereo_pub.publish(stereo_msg)

    def stop(self):
        self._stop.set()
        if hasattr(self, '_thread') and self._thread is not None:
            self._thread.join(timeout=1.0)
        if hasattr(self, 'cap'):
            self.cap.release()

    def destroy_node(self):
        self.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
