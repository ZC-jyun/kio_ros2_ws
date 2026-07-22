#!/usr/bin/env python3
"""perception_node — object detection + stereo depth estimation.

Subscribes to /camera/left_image and /camera/right_image.
Provides /perception/detect service for on-demand detection + depth.
Publishes /perception/detections and /perception/depth.
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np

from kio_teleop_openarm.lib.detector import ObjectDetector
from kio_teleop_openarm.lib.depth_estimator import StereoDepthEstimator


def _imgmsg_to_numpy(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image to numpy array without cv_bridge."""
    if msg.encoding == "rgb8":
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
    elif msg.encoding == "bgr8":
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        img = img[:, :, ::-1].copy()  # BGR → RGB
    elif msg.encoding == "mono8" or msg.encoding == "8UC1":
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width)
    elif msg.encoding == "32FC1":
        img = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.width)
    else:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
    return img


def _numpy_to_imgmsg(arr: np.ndarray, encoding="rgb8") -> Image:
    """Convert numpy array to sensor_msgs/Image without cv_bridge."""
    msg = Image()
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = False
    if arr.ndim == 3:
        msg.step = arr.shape[1] * arr.shape[2]
    elif encoding == "32FC1":
        msg.step = arr.shape[1] * 4
    else:
        msg.step = arr.shape[1]
    msg.data = np.ascontiguousarray(arr).tobytes()
    return msg


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        from pathlib import Path as _Path
        import groundingdino as _gdinopkg

        _default_model = str(_Path.home() / "kio_robot_zzc/models/weights/groundingdino_swint_ogc.pth")
        _default_config = str(_Path(_gdinopkg.__path__[0]) / "config/GroundingDINO_SwinT_OGC.py")

        # Look up calibration file in both source and install locations
        _candidates = [
            _Path(__file__).parents[2] / "config" / "stereo_calib.npz",
            _Path.home() / "kio_robot_zzc" / "kio_ros2_ws" / "config" / "stereo_calib.npz",
        ]
        _default_calib = ""
        for _c in _candidates:
            if _c.exists():
                _default_calib = str(_c)
                break

        self.declare_parameter("model_path", _default_model)
        self.declare_parameter("config_path", _default_config)
        self.declare_parameter("calib_path", _default_calib)
        self.declare_parameter("device", "cuda")

        model_path = self.get_parameter("model_path").value
        config_path = self.get_parameter("config_path").value
        calib_path = self.get_parameter("calib_path").value
        device = self.get_parameter("device").value

        self._left_img = None
        self._right_img = None

        # Detector (Grounding DINO)
        self._detector = None
        if model_path and config_path:
            try:
                self._detector = ObjectDetector(model_path, config_path, device)
                self.get_logger().info(f"Grounding DINO loaded on {device}")
            except Exception as e:
                self.get_logger().warn(f"Grounding DINO not available: {e}")

        # Depth estimator (SGBM)
        self._depth_est = None
        if calib_path:
            try:
                self._depth_est = StereoDepthEstimator(calib_path)
                self.get_logger().info(f"Stereo calibration loaded from {calib_path}")
            except Exception as e:
                self.get_logger().warn(f"Stereo depth not available: {e}")

        # Subscribers
        self._left_sub = self.create_subscription(
            Image, "/camera/left_image", self._left_cb, 10)
        self._right_sub = self.create_subscription(
            Image, "/camera/right_image", self._right_cb, 10)

        # Publishers
        try:
            from kio_teleop_openarm.msg import Detection, DetectionArray
            self._Detection = Detection
            self._DetectionArray = DetectionArray
            self._det_pub = self.create_publisher(
                DetectionArray, "/perception/detections", 10)
        except ImportError:
            self.get_logger().warn("Detection msg types not available yet")
            self._Detection = None
            self._DetectionArray = None
            self._det_pub = None

        self._depth_pub = self.create_publisher(Image, "/perception/depth", 10)

        # Service
        try:
            from kio_teleop_openarm.srv import DetectObjects
            self._detect_srv = self.create_service(
                DetectObjects, "/perception/detect", self._detect_cb)
            self.get_logger().info("/perception/detect service ready")
        except ImportError:
            self.get_logger().warn(
                "DetectObjects srv not available; rebuild after colcon build")

        self.get_logger().info("perception_node started")

    def _left_cb(self, msg: Image):
        self._left_img = _imgmsg_to_numpy(msg)

    def _right_cb(self, msg: Image):
        self._right_img = _imgmsg_to_numpy(msg)

    def _detect_cb(self, request, response):
        """Service callback: run detection + depth on current camera frame."""
        if self._left_img is None or self._right_img is None:
            self.get_logger().warn("No camera images received yet")
            response.success = False
            return response

        left = self._left_img.copy()
        right = self._right_img.copy()

        # 1. Detection
        text_prompt = request.text_prompt or ""
        if self._detector is not None:
            try:
                dets = self._detector.detect(
                    left,
                    caption=text_prompt,
                    box_threshold=(request.box_threshold
                                   if request.box_threshold > 0 else 0.25),
                    text_threshold=(request.text_threshold
                                    if request.text_threshold > 0 else 0.20),
                )
            except Exception as e:
                self.get_logger().error(f"Detection failed: {e}")
                dets = []
        else:
            dets = []

        if self._DetectionArray is not None and self._Detection is not None:
            det_array = self._DetectionArray()
            for d in dets:
                det_msg = self._Detection()
                det_msg.class_name = d["class_name"]
                det_msg.confidence = float(d["confidence"])
                det_msg.bbox = [float(v) for v in d["bbox"]]
                det_array.detections.append(det_msg)
            response.detections = det_array
            if self._det_pub is not None:
                self._det_pub.publish(det_array)

        # 2. Depth
        if self._depth_est is not None and self._depth_est.ready:
            try:
                depth = self._depth_est.compute_depth(left, right)
            except Exception as e:
                self.get_logger().error(f"Depth estimation failed: {e}")
                depth = np.zeros(left.shape[:2], dtype=np.float32)
        else:
            depth = np.zeros(left.shape[:2], dtype=np.float32)

        stamp = self.get_clock().now().to_msg()
        depth_msg = _numpy_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
        depth_msg.header = Header(stamp=stamp, frame_id="camera_left")
        response.depth = depth_msg
        self._depth_pub.publish(depth_msg)

        response.success = len(dets) > 0
        self.get_logger().info(
            f"Perception: {len(dets)} detections, depth ready")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
