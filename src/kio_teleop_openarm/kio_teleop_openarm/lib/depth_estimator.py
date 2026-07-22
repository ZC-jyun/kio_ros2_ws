"""SGBM stereo depth estimation — OpenCV-based, no ROS2 dependency."""
import cv2
import numpy as np


class StereoDepthEstimator:
    def __init__(self, calib_path: str = None):
        if calib_path is not None:
            self._load_calib(calib_path)
            self._ready = True
        else:
            self._ready = False

    def _load_calib(self, calib_path):
        calib = np.load(calib_path)
        self.K = calib["K_left"]
        self.Q = calib["Q"]
        self.R1 = calib["R1"]
        self.R2 = calib["R2"]
        self.P1 = calib["P1"]
        self.P2 = calib["P2"]
        self.dist_left = calib["dist_left"]
        self.dist_right = calib["dist_right"]
        self._rect_maps_built = False

    def _build_rect_maps(self, width, height):
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_left, self.R1, self.P1,
            (width, height), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K, self.dist_right, self.R2, self.P2,  # K_right ≈ K_left after stereoRectify
            (width, height), cv2.CV_32FC1)
        self._rect_maps_built = True

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=11,
            P1=8 * 3 * 11 ** 2,
            P2=32 * 3 * 11 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
        )

    @property
    def ready(self):
        return self._ready

    def compute_depth(self, left_img: np.ndarray,
                      right_img: np.ndarray) -> np.ndarray:
        """Compute depth map from rectified stereo images.

        Args:
            left_img:  (H, W) grayscale or (H, W, 3) RGB.
            right_img: (H, W) grayscale or (H, W, 3) RGB.

        Returns:
            depth_map: (H, W) float32, depth in meters. 0.0 = invalid.
        """
        if not self._ready:
            raise RuntimeError("StereoDepthEstimator not loaded with calibration")

        h, w = left_img.shape[:2]
        if not self._rect_maps_built or self.map1x.shape[:2] != (h, w):
            self._build_rect_maps(w, h)

        if left_img.ndim == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_img
        if right_img.ndim == 3:
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        else:
            right_gray = right_img

        left_rect = cv2.remap(left_gray, self.map1x, self.map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, self.map2x, self.map2y, cv2.INTER_LINEAR)

        disp = self.stereo.compute(left_rect, right_rect).astype(np.float32) / 16.0
        points_3d = cv2.reprojectImageTo3D(disp, self.Q)
        depth = points_3d[:, :, 2].astype(np.float32)
        depth[disp <= 0] = 0.0
        return depth
