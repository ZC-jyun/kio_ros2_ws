#!/usr/bin/env python3
"""Real-time Grounding DINO detection + depth visualization.

Press SPACE to capture & detect, ESC to quit.
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, "src/kio_teleop_openarm/kio_teleop_openarm/lib")
from detector import ObjectDetector
from depth_estimator import StereoDepthEstimator


def main():
    # ── Init detector ─────────────────────────────────────────
    model_path = str(Path.home() / "kio_robot_zzc/models/weights/groundingdino_swint_ogc.pth")
    import groundingdino
    config_path = str(Path(groundingdino.__path__[0]) / "config/GroundingDINO_SwinT_OGC.py")

    print("Loading Grounding DINO...")
    detector = ObjectDetector(model_path, config_path, device="cuda")
    print("Loading depth estimator...")
    depth_est = StereoDepthEstimator("config/stereo_calib.npz")
    K = depth_est.K
    fx, fy = K[0, 0], K[1, 1]
    cx_i, cy_i = K[0, 2], K[1, 2]

    # ── Open camera ────────────────────────────────────────────
    # Find SPCA2100 stereo camera
    camera_idx = None
    for i in range(4):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, frame = cap_test.read()
            if ret and frame is not None and frame.shape[1] == 2560:
                camera_idx = i
                cap_test.release()
                break
            cap_test.release()
    if camera_idx is None:
        print("SPCA2100 stereo camera (2560x720) not found, trying /dev/video2...")
        camera_idx = 2

    print(f"Opening camera /dev/video{camera_idx}")
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nSPACE = detect  |  ESC = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        left = frame[:, :w // 2]
        right = frame[:, w // 2:]
        left = cv2.resize(left, (640, 480))
        right = cv2.resize(right, (640, 480))

        # Show live preview
        preview = left.copy()
        cv2.putText(preview, "Press SPACE to detect", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Live Preview (left eye)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            print("Detecting...")
            t0 = time.time()
            results = detector.detect(left)
            detect_ms = (time.time() - t0) * 1000

            t0 = time.time()
            depth = depth_est.compute_depth(left, right)
            depth_ms = (time.time() - t0) * 1000

            # ── Draw results ────────────────────────────────────
            display = left.copy()
            for r in results:
                x1, y1, x2, y2 = [max(0, int(v)) for v in r["bbox"]]
                x1 = min(x1, 639)
                x2 = min(x2, 639)
                y1 = min(y1, 479)
                y2 = min(y2, 479)

                # Get depth at bbox center
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                roi = depth[max(0, cy-8):min(480, cy+8), max(0, cx-8):min(640, cx+8)]
                roi_valid = roi[roi > 0]
                z = np.median(roi_valid) if len(roi_valid) > 0 else None

                # Color by depth
                color = (0, 255, 0) if z and z < 2.0 else (0, 165, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                label = f"{r['class_name']} {r['confidence']:.2f}"
                if z is not None:
                    x3d = (cx - cx_i) * z / fx
                    y3d = (cy - cy_i) * z / fy
                    label += f" | {z:.2f}m ({x3d:.2f},{y3d:.2f})"
                else:
                    label += " | no depth"

                cv2.putText(display, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Show depth map alongside
            depth_norm = np.clip(depth, 0, 3.0) / 3.0
            depth_norm = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_color[depth <= 0] = 0

            combined = np.hstack([display, depth_color])
            cv2.putText(combined, f"Det: {detect_ms:.0f}ms | Depth: {depth_ms:.0f}ms | {len(results)} objects",
                        (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

            cv2.imshow("Detection + Depth (left=bbox, right=depth map)", combined)
            print(f"  {len(results)} objects, detect={detect_ms:.0f}ms, depth={depth_ms:.0f}ms")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
