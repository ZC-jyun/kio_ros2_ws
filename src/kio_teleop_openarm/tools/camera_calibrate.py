#!/usr/bin/env python3
"""SPCA2100 stereo camera calibration tool.

Usage:
  # Capture calibration image pairs
  python tools/camera_calibrate.py --capture --image-dir ./calib_images

  # Run calibration from captured images
  python tools/camera_calibrate.py --calibrate --image-dir ./calib_images \
      --output ./config/stereo_calib.npz
"""
import cv2
import numpy as np
from pathlib import Path

CHESSBOARD = (7, 4)
SQUARE_SIZE = 0.025  # 25mm
IMAGE_SIZE = (640, 480)


def capture_calibration_images(save_dir, num_pairs=25, device="/dev/video0"):
    """Capture stereo calibration image pairs from SPCA2100 stereo camera."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("[calib] Camera not available at /dev/video0")
        return

    print("[calib] Press SPACE to capture, ESC to exit. Target: 25 pairs.")
    idx = 0
    while idx < num_pairs:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        mid = w // 2
        left = cv2.resize(frame[:, :mid], IMAGE_SIZE)
        right = cv2.resize(frame[:, mid:], IMAGE_SIZE)

        display = np.hstack([left, right])
        cv2.putText(display, f"Captured: {idx}/{num_pairs}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Stereo Calibration - SPACE=save, ESC=quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            cv2.imwrite(str(save_dir / f"left_{idx:03d}.jpg"), left)
            cv2.imwrite(str(save_dir / f"right_{idx:03d}.jpg"), right)
            print(f"[calib] Saved pair {idx + 1}")
            idx += 1
        elif key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    print(f"[calib] Captured {idx} image pairs")


def calibrate_intrinsics(image_dir, chessboard_size, square_size):
    """Calibrate left/right camera intrinsics separately."""
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:chessboard_size[0],
                             0:chessboard_size[1]].T.reshape(-1, 2) * square_size)

    objpoints_l, imgpoints_l = [], []
    objpoints_r, imgpoints_r = [], []

    image_dir = Path(image_dir)
    left_files = sorted(image_dir.glob("left_*.jpg"))
    right_files = sorted(image_dir.glob("right_*.jpg"))

    for lf, rf in zip(left_files, right_files):
        img_l = cv2.imread(str(lf), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)

        ret_l, corners_l = cv2.findChessboardCorners(
            img_l, chessboard_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret_r, corners_r = cv2.findChessboardCorners(
            img_r, chessboard_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret_l and ret_r:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), criteria)
            objpoints_l.append(objp)
            imgpoints_l.append(corners_l)
            objpoints_r.append(objp)
            imgpoints_r.append(corners_r)
        else:
            print(f"[calib] Skipping {lf.name}: chessboard not detected in both eyes")

    print(f"[calib] Valid image pairs: {len(objpoints_l)}/{len(left_files)}")

    ret_l, K_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints_l, imgpoints_l, img_l.shape[::-1], None, None)
    ret_r, K_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints_r, imgpoints_r, img_r.shape[::-1], None, None)

    print(f"[calib] Left  RMS={ret_l:.4f}, K[0,0]={K_l[0,0]:.1f}")
    print(f"[calib] Right RMS={ret_r:.4f}, K[0,0]={K_r[0,0]:.1f}")

    return {
        "K_left": K_l, "dist_left": dist_l,
        "K_right": K_r, "dist_right": dist_r,
        "image_size": img_l.shape[::-1],
        "objpoints_l": objpoints_l, "imgpoints_l": imgpoints_l,
        "objpoints_r": objpoints_r, "imgpoints_r": imgpoints_r,
    }


def calibrate_stereo(intrinsics):
    """Stereo calibration: compute R, T, Q from paired intrinsics."""
    K_l = intrinsics["K_left"]
    dist_l = intrinsics["dist_left"]
    K_r = intrinsics["K_right"]
    dist_r = intrinsics["dist_right"]
    img_size = intrinsics["image_size"]

    ret, K_l_new, dist_l_new, K_r_new, dist_r_new, R, T, E, F = \
        cv2.stereoCalibrate(
            intrinsics["objpoints_l"], intrinsics["imgpoints_l"],
            intrinsics["imgpoints_r"],
            K_l, dist_l, K_r, dist_r, img_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l_new, dist_l_new, K_r_new, dist_r_new,
        img_size, R, T, alpha=0)

    baseline = np.linalg.norm(T)
    print(f"[calib] Stereo RMS={ret:.4f}, baseline={baseline:.4f}m")
    print(f"[calib] R=\n{R}")
    print(f"[calib] T={T.flatten()}")

    return {
        "K_left": K_l_new, "dist_left": dist_l_new,
        "K_right": K_r_new, "dist_right": dist_r_new,
        "R": R, "T": T, "Q": Q,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2,
    }


def save_calib(calib_data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path,
             K_left=calib_data["K_left"],
             dist_left=calib_data["dist_left"],
             K_right=calib_data["K_right"],
             dist_right=calib_data["dist_right"],
             R=calib_data["R"], T=calib_data["T"], Q=calib_data["Q"],
             R1=calib_data["R1"], R2=calib_data["R2"],
             P1=calib_data["P1"], P2=calib_data["P2"])
    print(f"[calib] Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SPCA2100 stereo calibration")
    parser.add_argument("--capture", action="store_true", help="Capture calibration images")
    parser.add_argument("--calibrate", action="store_true", help="Compute calibration params")
    parser.add_argument("--image-dir", default="./calib_images",
                        help="Directory for calibration images")
    parser.add_argument("--output", default="./config/stereo_calib.npz",
                        help="Output .npz path")
    parser.add_argument("--device", default="/dev/video2",
                        help="Video device (default: /dev/video2 for SPCA2100)")
    parser.add_argument("--square-size", type=float, default=0.012,
                        help="Chessboard square size in meters (default: 0.012 = 1.2cm)")
    args = parser.parse_args()

    if args.capture:
        capture_calibration_images(args.image_dir, device=args.device)

    if args.calibrate:
        intrinsics = calibrate_intrinsics(args.image_dir, CHESSBOARD, args.square_size)
        stereo = calibrate_stereo(intrinsics)
        save_calib(stereo, args.output)
