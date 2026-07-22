"""Stereo rendering utilities extracted from kio_teleop_upoo_mujoco."""

import numpy as np
import mujoco


def make_stereo_cameras(model, scene, cam_lookat, cam_distance, cam_azimuth, cam_elevation,
                         width=640, height=480, ipd=0.064):
    """Create left/right camera and render context pairs for stereo."""
    cam_left = mujoco.MjvCamera()
    cam_right = mujoco.MjvCamera()
    for cam in (cam_left, cam_right):
        cam.lookat[:] = cam_lookat
        cam.distance = cam_distance
        cam.azimuth = cam_azimuth
        cam.elevation = cam_elevation
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    forward = np.array([
        np.cos(np.radians(cam_elevation)) * np.sin(np.radians(cam_azimuth)),
        np.cos(np.radians(cam_elevation)) * np.cos(np.radians(cam_azimuth)),
        np.sin(np.radians(cam_elevation)),
    ])
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right_vec = np.cross(forward, np.array([0, 0, 1]))
    right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-8)
    cam_left.lookat[:] = cam_lookat - right_vec * (ipd / 2)
    cam_right.lookat[:] = cam_lookat + right_vec * (ipd / 2)

    r_left = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    r_right = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    vp = mujoco.MjrRect(0, 0, width, height)
    return cam_left, cam_right, r_left, r_right, vp


def render_stereo(model, data, scene, cam_left, cam_right, r_left, r_right, vp):
    """Render left+right eye views and return as uint8 RGB arrays (flipped vertically)."""
    opt = mujoco.MjvOption()
    mujoco.mjv_updateScene(model, data, opt, None, cam_left, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(vp, scene, r_left)
    mujoco.mjv_updateScene(model, data, opt, None, cam_right, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(vp, scene, r_right)
    left_rgb = np.empty((vp.height, vp.width, 3), dtype=np.uint8)
    right_rgb = np.empty((vp.height, vp.width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(left_rgb, None, vp, r_left)
    mujoco.mjr_readPixels(right_rgb, None, vp, r_right)
    return left_rgb[::-1, :], right_rgb[::-1, :]


def set_camera_free_pose(cam, position, lookat):
    """Set MuJoCo free camera from world position/lookat."""
    dir_vec = lookat - position
    dist = float(np.linalg.norm(dir_vec))
    if dist < 1e-6:
        return
    d = dir_vec / dist
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.elevation = float(np.degrees(np.arcsin(d[2])))
    cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
