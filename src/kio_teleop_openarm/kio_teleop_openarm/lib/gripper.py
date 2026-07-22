"""Gripper control logic extracted from kio_teleop_upoo_mujoco."""

import numpy as np


def normalized_pinch_metric(landmarks, thumb_tip_index=4, index_tip_index=9):
    """Compute normalized pinch distance from hand landmarks."""
    if landmarks is None:
        return np.nan
    lm = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)
    n = lm.shape[0]
    if n <= max(thumb_tip_index, index_tip_index):
        return np.nan
    thumb, index = lm[thumb_tip_index], lm[index_tip_index]
    pinch_dist = float(np.linalg.norm(thumb - index))
    palm_candidates = []
    for a, b in [(0, 10), (5, 20), (0, 5), (0, 17), (5, 17)]:
        if n > max(a, b):
            d = float(np.linalg.norm(lm[a] - lm[b]))
            if np.isfinite(d) and d > 1e-5:
                palm_candidates.append(d)
    palm = max(palm_candidates) if palm_candidates else 1.0
    return pinch_dist / max(palm, 1e-5)


def gripper_command_from_landmarks(
    landmarks,
    side,
    prev_metrics,
    enable_gripper,
    gripper_open_value,
    gripper_close_value,
    gripper_close_threshold,
    gripper_open_threshold,
    gripper_smoothing,
    thumb_tip_index=4,
    index_tip_index=9,
):
    """Compute gripper command from hand landmarks with exponential smoothing."""
    if not enable_gripper:
        return gripper_open_value, prev_metrics

    metric = normalized_pinch_metric(
        landmarks, thumb_tip_index=thumb_tip_index, index_tip_index=index_tip_index)
    attr = f'gripper_metric_{side}'
    prev_metric = prev_metrics.get(attr, None)
    if prev_metric is None:
        prev_metric = metric
    if not np.isfinite(metric):
        metric = prev_metric
    smoothed = (1.0 - gripper_smoothing) * prev_metric + gripper_smoothing * metric
    prev_metrics[attr] = smoothed

    denom = max(gripper_open_threshold - gripper_close_threshold, 1e-6)
    alpha_open = np.clip((smoothed - gripper_close_threshold) / denom, 0.0, 1.0)
    raw = alpha_open * gripper_open_value + (1.0 - alpha_open) * gripper_close_value
    return float(raw), prev_metrics
