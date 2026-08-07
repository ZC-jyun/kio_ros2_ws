#!/usr/bin/env python3
"""Replay VR grasp episodes and add a wrist RGB stream in a new dataset.

The original episodes contain robot qpos/qvel and actions but not the cup pose.
The cup starts upright at a known height, so this tool finds its initial x/y by
matching a low-resolution re-render of the recorded vr_center first frame.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from dm_control import mujoco


WORKSPACE = Path(__file__).resolve().parents[1]
TOOLS_DIR = WORKSPACE / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from vr_left_grasp_scene import CUP_X_RANGE, CUP_Y_RANGE, scene_xml


CAMERA_NAMES = ("vr_center", "wrist")
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
GRIPPER_OPEN = 0.044
CUP_INITIAL_Z = 0.43
CUP_INITIAL_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
CONTROL_TIMESTEP = 0.02

LEFT_ARM_JOINTS = (
    "upoo_left_J01", "upoo_left_J02", "upoo_left_J03",
    "upoo_left_J04", "upoo_left_J05", "upoo_left_J06",
)
RIGHT_ARM_JOINTS = (
    "upoo_right_J01", "upoo_right_J02", "upoo_right_J03",
    "upoo_right_J04", "upoo_right_J05", "upoo_right_J06",
)
LEFT_FINGERS = (
    "upoo_left_openarm_v1_finger_joint1",
    "upoo_left_openarm_v1_finger_joint2",
)
RIGHT_FINGERS = (
    "upoo_right_openarm_v1_finger_joint1",
    "upoo_right_openarm_v1_finger_joint2",
)


@dataclass(frozen=True)
class InitialPoseMatch:
    x: float
    y: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paired vr_center+wrist RGB episodes by MuJoCo replay."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=WORKSPACE / "data" / "sim_upoo_left_arm_grasp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "data" / "sim_upoo_left_arm_grasp_wrist",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        help="Episode indices to process. Defaults to every source episode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output episode files.",
    )
    parser.add_argument(
        "--coarse-step", type=float, default=0.01,
        help="Initial cup x/y search spacing in metres.",
    )
    parser.add_argument(
        "--refine-step", type=float, default=0.002,
        help="Second-pass cup x/y search spacing in metres.",
    )
    parser.add_argument(
        "--validation-stride", type=int, default=10,
        help="Compare every Nth reconstructed center frame with recorded RGB.",
    )
    parser.add_argument(
        "--max-center-mae", type=float, default=0.12,
        help="Reject an episode when normalized center-RGB MAE exceeds this value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and print intended work without rendering or writing output.",
    )
    args = parser.parse_args()
    if args.coarse_step <= 0 or args.refine_step <= 0:
        parser.error("search steps must be positive")
    if args.validation_stride <= 0:
        parser.error("--validation-stride must be positive")
    if not 0 <= args.max_center_mae <= 1:
        parser.error("--max-center-mae must be in [0, 1]")
    return args


def episode_paths(source_dir: Path, selected: list[int] | None) -> list[Path]:
    if selected is None:
        paths = sorted(source_dir.glob("episode_*.hdf5"))
    else:
        paths = [source_dir / f"episode_{index}.hdf5" for index in selected]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source episode(s): " + ", ".join(missing))
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files in {source_dir}")
    return paths


def set_robot_state(physics: mujoco.Physics, qpos: np.ndarray, qvel: np.ndarray) -> None:
    for index, name in enumerate(LEFT_ARM_JOINTS):
        physics.named.data.qpos[name] = qpos[index]
        physics.named.data.qvel[name] = qvel[index]
    for index, name in enumerate(RIGHT_ARM_JOINTS):
        physics.named.data.qpos[name] = qpos[7 + index]
        physics.named.data.qvel[name] = qvel[7 + index]

    left_position = qpos[6] * GRIPPER_OPEN
    left_velocity = qvel[6] * GRIPPER_OPEN
    right_position = qpos[13] * GRIPPER_OPEN
    right_velocity = qvel[13] * GRIPPER_OPEN
    for name in LEFT_FINGERS:
        physics.named.data.qpos[name] = left_position
        physics.named.data.qvel[name] = left_velocity
    for name in RIGHT_FINGERS:
        physics.named.data.qpos[name] = right_position
        physics.named.data.qvel[name] = right_velocity


def set_controls(physics: mujoco.Physics, action: np.ndarray) -> None:
    for index, name in enumerate(LEFT_ARM_JOINTS):
        physics.named.data.ctrl[f"{name}_ctrl"] = action[index]
    for index, name in enumerate(RIGHT_ARM_JOINTS):
        physics.named.data.ctrl[f"{name}_ctrl"] = action[7 + index]
    physics.named.data.ctrl["upoo_left_gripper_ctrl"] = action[6] * GRIPPER_OPEN
    physics.named.data.ctrl["upoo_right_gripper_ctrl"] = action[13] * GRIPPER_OPEN


def cup_qpos_address(physics: mujoco.Physics) -> int:
    cup_body_id = physics.model.name2id("cup", "body")
    cup_joint_id = int(physics.model.body_jntadr[cup_body_id])
    return int(physics.model.jnt_qposadr[cup_joint_id])


def set_initial_cup_pose(physics: mujoco.Physics, address: int, x: float, y: float) -> None:
    physics.data.qpos[address:address + 7] = np.array(
        [x, y, CUP_INITIAL_Z, *CUP_INITIAL_QUAT], dtype=np.float64
    )
    physics.data.qvel[address:address + 6] = 0.0


def normalized_mae(image: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(image.astype(np.float32) - reference.astype(np.float32)).mean() / 255.0)


def candidate_values(lower: float, upper: float, step: float) -> np.ndarray:
    count = int(np.floor((upper - lower) / step))
    values = lower + np.arange(count + 1, dtype=np.float64) * step
    return np.append(values, upper) if values[-1] < upper else values


def match_initial_cup_pose(
    physics: mujoco.Physics,
    cup_address: int,
    first_qpos: np.ndarray,
    first_qvel: np.ndarray,
    reference: np.ndarray,
    coarse_step: float,
    refine_step: float,
) -> InitialPoseMatch:
    set_robot_state(physics, first_qpos, first_qvel)
    target = reference[::4, ::4]

    def search(xs: np.ndarray, ys: np.ndarray, best: InitialPoseMatch | None = None) -> InitialPoseMatch:
        candidate = best
        for x in xs:
            for y in ys:
                set_initial_cup_pose(physics, cup_address, float(x), float(y))
                physics.forward()
                rendered = physics.render(height=target.shape[0], width=target.shape[1], camera_id="vr_center")
                score = normalized_mae(rendered, target)
                if candidate is None or score < candidate.score:
                    candidate = InitialPoseMatch(float(x), float(y), score)
        assert candidate is not None
        return candidate

    coarse = search(
        candidate_values(*CUP_X_RANGE, coarse_step),
        candidate_values(*CUP_Y_RANGE, coarse_step),
    )
    refine_x = candidate_values(
        max(CUP_X_RANGE[0], coarse.x - coarse_step),
        min(CUP_X_RANGE[1], coarse.x + coarse_step),
        refine_step,
    )
    refine_y = candidate_values(
        max(CUP_Y_RANGE[0], coarse.y - coarse_step),
        min(CUP_Y_RANGE[1], coarse.y + coarse_step),
        refine_step,
    )
    return search(refine_x, refine_y)


def replay_episode(
    source: Path,
    destination: Path,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    with h5py.File(source, "r") as root:
        valid_length = int(root.attrs["valid_length"])
        qpos = root["/observations/qpos"][:valid_length].astype(np.float64)
        qvel = root["/observations/qvel"][:valid_length].astype(np.float64)
        actions = root["/action"][:valid_length].astype(np.float64)
        center = root["/observations/images/vr_center"][:valid_length]

    physics = mujoco.Physics.from_xml_string(scene_xml(float(np.mean(CUP_X_RANGE)), float(np.mean(CUP_Y_RANGE))))
    cup_address = cup_qpos_address(physics)
    match = match_initial_cup_pose(
        physics, cup_address, qpos[0], qvel[0], center[0], args.coarse_step, args.refine_step
    )
    set_initial_cup_pose(physics, cup_address, match.x, match.y)
    physics.forward()

    wrist_frames: list[np.ndarray] = []
    center_errors: list[float] = []
    substeps = max(1, round(CONTROL_TIMESTEP / float(physics.model.opt.timestep)))
    for timestep in range(valid_length):
        # Keep the camera carrier at the recorded robot pose. The cup remains
        # dynamically replayed between samples, so center RGB can validate it.
        set_robot_state(physics, qpos[timestep], qvel[timestep])
        physics.forward()
        wrist_frames.append(physics.render(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, camera_id="wrist"))
        if timestep % args.validation_stride == 0 or timestep == valid_length - 1:
            replay_center = physics.render(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, camera_id="vr_center")
            center_errors.append(normalized_mae(replay_center, center[timestep]))
        set_controls(physics, actions[timestep])
        for _ in range(substeps):
            physics.step()

    center_mae = float(np.mean(center_errors))
    if center_mae > args.max_center_mae:
        return False, (
            f"rejected: center MAE={center_mae:.4f} exceeds {args.max_center_mae:.4f}; "
            f"initial cup=({match.x:.4f}, {match.y:.4f}), fit={match.score:.4f}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with h5py.File(destination, "r+") as root:
        images = root["/observations/images"]
        if "wrist" in images:
            del images["wrist"]
        images.create_dataset(
            "wrist",
            data=np.asarray(wrist_frames, dtype=np.uint8),
            chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            compression="lzf",
        )
        root.attrs["wrist_augmentation"] = "kinematic_robot_replay_v1"
        root.attrs["replay_initial_cup_pose"] = np.array(
            [match.x, match.y, CUP_INITIAL_Z, *CUP_INITIAL_QUAT], dtype=np.float32
        )
        root.attrs["replay_center_mae"] = center_mae
        root.attrs["replay_initial_center_mae"] = match.score
    return True, (
        f"written: center MAE={center_mae:.4f}, initial cup=({match.x:.4f}, {match.y:.4f}), "
        f"fit={match.score:.4f}"
    )


def main() -> int:
    args = parse_args()
    paths = episode_paths(args.source_dir, args.episodes)
    if args.dry_run:
        print(f"[dry-run] {len(paths)} episode(s): {args.source_dir} -> {args.output_dir}")
        for source in paths:
            print(f"[dry-run] {source.name}")
        return 0

    if args.output_dir.resolve() == args.source_dir.resolve():
        raise ValueError("--output-dir must be different from --source-dir")

    successes = 0
    for source in paths:
        destination = args.output_dir / source.name
        if destination.exists() and not args.overwrite:
            print(f"[skip] {source.name}: output exists (use --overwrite to replace)")
            continue
        ok, message = replay_episode(source, destination, args)
        print(f"[{'ok' if ok else 'skip'}] {source.name}: {message}")
        successes += int(ok)
    print(f"[result] {successes}/{len(paths)} episodes augmented")
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
