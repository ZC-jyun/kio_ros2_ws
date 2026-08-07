#!/usr/bin/env python3
"""Auto-grasp demo in MuJoCo — clean scene with bimanual arm and graspable cup.

SPACE = start auto-grasp  |  R = reset  |  ESC = quit
"""

import argparse
import sys
import time

import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/kiorobot/kio_robot_zzc/openarm-main/teleop_deploy")
sys.path.insert(0, "/home/kiorobot/kio_robot_zzc/kio_upoo-main")
import mujoco
from mujoco import viewer as mujoco_viewer

from upoo_cartesian_ik import (
    sample_joint_path,
    site_pose,
    smoothstep,
    solve_site_pose_ik,
)

# ── Custom content to inject into the arm XML ──
CUSTOM_ASSETS = """
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5" reflectance="0.2"/>
    <material name="table_mat" rgba="0.6 0.4 0.2 1.0" reflectance="0.1"/>
    <material name="red" rgba="1.0 0.2 0.2 1.0"/>
"""

CUSTOM_DEFAULTS = """
    <default class="cup_col">
      <geom type="box" size="0.035 0.035 0.035" mass="0.25" solref="0.005 1" friction="0.8 0.1 0.1"/>
    </default>
"""

CUSTOM_WORLDBODY = """
    <light pos="2 2 2" dir="-1 -1 -1" directional="false"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- Table -->
    <body name="table" pos="-0.45 0 0.35">
      <geom name="table_top" type="box" size="0.4 0.3 0.02" material="table_mat"/>
      <geom name="table_leg1" type="box" size="0.02 0.02 0.33" pos="0.35 0.25 -0.35"/>
      <geom name="table_leg2" type="box" size="0.02 0.02 0.33" pos="0.35 -0.25 -0.35"/>
      <geom name="table_leg3" type="box" size="0.02 0.02 0.33" pos="-0.35 0.25 -0.35"/>
      <geom name="table_leg4" type="box" size="0.02 0.02 0.33" pos="-0.35 -0.25 -0.35"/>
    </body>

    <!-- Cup on table (table top at z=0.35 + 0.02 = 0.37) -->
    <!-- IK target debug marker (green sphere with freejoint, moved in code) -->
    <body name="debug_target" pos="0.3 0 0.5" mocap="true">
      <geom name="debug_target_geom" type="sphere" size="0.025" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>
    </body>

    <body name="cup" pos="{cup_x} {cup_y} 0.43">
      <freejoint/>
      <geom name="cup_body" class="cup_col"/>
      <geom name="cup_vis" type="box" size="0.035 0.035 0.035" material="red" contype="0" conaffinity="0"/>
    </body>
"""

SCENE_HEADER = """<mujoco>
  <compiler angle="radian" autolimits="true" meshdir="{meshdir}"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic">
    <flag multiccd="enable"/>
  </option>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
  </visual>

  <contact>
    <!-- Disable self-collisions: adjacent links interpenetrate at home pose,
         and implicitfast constraint forces fight J02 actuators, preventing
         the right arm from moving. -->
    <exclude body1="upoo_right_base_link" body2="upoo_right_Link_02"/>
    <exclude body1="upoo_right_base_link" body2="upoo_right_Link_03"/>
    <exclude body1="upoo_right_Link_01" body2="upoo_right_Link_03"/>
    <exclude body1="upoo_left_base_link" body2="upoo_left_Link_02"/>
    <exclude body1="upoo_left_base_link" body2="upoo_left_Link_03"/>
    <exclude body1="upoo_left_Link_01" body2="upoo_left_Link_03"/>
    <!-- Cross-arm collisions at home pose -->
    <exclude body1="upoo_right_base_link" body2="upoo_left_Link_03"/>
    <exclude body1="upoo_right_base_link" body2="upoo_left_Link_04"/>
    <exclude body1="upoo_right_Link_01" body2="upoo_left_Link_03"/>
    <exclude body1="upoo_right_Link_01" body2="upoo_left_Link_04"/>
    <exclude body1="upoo_right_Link_02" body2="upoo_left_Link_03"/>
    <exclude body1="upoo_right_Link_02" body2="upoo_left_Link_04"/>
  </contact>
"""

SCENE_FOOTER = "</mujoco>"

MODEL_DIR = "/home/kiorobot/kio_robot_zzc/kio_upoo-main/openarm_mujoco-master/v2"

RIGHT_ARM = ["upoo_right_J01", "upoo_right_J02", "upoo_right_J03",
             "upoo_right_J04", "upoo_right_J05", "upoo_right_J06"]
RIGHT_FINGERS = ["upoo_right_openarm_v1_finger_joint1", "upoo_right_openarm_v1_finger_joint2"]
RIGHT_PADS = ["upoo_right_finger1_inner_pad", "upoo_right_finger2_inner_pad"]
GRIPPER_OPEN, GRIPPER_CLOSE = 0.044, 0.0

# Right-arm home pose: L-shape (upper arm up, forearm forward — "periscope")
#    [J01, J02, J03,   J04,  J05,  J06]
HOME_Q = np.array([0.0, 0.0, 0.0, 1.57, 1.57, 0.0])


def pad_contact_summary(model, data, obj_geom_id, pad_ids):
    """Return (counts, forces) for each pad contacting the object."""
    counts = [0, 0]
    forces = [0.0, 0.0]
    contact_force = np.zeros(6)
    for ci in range(data.ncon):
        contact = data.contact[ci]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        for i, pid in enumerate(pad_ids):
            if obj_geom_id in (g1, g2) and pid in (g1, g2):
                counts[i] += 1
                mujoco.mj_contactForce(model, data, ci, contact_force)
                forces[i] += abs(float(contact_force[0]))
    return counts, forces


def grasp_state(model, data, finger_qpos_adrs, obj_geom_id, pad_ids):
    q1 = float(data.qpos[finger_qpos_adrs[0]])
    q2 = float(data.qpos[finger_qpos_adrs[1]])
    counts, forces = pad_contact_summary(model, data, obj_geom_id, pad_ids)
    bilateral = counts[0] > 0 and counts[1] > 0
    return bilateral, q1, q2, counts, forces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-frame", type=int, default=10,
                        help="Physics steps per render frame (default: 10)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Target simulation speed; use 0 for as fast as possible.")
    parser.add_argument("--record", type=Path, default=None,
                        help="Output MP4 video file path (e.g. demo.mp4)")
    parser.add_argument("--record-fps", type=int, default=30,
                        help="FPS of the recorded video (default: 30)")
    parser.add_argument("--record-resolution", type=int, nargs=2,
                        default=[1280, 720], metavar=("WIDTH", "HEIGHT"),
                        help="Resolution of the recorded video (default: 1280 720)")
    parser.add_argument("--cup-x", type=float, default=-0.35,
                        help="Cup X position on table (default: -0.35)")
    parser.add_argument("--cup-y", type=float, default=0.08,
                        help="Cup Y position on table (default: 0.08)")
    args = parser.parse_args()

    import re, tempfile, os, shutil

    # ── Read the bimanual arm XML and strip outer <mujoco> wrapper ──
    bimanual_path = os.path.join(MODEL_DIR, "upoo_openarm_v1_hybrid_grasp_bimanual_v4.xml")
    with open(bimanual_path) as f:
        arm_xml = f.read()

    # Remove <?xml?> declaration, <mujoco ...> opening, </mujoco> closing,
    # and the original <compiler> + <option> elements (we provide our own).
    arm_xml = re.sub(r"<\?xml[^?]*\?>\s*", "", arm_xml)
    arm_xml = re.sub(r"<mujoco[^>]*>", "", arm_xml, count=1)
    arm_xml = re.sub(r"</mujoco>", "", arm_xml, count=1)
    arm_xml = re.sub(r"<compiler[^>]*/>\s*", "", arm_xml, count=1)
    arm_xml = re.sub(r"<option>.*?</option>\s*", "", arm_xml, count=1, flags=re.DOTALL)
    arm_xml = arm_xml.strip()

    # Fix 1: right J04 joint range.
    # The right arm home keyframe has J04=-1.57, but the joint range is [-0.78, 2.6]
    # (same as left arm). -1.57 is OUTSIDE this range, so MuJoCo's constraint solver
    # pushes J04 back toward -0.78 every step. This constraint force propagates
    # through the kinematic chain and cancels out the J02 actuator torque (25 Nm).
    # Widen right J04 range to [-1.57, 2.6] so the home position is legal.
    arm_xml = arm_xml.replace(
        'name="upoo_right_J04" type="hinge" axis="0 0 1" range="-0.78 2.6"',
        'name="upoo_right_J04" type="hinge" axis="0 0 1" range="-1.57 2.6"')

    # Fix 2: right J02 ctrlrange — was [-3.14, 0], widen to match joint range.
    arm_xml = arm_xml.replace(
        'ctrlrange="-3.14 0" forcelimited="true" forcerange="-30 30"',
        'ctrlrange="-3.14 1.57" forcelimited="true" forcerange="-30 30"')

    # Inject custom content into existing sections
    arm_xml = arm_xml.replace("<asset>", "<asset>" + CUSTOM_ASSETS)
    arm_xml = arm_xml.replace("<default>", "<default>" + CUSTOM_DEFAULTS)
    arm_xml = arm_xml.replace(
        "<worldbody>",
        "<worldbody>" + CUSTOM_WORLDBODY.format(cup_x=args.cup_x, cup_y=args.cup_y))

    # Resolve meshdir for the STL mesh files
    meshdir = MODEL_DIR

    # Assemble final scene
    scene_xml = (
        SCENE_HEADER.format(meshdir=meshdir)
        + arm_xml
        + "\n" + SCENE_FOOTER
    )

    # Write scene XML to temp file
    tmp_dir = tempfile.mkdtemp()
    scene_path = os.path.join(tmp_dir, "scene.xml")
    with open(scene_path, "w") as f:
        f.write(scene_xml)

    # Symlink mesh asset files so MuJoCo can find them
    assets_dir = os.path.join(MODEL_DIR, "assets")
    if os.path.isdir(assets_dir):
        dst = os.path.join(tmp_dir, "assets")
        if not os.path.exists(dst):
            os.symlink(assets_dir, dst)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    recorded_frames: list[np.ndarray] = []
    if args.record:
        width, height = args.record_resolution
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        print(f"Recording video: {width}x{height} @ {args.record_fps} fps → {args.record}")
    else:
        # Low-res GPU warmup — masks vsync jitter by keeping the GPU pipeline
        # busy between viewer.sync() calls, exactly like the recording path.
        width, height = 320, 240
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    recorder = mujoco.Renderer(model, height=height, width=width)

    grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "upoo_right_tcp")
    cup_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    debug_mocap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "debug_target")
    debug_mocap_id = model.body_mocapid[debug_mocap]  # mocap bodies use mocap id, not body id

    arm_dofs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in RIGHT_ARM]
    finger_dofs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in RIGHT_FINGERS]

    finger_act_ids = []  # actuator indices for right gripper
    for name in ("upoo_right_gripper_ctrl",):
        try:
            finger_act_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
        except ValueError:
            pass
    assert finger_act_ids, "Gripper actuator 'upoo_right_gripper_ctrl' not found in model!"

    pad_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pn) for pn in RIGHT_PADS)
    cup_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cup_body")

    # Map arm actuators → qpos addresses for right arm
    arm_act_to_dof = {}  # act_id → index into arm_dofs
    for act_id in range(model.nu):
        jid = model.actuator_trnid[act_id, 0]
        if jid < 0:
            continue
        adr = model.jnt_qposadr[jid]
        if adr in arm_dofs:
            arm_act_to_dof[act_id] = arm_dofs.index(adr)

    def set_arm_ctrl(q_target):
        """Set arm actuator ctrl to target joint angles (PD controllers execute)."""
        for act_id, idx in arm_act_to_dof.items():
            data.ctrl[act_id] = q_target[idx]

    def set_gripper_ctrl(val):
        for act_id in finger_act_ids:
            data.ctrl[act_id] = val

    # ── Set initial pose ──
    import math

    left_arm  = ["upoo_left_J01", "upoo_left_J02", "upoo_left_J03",
                 "upoo_left_J04", "upoo_left_J05", "upoo_left_J06"]
    left_dofs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in left_arm]
    left_home = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0])
    for i, adr in enumerate(left_dofs):
        data.qpos[adr] = left_home[i]
    for i, adr in enumerate(arm_dofs):
        data.qpos[adr] = HOME_Q[i]
    for fd in finger_dofs:
        data.qpos[fd] = GRIPPER_OPEN

    mujoco.mj_forward(model, data)
    set_arm_ctrl(HOME_Q)
    set_gripper_ctrl(GRIPPER_OPEN)

    # ── Plan Cartesian path using MuJoCo IK ──
    planning_data = mujoco.MjData(model)
    planning_data.qpos[:] = data.qpos
    planning_data.qvel[:] = 0.0
    mujoco.mj_forward(model, planning_data)
    start_pos, start_rot = site_pose(planning_data, grasp_site)
    cup_target = data.xpos[cup_body].copy()

    waypoint_count = 61
    path = [HOME_Q.copy()]
    seed_q = HOME_Q.copy()
    max_pos_err = 0.0
    max_ori_err = 0.0

    for i in range(1, waypoint_count):
        alpha = i / (waypoint_count - 1)
        target_pos = start_pos + (cup_target - start_pos) * smoothstep(alpha)
        seed_q, pos_err, ori_err, _ = solve_site_pose_ik(
            model, planning_data,
            joint_names=RIGHT_ARM,
            site_name="upoo_right_tcp",
            seed_q=seed_q,
            target_position=target_pos,
            target_rotation=start_rot,
        )
        path.append(seed_q.copy())
        max_pos_err = max(max_pos_err, pos_err)
        max_ori_err = max(max_ori_err, ori_err)

    path = np.array(path)
    bottom_q = path[-1]

    # ── Timing parameters ──
    SETTLE_TIME = 1.0
    DESCENT_TIME = 3.0
    CLOSE_RAMP_TIME = 1.0
    CONTACT_TIMEOUT = 10.0
    STABLE_CONTACT_TIME = 0.3
    PRELIFT_HOLD = 0.5
    LIFT_TIME = 3.0
    FINAL_HOLD = 3.0

    settle_end = SETTLE_TIME
    descent_end = settle_end + DESCENT_TIME
    close_ramp_end = descent_end + CLOSE_RAMP_TIME
    contact_deadline = close_ramp_end + CONTACT_TIMEOUT

    # ── State ──
    stage = ""
    stable_contact_start = None
    contact_acquired_time = None
    lift_start = None
    lift_end = None
    finish_time = None
    failure_time = None
    cup_settled_pos = None
    result_printed = False
    auto_started = False
    key_flags = {"space": False, "r": False}

    def key_cb(code):
        if code == 32: key_flags["space"] = True
        if code == 82: key_flags["r"] = True

    cup_pos = data.xpos[cup_body]
    grasp_home = data.site_xpos[grasp_site].copy()
    mujoco_tcp_home = grasp_home
    print("=" * 60)
    print("  SPACE = auto-grasp  |  R = reset  |  ESC = quit")
    print(f"  MuJoCo TCP at home:      ({mujoco_tcp_home[0]:.4f}, {mujoco_tcp_home[1]:.4f}, {mujoco_tcp_home[2]:.4f})")
    print(f"  Cup target:              ({cup_target[0]:.4f}, {cup_target[1]:.4f}, {cup_target[2]:.4f})")
    print(f"  Cup (live):              ({cup_pos[0]:.4f}, {cup_pos[1]:.4f}, {cup_pos[2]:.4f})")
    print(f"  Planned path: {waypoint_count} waypoints")
    print(f"  Max IK error — pos: {max_pos_err:.6f} m, "
          f"ori: {math.degrees(max_ori_err):.4f} deg")
    print(f"  Home Q:       {np.array2string(HOME_Q, precision=4)}")
    print(f"  Path[ 0] Q:   {np.array2string(path[0], precision=4)}")
    print(f"  Path[-1] Q:   {np.array2string(path[-1], precision=4)}")
    print(f"  TCP→cup dist: {np.linalg.norm(cup_target - start_pos):.4f} m")
    print("=" * 60)

    print(f"Physics timestep={model.opt.timestep:.6f} s, steps/frame={args.steps_per_frame}, speed={args.speed:g}x")

    wall_origin = time.perf_counter()
    sim_origin = float(data.time)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.sync()
        with viewer.lock():
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.sync()
        data.mocap_pos[debug_mocap_id] = data.xpos[cup_body].copy()

        while viewer.is_running():
            # ── Reset ──
            if key_flags["r"]:
                key_flags["r"] = False
                mujoco.mj_resetData(model, data)
                for i, adr in enumerate(left_dofs):
                    data.qpos[adr] = left_home[i]
                for i, adr in enumerate(arm_dofs):
                    data.qpos[adr] = HOME_Q[i]
                for fd in finger_dofs:
                    data.qpos[fd] = GRIPPER_OPEN
                set_arm_ctrl(HOME_Q)
                set_gripper_ctrl(GRIPPER_OPEN)
                stable_contact_start = None
                contact_acquired_time = None
                lift_start = None
                lift_end = None
                finish_time = None
                failure_time = None
                cup_settled_pos = None
                result_printed = False
                auto_started = False
                stage = ""
                data.mocap_pos[debug_mocap_id] = data.xpos[cup_body].copy()

            # ── SPACE to start ──
            if not auto_started and key_flags["space"]:
                key_flags["space"] = False
                auto_started = True
                mujoco.mj_resetData(model, data)
                for i, adr in enumerate(left_dofs):
                    data.qpos[adr] = left_home[i]
                for i, adr in enumerate(arm_dofs):
                    data.qpos[adr] = HOME_Q[i]
                for fd in finger_dofs:
                    data.qpos[fd] = GRIPPER_OPEN
                set_arm_ctrl(HOME_Q)
                set_gripper_ctrl(GRIPPER_OPEN)
                stable_contact_start = None
                contact_acquired_time = None
                lift_start = None
                lift_end = None
                finish_time = None
                failure_time = None
                cup_settled_pos = None
                result_printed = False
                stage = ""
                data.mocap_pos[debug_mocap_id] = data.xpos[cup_body].copy()
                wall_origin = time.perf_counter()
                sim_origin = float(data.time)
                print("SPACE pressed — starting auto-grasp sequence.")

            with viewer.lock():
                for _ in range(args.steps_per_frame):
                    if not auto_started:
                        # Idle: hold home pose, physics still runs (cube settles)
                        set_arm_ctrl(HOME_Q)
                        set_gripper_ctrl(GRIPPER_OPEN)
                        mujoco.mj_step(model, data)
                        data.mocap_pos[debug_mocap_id] = data.xpos[cup_body].copy()
                        continue

                    sim_time = float(data.time)

                    if sim_time < settle_end:
                        current_stage = "settling"
                        right_q = path[0]
                        gripper_target = GRIPPER_OPEN
                    elif sim_time < descent_end:
                        if current_stage != "Cartesian descent":
                            print(f"  Descent start — first target Q: {np.array2string(path[0], precision=4)}")
                            print(f"  Descent start — last  target Q: {np.array2string(path[-1], precision=4)}")
                        current_stage = "Cartesian descent"
                        alpha = (sim_time - settle_end) / max(DESCENT_TIME, 1e-12)
                        right_q = sample_joint_path(path, smoothstep(alpha))
                        gripper_target = GRIPPER_OPEN
                    elif sim_time < close_ramp_end:
                        current_stage = "closing gripper"
                        right_q = bottom_q
                        alpha = (sim_time - descent_end) / max(CLOSE_RAMP_TIME, 1e-12)
                        gripper_target = GRIPPER_OPEN * (1.0 - alpha)
                    elif lift_start is None and failure_time is None:
                        current_stage = (
                            "stable-contact hold"
                            if stable_contact_start is not None
                            else "waiting for bilateral contact"
                        )
                        right_q = bottom_q
                        gripper_target = GRIPPER_CLOSE
                    elif failure_time is not None:
                        current_stage = "contact-timeout hold"
                        right_q = bottom_q
                        gripper_target = GRIPPER_CLOSE
                    elif lift_start is not None and sim_time < lift_start:
                        current_stage = "pre-lift hold"
                        right_q = bottom_q
                        gripper_target = GRIPPER_CLOSE
                    elif lift_end is not None and sim_time < lift_end:
                        current_stage = "Cartesian lift"
                        alpha = (sim_time - lift_start) / max(LIFT_TIME, 1e-12)
                        right_q = sample_joint_path(path, 1.0 - smoothstep(alpha))
                        gripper_target = GRIPPER_CLOSE
                    else:
                        current_stage = "final hold"
                        right_q = path[0]
                        gripper_target = GRIPPER_CLOSE

                    if current_stage != stage:
                        stage = current_stage
                        print(f"Stage: {stage} (sim t={sim_time:.3f}s)")

                    # Apply control
                    set_arm_ctrl(right_q)
                    set_gripper_ctrl(gripper_target)

                    mujoco.mj_step(model, data)

                    # Track settled cup position
                    stepped_time = float(data.time)
                    if cup_settled_pos is None and stepped_time >= settle_end:
                        cup_settled_pos = np.asarray(data.xpos[cup_body], dtype=float).copy()

                    # Update debug marker
                    data.mocap_pos[debug_mocap_id] = data.xpos[cup_body].copy()

                    # Contact detection after gripper starts closing
                    if stepped_time >= close_ramp_end and lift_start is None and failure_time is None:
                        bilateral, q1, q2, cnts, forces = grasp_state(
                            model, data, finger_dofs, cup_geom_id, pad_ids)
                        if bilateral:
                            if stable_contact_start is None:
                                stable_contact_start = stepped_time
                                print(f"  Bilateral contact detected (sim t={stepped_time:.3f}s)")
                            if stepped_time - stable_contact_start >= STABLE_CONTACT_TIME:
                                contact_acquired_time = stepped_time
                                lift_start = contact_acquired_time + PRELIFT_HOLD
                                lift_end = lift_start + LIFT_TIME
                                finish_time = lift_end + FINAL_HOLD
                                print(f"  Stable grasp acquired! qpos={q1:.4f}/{q2:.4f}, "
                                      f"contacts={cnts[0]}/{cnts[1]}, forces={forces[0]:.2f}/{forces[1]:.2f}N")
                                print(f"  Lift scheduled for sim t={lift_start:.3f}s")
                        else:
                            if stable_contact_start is not None:
                                print("  Bilateral contact lost; timer reset.")
                            stable_contact_start = None

                        if stepped_time >= contact_deadline and lift_start is None:
                            failure_time = stepped_time
                            print(f"  Contact timeout at sim t={stepped_time:.3f}s")

                    # Print evaluation
                    evaluation_due = (finish_time is not None and stepped_time >= finish_time) or (
                        failure_time is not None and stepped_time >= failure_time + FINAL_HOLD)
                    if evaluation_due and not result_printed:
                        result_printed = True
                        cur_pos = np.asarray(data.xpos[cup_body], dtype=float).copy()
                        if cup_settled_pos is None:
                            cup_settled_pos = cur_pos.copy()
                        lift_height = float(cur_pos[2] - cup_settled_pos[2])
                        final_bilateral, _, _, _, _ = grasp_state(
                            model, data, finger_dofs, cup_geom_id, pad_ids)
                        print(f"\nGrasp evaluation:")
                        print(f"  Lift height:   {lift_height:.4f} m")
                        print(f"  Final contact: {'yes' if final_bilateral else 'no'}")
                        print(f"  Result:        {'SUCCESS' if (contact_acquired_time and final_bilateral and lift_height > 0.02) else 'INCOMPLETE'}")

            viewer.sync()

            recorder.update_scene(data, camera=viewer.cam)
            if args.record:
                recorded_frames.append(recorder.render().copy())
            else:
                recorder.render()  # GPU warmup — no pixel capture

            if args.speed > 0:
                target_wall = wall_origin + (float(data.time) - sim_origin) / args.speed
                remaining = target_wall - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

    viewer.close()

    if args.record and recorded_frames:
        print(f"Writing {len(recorded_frames)} frames to {args.record}...")
        import imageio
        imageio.mimsave(
            str(args.record), recorded_frames,
            fps=args.record_fps, codec="libx264",
            output_params=["-preset", "fast", "-crf", "23"],
        )
        print("Done.")
    recorder.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
