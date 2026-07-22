#!/usr/bin/env python3
"""Standalone MuJoCo viewer — no ROS2, just the robot model and sliders."""
import sys
sys.path.insert(0, "/home/kiorobot/kio_robot_zzc/openarm-main/teleop_deploy")

import mujoco
from mujoco import viewer
import openarm_mujoco.v2 as openarm_mujoco

xml_path = openarm_mujoco.openarm_upoo_bimanual_xml()
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Home: all arm joints at 0, grippers open
LEFT_ARM = ["upoo_left_Base_J01", "upoo_left_J02", "upoo_left_J03",
            "upoo_left_J04", "upoo_left_J05", "upoo_left_J06"]
RIGHT_ARM = ["upoo_right_Base_J01", "upoo_right_J02", "upoo_right_J03",
             "upoo_right_J04", "upoo_right_J05", "upoo_right_J06"]
FINGERS = ["upoo_left_finger_left_joint", "upoo_left_finger_right_joint",
           "upoo_right_finger_left_joint", "upoo_right_finger_right_joint"]

for name in LEFT_ARM + RIGHT_ARM:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = 0.0
for name in FINGERS:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = 0.044

# Sync ctrl <- qpos
for i in range(model.nu):
    if model.actuator_trntype[i] == 0:  # joint type
        jid = model.actuator_trnid[i, 0]
        data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]

mujoco.mj_forward(model, data)

print("MuJoCo viewer — drag ctrl sliders to move joints. Close window to exit.")
viewer.launch(model, data)
