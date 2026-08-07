#!/usr/bin/env python3
"""Quick MuJoCo model viewer — ESC to quit."""
import sys
import os
import tempfile

model_xml = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/kio_robot_zzc/kio_ros2_ws/assets/mujoco/upoo_bimanual.xml")

import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path(model_xml)
data = mujoco.MjData(model)

print(f"Model: {model_xml}")
print(f"Bodies: {model.nbody}, Joints: {model.njnt}, Geoms: {model.ngeom}")
print("ESC = quit")

viewer.launch(model, data)

