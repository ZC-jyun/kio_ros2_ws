# kio_ros2_ws

KIO Robot ROS2 workspace — UPOO dual-arm VR teleoperation.

## Setup

```bash
# Build
colcon build --packages-select kio_teleop_openarm
source install/setup.bash

# Launch
ros2 launch kio_teleop_openarm teleop.launch.py motor_enable:=true
```

## Dependencies

- ROS2 Humble
- [openarm-main](https://github.com/KioRobot/OPENARM) (MuJoCo model, motor SDK, hardware bridge)
- Grounding DINO (detection)
- PyTorch

## Nodes

| Node | Description |
|------|-------------|
| `controller.py` | DLS IK solver, VR hand tracking, gripper control |
| `simulator.py` | MuJoCo simulation, publishes `/joint_state` |
| `motor_bridge.py` | DM motor CAN bus bridge, subscribes `/joint_target` |
| `vr_bridge.py` | VR headset data relay |
| `camera_node.py` | SPCA2100 stereo camera driver |
| `perception_node.py` | Object detection + depth estimation |
| `grasp_planner_node.py` | Grasp planning + trajectory generation |
| `trajectory_executor_node.py` | Joint trajectory playback |
| `app_bridge.py` | WebSocket bridge for App UI |
