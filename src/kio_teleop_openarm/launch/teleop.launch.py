#!/usr/bin/env python3
"""Launch file for kio_teleop_openarm — UPOO Bimanual VR Teleop with ROS2."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("kio_teleop_openarm")
    config_dir = os.path.join(pkg_dir, "config")

    motor_enable = LaunchConfiguration("motor_enable")
    camera_enable = LaunchConfiguration("camera_enable")
    perception_enable = LaunchConfiguration("perception_enable")

    motor_enable_arg = DeclareLaunchArgument(
        "motor_enable", default_value="false",
        description="Enable real motor hardware bridge")

    camera_enable_arg = DeclareLaunchArgument(
        "camera_enable", default_value="false",
        description="Enable SPCA2100 stereo camera driver")

    perception_enable_arg = DeclareLaunchArgument(
        "perception_enable", default_value="false",
        description="Enable perception pipeline (detection + depth + grasp planning)")

    model_type_arg = DeclareLaunchArgument(
        "model_type", default_value="upoo_bimanual",
        description="MuJoCo model type or path")

    # ── Core teleop (always on) ──

    vr_bridge = Node(
        package="kio_teleop_openarm",
        executable="vr_bridge.py",
        name="vr_bridge",
        parameters=[os.path.join(config_dir, "vr_bridge.yaml")],
        output="screen",
    )

    controller = Node(
        package="kio_teleop_openarm",
        executable="controller.py",
        name="controller",
        parameters=[os.path.join(config_dir, "controller.yaml")],
        output="screen",
    )

    simulator = Node(
        package="kio_teleop_openarm",
        executable="simulator.py",
        name="simulator",
        parameters=[os.path.join(config_dir, "simulator.yaml")],
        output="screen",
    )

    motor_bridge = Node(
        package="kio_teleop_openarm",
        executable="motor_bridge.py",
        name="motor_bridge",
        parameters=[os.path.join(config_dir, "motor_bridge.yaml")],
        output="screen",
        condition=IfCondition(motor_enable),
    )

    keyboard_interface = Node(
        package="kio_teleop_openarm",
        executable="keyboard_interface.py",
        name="keyboard_interface",
        output="screen",
    )

    # ── Perception pipeline (W2+W3) ──

    camera_node = Node(
        package="kio_teleop_openarm",
        executable="camera_node.py",
        name="camera_node",
        parameters=[os.path.join(config_dir, "camera.yaml")],
        output="screen",
        condition=IfCondition(camera_enable),
    )

    perception_node = Node(
        package="kio_teleop_openarm",
        executable="perception_node.py",
        name="perception_node",
        parameters=[os.path.join(config_dir, "perception.yaml")],
        output="screen",
        condition=IfCondition(perception_enable),
    )

    grasp_planner_node = Node(
        package="kio_teleop_openarm",
        executable="grasp_planner_node.py",
        name="grasp_planner_node",
        output="screen",
        condition=IfCondition(perception_enable),
    )

    trajectory_executor = Node(
        package="kio_teleop_openarm",
        executable="trajectory_executor_node.py",
        name="trajectory_executor_node",
        output="screen",
    )

    # ── High-level state machines (W4+W5+W6) ──

    auto_grasp_state_node = Node(
        package="kio_teleop_openarm",
        executable="auto_grasp_state_node.py",
        name="auto_grasp_state_node",
        output="screen",
    )

    demo_state_node = Node(
        package="kio_teleop_openarm",
        executable="demo_state_node.py",
        name="demo_state_node",
        output="screen",
    )

    app_bridge = Node(
        package="kio_teleop_openarm",
        executable="app_bridge.py",
        name="app_bridge",
        output="screen",
    )

    return LaunchDescription([
        motor_enable_arg,
        camera_enable_arg,
        perception_enable_arg,
        model_type_arg,
        vr_bridge,
        controller,
        simulator,
        motor_bridge,
        keyboard_interface,
        camera_node,
        perception_node,
        grasp_planner_node,
        trajectory_executor,
        auto_grasp_state_node,
        demo_state_node,
        app_bridge,
    ])
