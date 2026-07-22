cmake_minimum_required(VERSION 3.8)
project(kio_teleop_openarm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/GraspCandidate.msg"
  "msg/GraspCandidateArray.msg"
  "srv/GripperCmd.srv"
  "srv/DetectObjects.srv"
  "srv/PlanGrasp.srv"
  "srv/SelectGrasp.srv"
  "srv/DemoTrigger.srv"
  DEPENDENCIES sensor_msgs std_msgs geometry_msgs trajectory_msgs
)

# Install Python package via direct directory install (avoids cmake target conflict)
install(DIRECTORY ${PROJECT_NAME}
  DESTINATION lib/python3.12/site-packages
)

# Install config and launch
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# Install tools (standalone scripts)
install(PROGRAMS
  tools/camera_calibrate.py
  tools/hand_eye_calibrate.py
  tools/verify_door_position.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install node entry points
install(PROGRAMS
  ${PROJECT_NAME}/nodes/vr_bridge.py
  ${PROJECT_NAME}/nodes/controller.py
  ${PROJECT_NAME}/nodes/simulator.py
  ${PROJECT_NAME}/nodes/motor_bridge.py
  ${PROJECT_NAME}/nodes/keyboard_interface.py
  ${PROJECT_NAME}/nodes/camera_node.py
  ${PROJECT_NAME}/nodes/perception_node.py
  ${PROJECT_NAME}/nodes/grasp_planner_node.py
  ${PROJECT_NAME}/nodes/trajectory_executor_node.py
  ${PROJECT_NAME}/nodes/auto_grasp_state_node.py
  ${PROJECT_NAME}/nodes/app_bridge.py
  ${PROJECT_NAME}/nodes/demo_state_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
