#!/usr/bin/env python3
"""grasp_planner_node — spatial mapping + grasp generation + trajectory planning.

Subscribes to /joint_state, /perception/detections, /perception/depth.
Publishes /grasp/candidates.
Provides /grasp/plan and /grasp/select services.
"""

import os
import sys
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

_DEPLOY_DIR = Path("/home/kiorobot/kio_robot_zzc/openarm-main/teleop_deploy")
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

import mujoco
import openarm_mujoco.v2 as openarm_mujoco

from kio_teleop_openarm.lib.transforms import make_transform, quat_xyzw_from_matrix
from kio_teleop_openarm.lib.ik_solver import IKSolver
from kio_teleop_openarm.lib.spatial_mapper import SpatialMapper, load_hand_eye_and_build_T_base_cam
from kio_teleop_openarm.lib.grasp_generator import GraspGenerator
from kio_teleop_openarm.lib.trajectory_planner import TrajectoryPlanner

# Joint name constants — must match controller.py
LEFT_ARM_JOINTS = [
    "upoo_left_Base_J01", "upoo_left_J02", "upoo_left_J03",
    "upoo_left_J04", "upoo_left_J05", "upoo_left_J06",
]
RIGHT_ARM_JOINTS = [
    "upoo_right_Base_J01", "upoo_right_J02", "upoo_right_J03",
    "upoo_right_J04", "upoo_right_J05", "upoo_right_J06",
]
FINGER_JOINTS = [
    "upoo_left_finger_left_joint", "upoo_left_finger_right_joint",
    "upoo_right_finger_left_joint", "upoo_right_finger_right_joint",
]
ALL_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + FINGER_JOINTS
LEFT_EE_BODY = "upoo_left_Link_06"
RIGHT_EE_BODY = "upoo_right_Link_06"


def mat4_to_pose(mat):
    """4x4 numpy → geometry_msgs/Pose."""
    from geometry_msgs.msg import Pose, Point, Quaternion
    from scipy.spatial.transform import Rotation
    pose = Pose()
    pose.position = Point(x=float(mat[0, 3]), y=float(mat[1, 3]), z=float(mat[2, 3]))
    r = Rotation.from_matrix(mat[:3, :3])
    q = r.as_quat()  # [x, y, z, w]
    pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
    return pose


class GraspPlannerNode(Node):
    def __init__(self):
        super().__init__("grasp_planner_node")

        self.declare_parameter("model_type", "upoo_bimanual")
        self.declare_parameter("calib_path", "")
        self.declare_parameter("hand_eye_path", "")
        self.declare_parameter("table_height", 0.0)
        model_type = self.get_parameter("model_type").value
        calib_path = self.get_parameter("calib_path").value
        hand_eye_path = self.get_parameter("hand_eye_path").value
        self.table_height = self.get_parameter("table_height").value

        # ── MuJoCo kinematics model ──
        model_loaders = {
            "upoo_cell": openarm_mujoco.openarm_upoo_cell_xml,
            "upoo_bimanual": openarm_mujoco.openarm_upoo_bimanual_xml,
        }
        if model_type in model_loaders:
            xml_path = model_loaders[model_type]()
        elif os.path.exists(model_type):
            xml_path = os.path.abspath(model_type)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")
        self.get_logger().info(f"Loading model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ── Joint indices ──
        def _qpos_idx(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_qposadr[jid]

        def _dof_idx(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_dofadr[jid]

        self.left_qpos = np.array([_qpos_idx(n) for n in LEFT_ARM_JOINTS], dtype=int)
        self.right_qpos = np.array([_qpos_idx(n) for n in RIGHT_ARM_JOINTS], dtype=int)
        self.left_dofs = np.array([_dof_idx(n) for n in LEFT_ARM_JOINTS], dtype=int)
        self.right_dofs = np.array([_dof_idx(n) for n in RIGHT_ARM_JOINTS], dtype=int)
        self.left_finger_qpos = np.array([_qpos_idx(n) for n in FINGER_JOINTS[:2]], dtype=int)
        self.right_finger_qpos = np.array([_qpos_idx(n) for n in FINGER_JOINTS[2:]], dtype=int)
        all_qpos_indices = np.concatenate([
            self.left_qpos, self.right_qpos,
            self.left_finger_qpos, self.right_finger_qpos,
        ])

        self.left_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_BODY)
        self.right_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY)

        # ── Build joint name → qpos index map ──
        self.joint_name_to_qpos = {}
        for name, idx in zip(ALL_JOINTS, all_qpos_indices):
            self.joint_name_to_qpos[name] = int(idx)

        # ── IK Solvers ──
        self.left_ik = IKSolver(
            self.model, self.data, self.left_body, self.left_dofs, self.left_qpos)
        self.right_ik = IKSolver(
            self.model, self.data, self.right_body, self.right_dofs, self.right_qpos)

        # ── Spatial mapper ──
        self._mapper = None
        if calib_path and hand_eye_path:
            try:
                calib = np.load(calib_path)
                K_left = calib["K_left"]
                T_base_cam = load_hand_eye_and_build_T_base_cam(
                    hand_eye_path, self.model, self.data, self.left_body)
                self._mapper = SpatialMapper(K_left, T_base_cam)
                self.get_logger().info("Spatial mapper initialized")
            except Exception as e:
                self.get_logger().warn(f"Spatial mapper not available: {e}")

        # ── Grasp generator ──
        self._grasp_gen = GraspGenerator(
            self.left_ik, self.right_ik,
            self.model, self.data,
            self.left_body, self.right_body,
            table_height=self.table_height)

        # ── Trajectory planner ──
        self._traj_planner = TrajectoryPlanner(
            self.left_ik, self.right_ik,
            self.model, self.data,
            self.left_body, self.right_body,
            self.left_qpos, self.right_qpos,
            table_height=self.table_height)

        # ── State ──
        self._latest_qpos = None
        self._latest_detections = []
        self._latest_depth = None
        self._candidates = []
        self._selected_obj_idx = -1
        self._selected_grasp_idx = -1

        # ── Subscribers ──
        self._js_sub = self.create_subscription(
            JointState, "/joint_state", self._js_cb, 10)

        try:
            from kio_teleop_openarm.msg import DetectionArray
            self._det_sub = self.create_subscription(
                DetectionArray, "/perception/detections", self._det_cb, 10)
        except ImportError:
            self.get_logger().warn("DetectionArray msg not available")
            self._det_sub = None

        from sensor_msgs.msg import Image
        self._depth_sub = self.create_subscription(
            Image, "/perception/depth", self._depth_cb, 10)

        # ── Publishers ──
        try:
            from kio_teleop_openarm.msg import GraspCandidate, GraspCandidateArray
            self._GraspCandidate = GraspCandidate
            self._GraspCandidateArray = GraspCandidateArray
            self._cand_pub = self.create_publisher(
                GraspCandidateArray, "/grasp/candidates", 10)
        except ImportError:
            self.get_logger().warn("GraspCandidate msg types not available")
            self._GraspCandidate = None
            self._GraspCandidateArray = None
            self._cand_pub = None

        self._traj_pub = self.create_publisher(
            JointTrajectory, "/trajectory/plan", 10)

        # ── Services ──
        try:
            from kio_teleop_openarm.srv import PlanGrasp, SelectGrasp
            self.create_service(PlanGrasp, "/grasp/plan", self._plan_cb)
            self.create_service(SelectGrasp, "/grasp/select", self._select_cb)
            self.get_logger().info("/grasp/plan and /grasp/select services ready")
        except ImportError:
            self.get_logger().warn("PlanGrasp/SelectGrasp srv not available")

        self.get_logger().info("grasp_planner_node started")

    # ── Callbacks ──
    def _js_cb(self, msg: JointState):
        qpos = {}
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_name_to_qpos:
                qpos[name] = float(pos)
        if len(qpos) >= 16:
            for name, qidx in self.joint_name_to_qpos.items():
                self.data.qpos[qidx] = qpos.get(name, 0.0)
            self._latest_qpos = self.data.qpos.copy()

    def _det_cb(self, msg):
        self._latest_detections = []
        for d in msg.detections:
            self._latest_detections.append({
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": list(d.bbox),
            })
        if self._latest_depth is not None and self._mapper is not None:
            self._generate_candidates()

    def _depth_cb(self, msg):
        if msg.encoding == "32FC1":
            self._latest_depth = np.frombuffer(
                msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        else:
            self._latest_depth = None

    def _generate_candidates(self):
        """Run spatial_mapper + grasp_generator on latest detections."""
        if self._latest_qpos is None:
            return
        candidates = []
        for i, det in enumerate(self._latest_detections):
            obj_3d = self._mapper.get_object_3d_center(det["bbox"], self._latest_depth)
            if obj_3d is None:
                continue
            grasps = self._grasp_gen.generate_candidates(det, obj_3d, arm="left")
            for g in grasps:
                g["obj_idx"] = i
                candidates.append(g)

        self._candidates = candidates
        self.get_logger().info(f"Generated {len(candidates)} grasp candidates")

        if self._GraspCandidateArray is not None and self._cand_pub is not None:
            arr = self._GraspCandidateArray()
            for c in candidates:
                gc = self._GraspCandidate()
                gc.grasp_id = str(c.get("grasp_id", ""))
                gc.description = c.get("description", "")
                gc.score = float(c.get("score", 0.0))
                gc.pre_grasp_pose = mat4_to_pose(c["pre_grasp_pose"])
                gc.grasp_pose = mat4_to_pose(c["grasp_pose"])
                arr.candidates.append(gc)
            self._cand_pub.publish(arr)

    # ── Service callbacks ──
    def _plan_cb(self, request, response):
        """Plan a trajectory for the given grasp candidate."""
        if self._latest_qpos is None:
            response.success = False
            return response

        from geometry_msgs.msg import Pose
        from scipy.spatial.transform import Rotation

        def pose_to_mat4(pose: Pose):
            q = pose.orientation
            r = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            t = np.eye(4)
            t[:3, :3] = r
            t[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
            return t

        pre_grasp = pose_to_mat4(request.candidate.pre_grasp_pose)
        grasp = pose_to_mat4(request.candidate.grasp_pose)

        q_curr = self._latest_qpos.copy()
        traj = self._traj_planner.plan_pick(q_curr, pre_grasp, grasp, arm="left")
        if traj is None:
            traj = self._traj_planner.plan_pick_simple(q_curr, pre_grasp, arm="left")

        if traj is None:
            response.success = False
            return response

        jt = JointTrajectory()
        jt.joint_names = ALL_JOINTS
        for t_rel, q in traj:
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            pt.time_from_start.sec = int(t_rel)
            pt.time_from_start.nanosec = int((t_rel - int(t_rel)) * 1e9)
            jt.points.append(pt)

        response.trajectory = jt
        response.success = True
        self._traj_pub.publish(jt)
        self.get_logger().info(f"Planned trajectory: {len(traj)} points, {traj[-1][0]:.1f}s")
        return response

    def _select_cb(self, request, response):
        """Store the user's selected grasp candidate."""
        self._selected_obj_idx = request.obj_idx
        self._selected_grasp_idx = request.grasp_idx
        self.get_logger().info(
            f"Selected grasp: obj={request.obj_idx}, grasp={request.grasp_idx}")
        response.success = True
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
