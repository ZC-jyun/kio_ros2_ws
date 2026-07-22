#!/usr/bin/env python3
"""app_bridge — WebSocket ↔ ROS2 bridge for mobile app communication.

Forwards ROS2 state topics to WebSocket clients.
Receives WebSocket commands and calls ROS2 services.
"""

import asyncio
import json
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class AppBridgeNode(Node):
    def __init__(self):
        super().__init__("app_bridge")

        self.declare_parameter("ws_host", "0.0.0.0")
        self.declare_parameter("ws_port", 8765)
        ws_host = self.get_parameter("ws_host").value
        ws_port = self.get_parameter("ws_port").value

        # Connected WebSocket clients
        self._clients = set()
        self._clients_lock = threading.Lock()

        # Pending requests from WebSocket → ROS2
        self._pending_auto_grasp = False
        self._pending_selection = None
        self._pending_demo = None
        self._pending_estop = 0
        self._lock = threading.Lock()

        # ── Subscribers (forward to WebSocket) ──
        self._auto_grasp_state_sub = self.create_subscription(
            String, "/auto_grasp/state", self._forward_cb("auto_grasp_state"), 10)
        self._demo_state_sub = self.create_subscription(
            String, "/demo/state", self._forward_cb("demo_state"), 10)

        try:
            from kio_teleop_openarm.msg import GraspCandidateArray
            self._grasp_cand_sub = self.create_subscription(
                GraspCandidateArray, "/grasp/candidates", self._grasp_cand_cb, 10)
        except ImportError:
            self._grasp_cand_sub = None

        # ── Service clients ──
        self._clients_ready = {}

        # ── Timer to process pending WebSocket requests ──
        self._timer = self.create_timer(0.1, self._process_pending)

        # ── Start WebSocket server in background thread ──
        self._ws_host = ws_host
        self._ws_port = ws_port
        self._loop = asyncio.new_event_loop()
        self._ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self._ws_thread.start()

        self.get_logger().info(f"app_bridge started on ws://{ws_host}:{ws_port}")

    # ── Subscriber callbacks ──

    def _forward_cb(self, msg_type: str):
        def cb(msg: String):
            self._broadcast({msg_type: json.loads(msg.data) if msg.data else {}})
        return cb

    def _grasp_cand_cb(self, msg):
        candidates = []
        for c in msg.candidates:
            candidates.append({
                "grasp_id": c.grasp_id,
                "description": c.description,
                "score": c.score,
            })
        self._broadcast({"type": "grasp_candidates", "candidates": candidates})

    # ── WebSocket server ──

    def _run_ws_server(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(
                asyncio.start_server(
                    self._handle_client,
                    self._ws_host, self._ws_port))
            self._loop.run_forever()
        except Exception as e:
            self.get_logger().error(f"WebSocket server error: {e}")

    async def _handle_client(self, websocket):
        self.get_logger().info(f"App connected: {websocket.remote_address}")
        with self._clients_lock:
            self._clients.add(websocket)
        try:
            async for message in websocket:
                await self._on_message(message)
        except Exception:
            pass
        finally:
            with self._clients_lock:
                self._clients.discard(websocket)
            self.get_logger().info("App disconnected")

    async def _on_message(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        msg_type = data.get("type", "")

        if msg_type == "auto_grasp":
            cmd = data.get("cmd", "")
            if cmd == "detect":
                with self._lock:
                    self._pending_auto_grasp = True
                self.get_logger().info("← auto_grasp detect")
        elif msg_type == "select_grasp":
            with self._lock:
                self._pending_selection = {
                    "obj_idx": int(data.get("object_id", 0)),
                    "grasp_idx": int(data.get("grasp_id", 0)),
                }
            self.get_logger().info(
                f"← select_grasp obj={self._pending_selection['obj_idx']} "
                f"grasp={self._pending_selection['grasp_idx']}")
        elif msg_type == "demo":
            with self._lock:
                self._pending_demo = data.get("demo_type", "")
            self.get_logger().info(f"← demo {self._pending_demo}")
        elif msg_type == "estop":
            with self._lock:
                self._pending_estop = int(data.get("code", 1))
            self.get_logger().info(f"← estop code={self._pending_estop}")

    def _broadcast(self, data: dict):
        with self._clients_lock:
            if not self._clients:
                return
            dead = set()
            for ws in self._clients:
                try:
                    asyncio.run_coroutine_threadsafe(
                        ws.send(json.dumps(data)), self._loop)
                except Exception:
                    dead.add(ws)
            self._clients -= dead

    # ── Process pending WebSocket → ROS2 requests ──

    def _process_pending(self):
        # Auto grasp request
        with self._lock:
            do_auto_grasp = self._pending_auto_grasp
            self._pending_auto_grasp = False
            selection = self._pending_selection
            self._pending_selection = None
            demo_type = self._pending_demo
            self._pending_demo = None

        if do_auto_grasp:
            self._call_service("/auto_grasp/start", "std_srvs/srv/Trigger", {})

        if selection is not None:
            self._call_service("/grasp/select", "kio_teleop_openarm/srv/SelectGrasp",
                              {"obj_idx": selection["obj_idx"],
                               "grasp_idx": selection["grasp_idx"]})

        if demo_type:
            self._call_service("/demo/trigger", "kio_teleop_openarm/srv/DemoTrigger",
                              {"demo_type": demo_type})

    def _call_service(self, srv_name: str, srv_type: str, request_dict: dict):
        if srv_name not in self._clients_ready:
            try:
                if srv_type == "std_srvs/srv/Trigger":
                    from std_srvs.srv import Trigger
                    self._clients_ready[srv_name] = self.create_client(Trigger, srv_name)
                elif srv_type == "kio_teleop_openarm/srv/SelectGrasp":
                    from kio_teleop_openarm.srv import SelectGrasp
                    self._clients_ready[srv_name] = self.create_client(SelectGrasp, srv_name)
                elif srv_type == "kio_teleop_openarm/srv/DemoTrigger":
                    from kio_teleop_openarm.srv import DemoTrigger
                    self._clients_ready[srv_name] = self.create_client(DemoTrigger, srv_name)
                else:
                    self.get_logger().warn(f"Unknown service type: {srv_type}")
                    return
            except Exception as e:
                self.get_logger().warn(f"Cannot create client for {srv_name}: {e}")
                return

        client = self._clients_ready[srv_name]
        if not client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(f"Service {srv_name} not available")
            return

        if srv_type == "std_srvs/srv/Trigger":
            req = client.srv_type.Request()
        elif srv_type == "kio_teleop_openarm/srv/SelectGrasp":
            req = client.srv_type.Request()
            req.obj_idx = request_dict.get("obj_idx", 0)
            req.grasp_idx = request_dict.get("grasp_idx", 0)
        elif srv_type == "kio_teleop_openarm/srv/DemoTrigger":
            req = client.srv_type.Request()
            req.demo_type = request_dict.get("demo_type", "")

        future = client.call_async(req)
        self.get_logger().info(f"Called {srv_name}")


def main(args=None):
    rclpy.init(args=args)
    node = AppBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
