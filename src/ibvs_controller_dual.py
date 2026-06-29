#!/usr/bin/env python3
"""
Dual-robot improved IBVS controller for a combined Jacobian with columns:
    [NS1.x, NS1.y, NS1.Rz, NS2.x, NS2.y, NS2.Rz]

Control vector:
    delta_u = [dNS1x, dNS1y, dNS1Rz, dNS2x, dNS2y, dNS2Rz]

Characteristics:
- X and Y commands are expressed in each robot base frame.
- Rz is a rotation around the Z axis of each robot base frame.
- Z is held fixed for both robots.
- Uses a damped and unit-normalised pseudoinverse.
- Accumulates commands from the previous commanded target for each robot.
- Uses actual robot poses only for feedback and tracking-gate safety.
- Separates reachable and unreachable feature error.
- Saves numerical results and a PNG plot.

QTM MIGRATION NOTE:
  s is [x1, y1, x2, y2, x3, y3] -- world-frame X,Y in METERS for the
  3 QTM rigid bodies published by rod_perception.py.
  J_dual must have shape (6, 6), with columns:
      [NS1.x, NS1.y, NS1.Rz, NS2.x, NS2.y, NS2.Rz]
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray


# =============================================================================
# Controller parameters
# =============================================================================
RATE_HZ = 10.0
NUM_S_VALUES = 6
NUM_DOFS = 6

CONTROL_GAIN = 0.5
DAMPING = 0.05

# One triplet per robot: [x, y, Rz]
CONTROL_SCALES = np.array(
    [
        0.010,
        0.010,
        np.deg2rad(5.0),
        0.010,
        0.010,
        np.deg2rad(5.0),
    ],
    dtype=float,
)

DEADBAND_M = 0.005
REACHABLE_DEADBAND_M = 0.004
MAX_DURATION = 300.0
KEYPOINT_TIMEOUT = 0.5
POSE_TIMEOUT = 0.5

# Maximum increment per iteration.
# At 10 Hz:
#   0.0010 m/tick = 10 mm/s max in X/Y
#   0.0020 rad/tick = ~1.15 deg/s max in Rz
MAX_DU_X = 0.0010
MAX_DU_Y = 0.0010
MAX_DU_RZ = 0.0020
MAX_DU = np.array(
    [
        MAX_DU_X,
        MAX_DU_Y,
        MAX_DU_RZ,
        MAX_DU_X,
        MAX_DU_Y,
        MAX_DU_RZ,
    ],
    dtype=float,
)

# Tracking warnings. These are only warnings, not hard stops.
TRACKING_TOL_XY = 0.030
TRACKING_TOL_RZ = np.deg2rad(5.0)

# Stop generating new commands if either commanded target is too far ahead.
TRACKING_GATE_XY = 0.025
TRACKING_GATE_RZ = np.deg2rad(3.0)

POSE_CLAMP_X = 0.10
POSE_CLAMP_Y = 0.10
YAW_CLAMP = np.deg2rad(10.0)

Z_DRIFT_WARNING = 0.005
Z_DRIFT_ABORT = 0.020

ERROR_SIGN = +1

STAGNATION_WINDOW_SEC = 12.0
STAGNATION_MIN_IMPROVEMENT_M = 0.00010
STAGNATION_SAMPLES = max(
    2,
    int(STAGNATION_WINDOW_SEC * RATE_HZ),
)


# =============================================================================
# ROS topics
# =============================================================================
ROBOT_CONFIGS = {
    "NS1": {
        "target_pose": "/NS1/my_cartesian_impedance_controller/target_pose",
        "current_pose": "/NS1/franka_robot_state_broadcaster/current_pose",
        "frame_id": "NS1_base",
    },
    "NS2": {
        "target_pose": "/NS2/my_cartesian_impedance_controller/target_pose",
        "current_pose": "/NS2/franka_robot_state_broadcaster/current_pose",
        "frame_id": "NS2_base",
    },
}

# Keep this equal to your rod_perception.py output topic.
# If your dual setup publishes /rod_keypoints instead, change only this line.
KEYPOINT_TOPIC = "/NS2/rod_keypoints"


# =============================================================================
# Files
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

J_DUAL_PATH = os.path.join(SCRIPT_DIR, "J_dual.npy")
S_TARGET_PATH = os.path.join(SCRIPT_DIR, "s_target.npy")

RESULT_DATA_PATH = os.path.join(SCRIPT_DIR, "ibvs_result_dual_xy_rz.npz")
RESULT_PLOT_PATH = os.path.join(SCRIPT_DIR, "ibvs_result_dual_xy_rz.png")


@dataclass
class RobotRuntime:
    name: str
    target_topic: str
    current_topic: str
    frame_id: str

    actual_xyz: np.ndarray = None
    actual_q: np.ndarray = None
    last_pose_time: float = None

    cmd_xyz: np.ndarray = None
    cmd_q: np.ndarray = None
    start_xyz: np.ndarray = None
    start_q: np.ndarray = None
    cmd_yaw_offset: float = 0.0

    pub: object = None
    sub: object = None


class DualIBVSController(Node):
    DOF_NAMES = [
        "NS1.x",
        "NS1.y",
        "NS1.Rz",
        "NS2.x",
        "NS2.y",
        "NS2.Rz",
    ]

    ROBOT_ORDER = ["NS1", "NS2"]

    def __init__(self):
        super().__init__("ibvs_controller_dual")

        # =====================================================================
        # Load Jacobian
        # =====================================================================
        self.get_logger().info(f"Loading dual Jacobian from {J_DUAL_PATH}")
        if not os.path.isfile(J_DUAL_PATH):
            raise FileNotFoundError(
                f"Dual Jacobian not found: {J_DUAL_PATH}. "
                "Run the dual [NS1.x, NS1.y, NS1.Rz, NS2.x, NS2.y, NS2.Rz] estimator first."
            )

        self.J = np.asarray(np.load(J_DUAL_PATH), dtype=float)
        if self.J.shape != (NUM_S_VALUES, NUM_DOFS):
            raise ValueError(
                f"Expected J_dual shape ({NUM_S_VALUES}, {NUM_DOFS}) with columns "
                f"{self.DOF_NAMES}, but received {self.J.shape}"
            )
        if not np.all(np.isfinite(self.J)):
            raise ValueError("The dual Jacobian contains NaN or infinite values")

        self.rank_j = int(np.linalg.matrix_rank(self.J))
        self.cond_j = float(np.linalg.cond(self.J))
        self.get_logger().info(
            f"Dual Jacobian columns={self.DOF_NAMES}, "
            f"shape={self.J.shape}, rank={self.rank_j}, condition={self.cond_j:.2f}"
        )

        if self.rank_j < NUM_S_VALUES:
            self.get_logger().warn(
                f"Dual Jacobian rank is {self.rank_j}, expected {NUM_S_VALUES}. "
                "Controller can still run with damping, but some feature error may be unreachable."
            )
        if self.cond_j > 100.0:
            self.get_logger().warn(f"High dual Jacobian condition number: {self.cond_j:.2f}")

        self.error_projector = self.J @ np.linalg.pinv(self.J, rcond=1e-5)

        # =====================================================================
        # Load target
        # =====================================================================
        self.get_logger().info(f"Loading target from {S_TARGET_PATH}")
        if not os.path.isfile(S_TARGET_PATH):
            raise FileNotFoundError(f"Target file not found: {S_TARGET_PATH}")

        self.s_target = np.asarray(np.load(S_TARGET_PATH), dtype=float).reshape(-1)
        if self.s_target.size != NUM_S_VALUES:
            raise ValueError(
                f"Expected {NUM_S_VALUES} target feature values, got {self.s_target.size}"
            )
        if not np.all(np.isfinite(self.s_target)):
            raise ValueError("s_target contains NaN or infinite values")

        # =====================================================================
        # ROS communication
        # =====================================================================
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        measurement_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.robots: Dict[str, RobotRuntime] = {}
        for name in self.ROBOT_ORDER:
            cfg = ROBOT_CONFIGS[name]
            robot = RobotRuntime(
                name=name,
                target_topic=cfg["target_pose"],
                current_topic=cfg["current_pose"],
                frame_id=cfg["frame_id"],
            )
            robot.pub = self.create_publisher(PoseStamped, robot.target_topic, command_qos)
            robot.sub = self.create_subscription(
                PoseStamped,
                robot.current_topic,
                self._make_pose_cb(name),
                measurement_qos,
            )
            self.robots[name] = robot

        self.keypoint_sub = self.create_subscription(
            Float64MultiArray,
            KEYPOINT_TOPIC,
            self._keypoints_cb,
            measurement_qos,
        )

        # =====================================================================
        # Measurements
        # =====================================================================
        self.s_current = None
        self.last_keypoint_time = None

        # =====================================================================
        # Command state
        # =====================================================================
        self.start_time = None
        self.finished = False

        # =====================================================================
        # Logs
        # =====================================================================
        self.reachable_error_history = deque(maxlen=STAGNATION_SAMPLES)
        self.last_blocked_dofs = []

        self.log_time = []
        self.log_error = []
        self.log_rmse = []
        self.log_reachable_error = []
        self.log_unreachable_error = []
        self.log_du = []
        self.log_z_drift_ns1 = []
        self.log_z_drift_ns2 = []
        self.log_tracking_xy_ns1 = []
        self.log_tracking_xy_ns2 = []
        self.log_tracking_rz_ns1 = []
        self.log_tracking_rz_ns2 = []

        self.timer = self.create_timer(1.0 / RATE_HZ, self._control_loop)

        self.get_logger().info(
            "Dual improved IBVS ready: "
            "DoFs=[NS1.x, NS1.y, NS1.Rz, NS2.x, NS2.y, NS2.Rz], "
            f"feature vector size={NUM_S_VALUES}, Z fixed for both robots, "
            f"gain={CONTROL_GAIN}, damping={DAMPING}"
        )

    # =========================================================================
    # ROS callbacks
    # =========================================================================
    def _keypoints_cb(self, msg: Float64MultiArray):
        values = np.asarray(msg.data, dtype=float).reshape(-1)
        if values.size != NUM_S_VALUES or not np.all(np.isfinite(values)):
            self.get_logger().warn(
                "Ignoring invalid keypoint message "
                f"(expected size {NUM_S_VALUES}, got {values.size})",
                throttle_duration_sec=2.0,
            )
            return
        self.s_current = values
        self.last_keypoint_time = time.monotonic()

    def _make_pose_cb(self, robot_name: str):
        def _pose_cb(msg: PoseStamped):
            pose = msg.pose
            xyz = np.array(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ],
                dtype=float,
            )
            q = np.array(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                dtype=float,
            )
            q_norm = np.linalg.norm(q)
            if (
                not np.all(np.isfinite(xyz))
                or not np.all(np.isfinite(q))
                or q_norm < 1e-9
            ):
                return
            robot = self.robots[robot_name]
            robot.actual_xyz = xyz
            robot.actual_q = q / q_norm
            robot.last_pose_time = time.monotonic()

        return _pose_cb

    # =========================================================================
    # Pose helpers
    # =========================================================================
    @staticmethod
    def _relative_rotvec(current_q: np.ndarray, reference_q: np.ndarray) -> np.ndarray:
        return (R.from_quat(current_q) * R.from_quat(reference_q).inv()).as_rotvec()

    def _publish_robot_cmd(self, robot: RobotRuntime):
        if robot.cmd_xyz is None or robot.cmd_q is None:
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = robot.frame_id

        msg.pose.position.x = float(robot.cmd_xyz[0])
        msg.pose.position.y = float(robot.cmd_xyz[1])
        msg.pose.position.z = float(robot.cmd_xyz[2])

        msg.pose.orientation.x = float(robot.cmd_q[0])
        msg.pose.orientation.y = float(robot.cmd_q[1])
        msg.pose.orientation.z = float(robot.cmd_q[2])
        msg.pose.orientation.w = float(robot.cmd_q[3])

        robot.pub.publish(msg)

    def _publish_all_cmds(self):
        for robot in self.robots.values():
            self._publish_robot_cmd(robot)

    @staticmethod
    def _x_bounds(robot: RobotRuntime) -> Tuple[float, float]:
        return robot.start_xyz[0] - POSE_CLAMP_X, robot.start_xyz[0] + POSE_CLAMP_X

    @staticmethod
    def _y_bounds(robot: RobotRuntime) -> Tuple[float, float]:
        return robot.start_xyz[1] - POSE_CLAMP_Y, robot.start_xyz[1] + POSE_CLAMP_Y

    def _apply_xy_clamp(self, robot: RobotRuntime):
        x_low, x_high = self._x_bounds(robot)
        y_low, y_high = self._y_bounds(robot)

        requested_x = float(robot.cmd_xyz[0])
        requested_y = float(robot.cmd_xyz[1])

        robot.cmd_xyz[0] = float(np.clip(requested_x, x_low, x_high))
        robot.cmd_xyz[1] = float(np.clip(requested_y, y_low, y_high))
        robot.cmd_xyz[2] = robot.start_xyz[2]

        if abs(robot.cmd_xyz[0] - requested_x) > 1e-9:
            self.get_logger().warn(
                f"{robot.name} X clamp: requested={requested_x:.4f}, "
                f"range=[{x_low:.4f}, {x_high:.4f}]",
                throttle_duration_sec=1.0,
            )
        if abs(robot.cmd_xyz[1] - requested_y) > 1e-9:
            self.get_logger().warn(
                f"{robot.name} Y clamp: requested={requested_y:.4f}, "
                f"range=[{y_low:.4f}, {y_high:.4f}]",
                throttle_duration_sec=1.0,
            )

    # =========================================================================
    # DoF helpers
    # =========================================================================
    @staticmethod
    def _robot_name_from_dof(dof_index: int) -> str:
        if 0 <= dof_index <= 2:
            return "NS1"
        if 3 <= dof_index <= 5:
            return "NS2"
        raise ValueError(f"Invalid DoF index: {dof_index}")

    @staticmethod
    def _local_axis_from_dof(dof_index: int) -> int:
        return dof_index % 3

    def _dof_is_blocked(self, dof_index: int, increment: float) -> bool:
        robot_name = self._robot_name_from_dof(dof_index)
        local_axis = self._local_axis_from_dof(dof_index)
        robot = self.robots[robot_name]

        if robot.actual_xyz is None or robot.actual_q is None:
            return True

        if local_axis == 0:
            low, high = self._x_bounds(robot)
            current = robot.actual_xyz[0]
            return (
                (current <= low + 1e-5 and increment < 0.0)
                or (current >= high - 1e-5 and increment > 0.0)
            )

        if local_axis == 1:
            low, high = self._y_bounds(robot)
            current = robot.actual_xyz[1]
            return (
                (current <= low + 1e-5 and increment < 0.0)
                or (current >= high - 1e-5 and increment > 0.0)
            )

        if local_axis == 2:
            current_yaw = float(self._relative_rotvec(robot.actual_q, robot.start_q)[2])
            return (
                (current_yaw <= -YAW_CLAMP + 1e-5 and increment < 0.0)
                or (current_yaw >= YAW_CLAMP - 1e-5 and increment > 0.0)
            )

        raise ValueError(f"Invalid local DoF axis: {local_axis}")

    # =========================================================================
    # Damped normalised pseudoinverse
    # =========================================================================
    def _calculate_damped_command(self, feature_error: np.ndarray, available: List[int]) -> np.ndarray:
        J_available = self.J[:, available]
        scales = CONTROL_SCALES[available]
        scale_matrix = np.diag(scales)

        J_normalised = J_available @ scale_matrix
        hessian = (
            J_normalised.T @ J_normalised
            + (DAMPING ** 2) * np.eye(len(available), dtype=float)
        )
        gradient = J_normalised.T @ feature_error

        try:
            normalised_du = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            normalised_du = np.linalg.pinv(hessian, rcond=1e-6) @ gradient

        physical_du = CONTROL_GAIN * (scale_matrix @ normalised_du)
        return np.asarray(physical_du, dtype=float).reshape(-1)

    def _compute_command(self, feature_error: np.ndarray):
        available = list(range(NUM_DOFS))
        blocked = []

        for _ in range(NUM_DOFS + 1):
            full_du = np.zeros(NUM_DOFS, dtype=float)
            if not available:
                return full_du, [self.DOF_NAMES[index] for index in blocked]

            du_available = self._calculate_damped_command(feature_error, available)
            full_du[available] = du_available

            newly_blocked = [
                index
                for index in available
                if self._dof_is_blocked(index, full_du[index])
            ]

            if not newly_blocked:
                return full_du, [self.DOF_NAMES[index] for index in blocked]

            for index in newly_blocked:
                if index not in blocked:
                    blocked.append(index)

            available = [index for index in available if index not in newly_blocked]

        return np.zeros(NUM_DOFS, dtype=float), [self.DOF_NAMES[index] for index in blocked]

    # =========================================================================
    # Safety and state helpers
    # =========================================================================
    def _all_publishers_connected(self) -> bool:
        ok = True
        for robot in self.robots.values():
            if robot.pub.get_subscription_count() < 1:
                self.get_logger().warn(
                    f"Waiting for controller subscriber on {robot.target_topic}",
                    throttle_duration_sec=2.0,
                )
                ok = False
        return ok

    def _publisher_conflict_exists(self) -> bool:
        conflict = False
        for robot in self.robots.values():
            if self.count_publishers(robot.target_topic) > 1:
                self.get_logger().error(
                    f"Another target_pose publisher is active on {robot.target_topic}"
                )
                conflict = True
        return conflict

    def _all_poses_available_and_fresh(self, now: float) -> bool:
        ok = True
        for robot in self.robots.values():
            if robot.actual_xyz is None or robot.actual_q is None:
                self.get_logger().info(
                    f"Waiting for {robot.current_topic}",
                    throttle_duration_sec=2.0,
                )
                ok = False
                continue

            if robot.last_pose_time is None or now - robot.last_pose_time > POSE_TIMEOUT:
                self.get_logger().warn(
                    f"{robot.name} current pose is stale; holding commands",
                    throttle_duration_sec=1.0,
                )
                ok = False
        return ok

    def _commands_initialised(self) -> bool:
        return all(robot.cmd_xyz is not None and robot.cmd_q is not None for robot in self.robots.values())

    def _initialise_commands(self, now: float):
        for robot in self.robots.values():
            robot.cmd_xyz = robot.actual_xyz.copy()
            robot.cmd_q = robot.actual_q.copy()
            robot.start_xyz = robot.actual_xyz.copy()
            robot.start_q = robot.actual_q.copy()
            robot.cmd_yaw_offset = 0.0

        self.start_time = now

        lines = ["Dual command initialised:"]
        for robot in self.robots.values():
            lines.append(
                f"  {robot.name}: xyz={np.round(robot.cmd_xyz, 5)}, "
                f"q={np.round(robot.cmd_q, 6)}"
            )
        lines.append(f"  s_target={np.round(self.s_target, 5)} (m)")
        lines.append(f"  s_current={np.round(self.s_current, 5)} (m)")
        self.get_logger().info("\n".join(lines))

        self._publish_all_cmds()

    def _compute_tracking_and_z(self):
        data = {}
        abort_reason = None
        gate_blocked = False

        for robot in self.robots.values():
            z_drift = float(robot.actual_xyz[2] - robot.start_xyz[2])
            tracking_xy = float(np.linalg.norm(robot.actual_xyz[:2] - robot.cmd_xyz[:2]))
            tracking_rotvec = self._relative_rotvec(robot.cmd_q, robot.actual_q)
            tracking_rz = float(abs(tracking_rotvec[2]))

            data[robot.name] = {
                "z_drift": z_drift,
                "tracking_xy": tracking_xy,
                "tracking_rz": tracking_rz,
            }

            if abs(z_drift) > Z_DRIFT_WARNING:
                self.get_logger().warn(
                    f"{robot.name} actual Z drift={z_drift * 1000:.1f} mm",
                    throttle_duration_sec=1.0,
                )
            if abs(z_drift) > Z_DRIFT_ABORT:
                abort_reason = f"{robot.name} excessive Z drift"

            if tracking_xy > TRACKING_TOL_XY:
                self.get_logger().warn(
                    f"{robot.name} large XY tracking error={tracking_xy * 1000:.1f} mm",
                    throttle_duration_sec=1.0,
                )
            if tracking_rz > TRACKING_TOL_RZ:
                self.get_logger().warn(
                    f"{robot.name} large Rz tracking error={np.rad2deg(tracking_rz):.2f} deg",
                    throttle_duration_sec=1.0,
                )

            if tracking_xy > TRACKING_GATE_XY or tracking_rz > TRACKING_GATE_RZ:
                gate_blocked = True

        return data, abort_reason, gate_blocked

    # =========================================================================
    # Logging
    # =========================================================================
    def _append_logs(
        self,
        elapsed,
        error_norm,
        error_rmse,
        reachable_norm,
        unreachable_norm,
        applied_du,
        safety_data,
    ):
        self.log_time.append(float(elapsed))
        self.log_error.append(float(error_norm))
        self.log_rmse.append(float(error_rmse))
        self.log_reachable_error.append(float(reachable_norm))
        self.log_unreachable_error.append(float(unreachable_norm))
        self.log_du.append(np.asarray(applied_du, dtype=float).reshape(NUM_DOFS).tolist())

        self.log_z_drift_ns1.append(float(safety_data["NS1"]["z_drift"]))
        self.log_z_drift_ns2.append(float(safety_data["NS2"]["z_drift"]))
        self.log_tracking_xy_ns1.append(float(safety_data["NS1"]["tracking_xy"]))
        self.log_tracking_xy_ns2.append(float(safety_data["NS2"]["tracking_xy"]))
        self.log_tracking_rz_ns1.append(float(safety_data["NS1"]["tracking_rz"]))
        self.log_tracking_rz_ns2.append(float(safety_data["NS2"]["tracking_rz"]))

    # =========================================================================
    # Main loop
    # =========================================================================
    def _control_loop(self):
        if self.finished:
            return

        if self._publisher_conflict_exists():
            self._finish("publisher conflict")
            return

        if not self._all_publishers_connected():
            return

        if self.s_current is None:
            self.get_logger().info(
                f"Waiting for {KEYPOINT_TOPIC}",
                throttle_duration_sec=2.0,
            )
            return

        now = time.monotonic()

        if self.last_keypoint_time is None or now - self.last_keypoint_time > KEYPOINT_TIMEOUT:
            self.get_logger().warn(
                "Keypoints are stale; holding commands",
                throttle_duration_sec=1.0,
            )
            self._publish_all_cmds()
            return

        if not self._all_poses_available_and_fresh(now):
            self._publish_all_cmds()
            return

        if not self._commands_initialised():
            self._initialise_commands(now)
            return

        elapsed = now - self.start_time
        if elapsed > MAX_DURATION:
            self._finish("timeout")
            return

        # =====================================================================
        # Tracking and Z drift
        # =====================================================================
        safety_data, abort_reason, gate_blocked = self._compute_tracking_and_z()
        if abort_reason is not None:
            self.get_logger().error(
                "Z drift is too large for the dual [x, y, Rz] Jacobian"
            )
            self._finish(abort_reason)
            return

        # =====================================================================
        # Feature error
        # =====================================================================
        feature_error = ERROR_SIGN * (self.s_target - self.s_current)

        error_norm = float(np.linalg.norm(feature_error))
        error_rmse = float(np.sqrt(np.mean(feature_error ** 2)))

        reachable_error = self.error_projector @ feature_error
        unreachable_error = feature_error - reachable_error

        reachable_norm = float(np.linalg.norm(reachable_error))
        unreachable_norm = float(np.linalg.norm(unreachable_error))

        self.get_logger().info(
            f"error={error_norm * 1000:.2f} mm, "
            f"reachable={reachable_norm * 1000:.2f} mm, "
            f"unreachable={unreachable_norm * 1000:.2f} mm",
            throttle_duration_sec=1.0,
        )

        if error_norm < DEADBAND_M:
            self._append_logs(
                elapsed,
                error_norm,
                error_rmse,
                reachable_norm,
                unreachable_norm,
                np.zeros(NUM_DOFS, dtype=float),
                safety_data,
            )
            self.get_logger().info(f"Converged: ||e||={error_norm * 1000:.2f} mm")
            self._finish("convergence")
            return

        if reachable_norm < REACHABLE_DEADBAND_M:
            self._append_logs(
                elapsed,
                error_norm,
                error_rmse,
                reachable_norm,
                unreachable_norm,
                np.zeros(NUM_DOFS, dtype=float),
                safety_data,
            )
            self.get_logger().warn(
                "The reachable component of the error is already small. "
                "Remaining error cannot be corrected using the dual [x, y, Rz] control space."
            )
            self._finish("best reachable solution")
            return

        # =====================================================================
        # Wait for robot tracking
        # =====================================================================
        if gate_blocked:
            self.get_logger().warn(
                "Waiting for robot tracking: "
                f"NS1 XY={safety_data['NS1']['tracking_xy'] * 1000:.1f} mm, "
                f"NS1 Rz={np.rad2deg(safety_data['NS1']['tracking_rz']):.2f} deg, "
                f"NS2 XY={safety_data['NS2']['tracking_xy'] * 1000:.1f} mm, "
                f"NS2 Rz={np.rad2deg(safety_data['NS2']['tracking_rz']):.2f} deg",
                throttle_duration_sec=1.0,
            )
            self._append_logs(
                elapsed,
                error_norm,
                error_rmse,
                reachable_norm,
                unreachable_norm,
                np.zeros(NUM_DOFS, dtype=float),
                safety_data,
            )
            self._publish_all_cmds()
            return

        # =====================================================================
        # Stagnation detection
        # =====================================================================
        self.reachable_error_history.append(reachable_norm)
        if len(self.reachable_error_history) == STAGNATION_SAMPLES:
            improvement = self.reachable_error_history[0] - self.reachable_error_history[-1]
            if improvement < STAGNATION_MIN_IMPROVEMENT_M:
                self._append_logs(
                    elapsed,
                    error_norm,
                    error_rmse,
                    reachable_norm,
                    unreachable_norm,
                    np.zeros(NUM_DOFS, dtype=float),
                    safety_data,
                )
                self.get_logger().warn(
                    "Reachable error stagnated: "
                    f"improvement={improvement * 1000:.3f} mm "
                    f"over {STAGNATION_WINDOW_SEC:.1f} s"
                )
                self._finish("reachable-error stagnation")
                return

        # =====================================================================
        # Calculate command
        # =====================================================================
        du_raw, blocked_dofs = self._compute_command(feature_error)

        if blocked_dofs != self.last_blocked_dofs:
            if blocked_dofs:
                self.get_logger().warn(f"Saturated DoFs removed: {blocked_dofs}")
            self.last_blocked_dofs = blocked_dofs.copy()

        requested_du = np.clip(du_raw, -MAX_DU, MAX_DU)

        # =====================================================================
        # Generate next commands from previous commanded targets
        # =====================================================================
        applied_du = np.zeros(NUM_DOFS, dtype=float)

        for robot_name in self.ROBOT_ORDER:
            robot = self.robots[robot_name]
            offset = 0 if robot_name == "NS1" else 3

            requested_dx = float(requested_du[offset + 0])
            requested_dy = float(requested_du[offset + 1])
            requested_dyaw = float(requested_du[offset + 2])

            previous_cmd_xyz = robot.cmd_xyz.copy()
            previous_cmd_yaw = float(robot.cmd_yaw_offset)

            requested_total_yaw = float(
                np.clip(
                    previous_cmd_yaw + requested_dyaw,
                    -YAW_CLAMP,
                    YAW_CLAMP,
                )
            )

            robot.cmd_xyz = previous_cmd_xyz.copy()
            robot.cmd_xyz[0] += requested_dx
            robot.cmd_xyz[1] += requested_dy
            self._apply_xy_clamp(robot)

            base_rz = R.from_rotvec([0.0, 0.0, requested_total_yaw])
            robot.cmd_q = (base_rz * R.from_quat(robot.start_q)).as_quat()
            robot.cmd_yaw_offset = requested_total_yaw

            applied_du[offset + 0] = float(robot.cmd_xyz[0] - previous_cmd_xyz[0])
            applied_du[offset + 1] = float(robot.cmd_xyz[1] - previous_cmd_xyz[1])
            applied_du[offset + 2] = float(requested_total_yaw - previous_cmd_yaw)

        self._publish_all_cmds()

        self.get_logger().info(
            "applied_du=["
            f"NS1: {applied_du[0] * 1000:.3f} mm x, "
            f"{applied_du[1] * 1000:.3f} mm y, "
            f"{np.rad2deg(applied_du[2]):.4f} deg Rz | "
            f"NS2: {applied_du[3] * 1000:.3f} mm x, "
            f"{applied_du[4] * 1000:.3f} mm y, "
            f"{np.rad2deg(applied_du[5]):.4f} deg Rz] "
            f"blocked={blocked_dofs}"
        )

        self._append_logs(
            elapsed,
            error_norm,
            error_rmse,
            reachable_norm,
            unreachable_norm,
            applied_du,
            safety_data,
        )

    # =========================================================================
    # Save results
    # =========================================================================
    def _finish(self, reason: str):
        if self.finished:
            return

        self.finished = True
        self.timer.cancel()
        self._publish_all_cmds()
        self.get_logger().info(f"Stopping dual IBVS: {reason}")

        if not self.log_time:
            self.get_logger().warn(
                "No samples were recorded. No result plot was generated."
            )
            raise SystemExit

        time_data = np.asarray(self.log_time, dtype=float)
        error_data = np.asarray(self.log_error, dtype=float)
        rmse_data = np.asarray(self.log_rmse, dtype=float)
        reachable_data = np.asarray(self.log_reachable_error, dtype=float)
        unreachable_data = np.asarray(self.log_unreachable_error, dtype=float)
        du_data = np.asarray(self.log_du, dtype=float)

        z_drift_ns1_data = np.asarray(self.log_z_drift_ns1, dtype=float)
        z_drift_ns2_data = np.asarray(self.log_z_drift_ns2, dtype=float)
        tracking_xy_ns1_data = np.asarray(self.log_tracking_xy_ns1, dtype=float)
        tracking_xy_ns2_data = np.asarray(self.log_tracking_xy_ns2, dtype=float)
        tracking_rz_ns1_data = np.asarray(self.log_tracking_rz_ns1, dtype=float)
        tracking_rz_ns2_data = np.asarray(self.log_tracking_rz_ns2, dtype=float)

        np.savez(
            RESULT_DATA_PATH,
            reason=np.asarray([reason]),
            dof_names=np.asarray(self.DOF_NAMES),
            jacobian=self.J,
            jacobian_rank=np.asarray([self.rank_j]),
            jacobian_condition=np.asarray([self.cond_j]),
            time=time_data,
            error_norm=error_data,
            rmse=rmse_data,
            reachable_error=reachable_data,
            unreachable_error=unreachable_data,
            applied_du=du_data,
            z_drift_ns1=z_drift_ns1_data,
            z_drift_ns2=z_drift_ns2_data,
            tracking_xy_ns1=tracking_xy_ns1_data,
            tracking_xy_ns2=tracking_xy_ns2_data,
            tracking_rz_ns1=tracking_rz_ns1_data,
            tracking_rz_ns2=tracking_rz_ns2_data,
        )
        self.get_logger().info(f"Results saved to {RESULT_DATA_PATH}")

        figure, axes = plt.subplots(6, 1, figsize=(12, 19))

        axes[0].plot(time_data, error_data * 1000.0, label="Total error")
        axes[0].plot(time_data, reachable_data * 1000.0, label="Reachable error")
        axes[0].plot(time_data, unreachable_data * 1000.0, label="Unreachable error")
        axes[0].axhline(DEADBAND_M * 1000.0, linestyle="--", label="Total deadband")
        axes[0].axhline(
            REACHABLE_DEADBAND_M * 1000.0,
            linestyle=":",
            label="Reachable deadband",
        )
        axes[0].set_title(f"Dual [x, y, Rz] improved IBVS: {reason}")
        axes[0].set_ylabel("Error norm (mm)")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(time_data, rmse_data * 1000.0, label="RMSE")
        axes[1].set_ylabel("RMSE (mm)")
        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(time_data, du_data[:, 0] * 1000.0, label="NS1 dx (mm)")
        axes[2].plot(time_data, du_data[:, 1] * 1000.0, label="NS1 dy (mm)")
        axes[2].plot(time_data, np.rad2deg(du_data[:, 2]), label="NS1 dRz (deg)")
        axes[2].set_ylabel("NS1 increment")
        axes[2].grid(True)
        axes[2].legend()

        axes[3].plot(time_data, du_data[:, 3] * 1000.0, label="NS2 dx (mm)")
        axes[3].plot(time_data, du_data[:, 4] * 1000.0, label="NS2 dy (mm)")
        axes[3].plot(time_data, np.rad2deg(du_data[:, 5]), label="NS2 dRz (deg)")
        axes[3].set_ylabel("NS2 increment")
        axes[3].grid(True)
        axes[3].legend()

        axes[4].plot(time_data, z_drift_ns1_data * 1000.0, label="NS1 Z drift")
        axes[4].plot(time_data, z_drift_ns2_data * 1000.0, label="NS2 Z drift")
        axes[4].axhline(Z_DRIFT_WARNING * 1000.0, linestyle="--", label="Warning")
        axes[4].axhline(-Z_DRIFT_WARNING * 1000.0, linestyle="--")
        axes[4].axhline(Z_DRIFT_ABORT * 1000.0, linestyle=":", label="Abort")
        axes[4].axhline(-Z_DRIFT_ABORT * 1000.0, linestyle=":")
        axes[4].set_ylabel("Z drift (mm)")
        axes[4].grid(True)
        axes[4].legend()

        axes[5].plot(time_data, tracking_xy_ns1_data * 1000.0, label="NS1 XY tracking (mm)")
        axes[5].plot(time_data, tracking_xy_ns2_data * 1000.0, label="NS2 XY tracking (mm)")
        axes[5].plot(time_data, np.rad2deg(tracking_rz_ns1_data), label="NS1 Rz tracking (deg)")
        axes[5].plot(time_data, np.rad2deg(tracking_rz_ns2_data), label="NS2 Rz tracking (deg)")
        axes[5].axhline(TRACKING_GATE_XY * 1000.0, linestyle="--", label="XY command gate")
        axes[5].axhline(np.rad2deg(TRACKING_GATE_RZ), linestyle=":", label="Rz command gate")
        axes[5].set_ylabel("Tracking error")
        axes[5].set_xlabel("Time (s)")
        axes[5].grid(True)
        axes[5].legend()

        plt.tight_layout()
        plt.savefig(RESULT_PLOT_PATH, dpi=150)
        plt.close(figure)

        self.get_logger().info(f"Plot saved to {RESULT_PLOT_PATH}")
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = DualIBVSController()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        node.get_logger().warn("Dual IBVS cancelled by user")
        if not node.finished:
            try:
                node._finish("cancelled by user")
            except SystemExit:
                pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
