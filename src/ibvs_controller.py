#!/usr/bin/env python3
"""
Improved IBVS controller for a Jacobian with columns:
    [x, y, Rz]

Control vector:
    delta_u = [dx, dy, dRz]

Characteristics:
- X and Y are expressed in NS2_base.
- Rz is a rotation around the Z axis of NS2_base.
- Z is held fixed.
- Uses a damped and unit-normalised pseudoinverse.
- Accumulates commands from the previous commanded target.
- Uses the actual robot pose only for feedback and tracking-gate safety.
- Separates reachable and unreachable image error.
- Saves numerical results and a PNG plot.
--------------------------------------------------------------------------
QTM MIGRATION NOTE:
  s is now [x1, y1, x2, y2, x3, y3] -- world-frame X,Y in METERS for the
  3 QTM rigid bodies published by rod_perception.py.

  J must have shape (6, 3), with columns [x, y, Rz].
--------------------------------------------------------------------------
"""

import os
import time
from collections import deque

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

CONTROL_GAIN = 0.5

DAMPING = 0.05

CONTROL_SCALES = np.array(
    [
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

# Tracking warnings.
# These are only warnings, not hard stops.
TRACKING_TOL_XY = 0.030
TRACKING_TOL_RZ = np.deg2rad(5.0)

# Stop generating new commands only if the commanded target is too far
# ahead of the real robot.
#
# Before: 0.006 m = 6 mm, too strict.
# Your log shows the controller constantly stuck at ~6.1 mm.
# Now: 0.025 m = 25 mm, allowing the Franka impedance controller to follow.
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

TARGET_POSE_TOPIC = "/NS2/my_cartesian_impedance_controller/target_pose"
CURRENT_POSE_TOPIC = "/NS2/franka_robot_state_broadcaster/current_pose"
KEYPOINT_TOPIC = "/NS2/rod_keypoints"
FRAME_ID = "NS2_base"


# =============================================================================
# Files
# =============================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

J_HAT_PATH = os.path.join(
    SCRIPT_DIR,
    "J_hat.npy",
)

S_TARGET_PATH = os.path.join(
    SCRIPT_DIR,
    "s_target.npy",
)

RESULT_DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "ibvs_result_xy_rz.npz",
)

RESULT_PLOT_PATH = os.path.join(
    SCRIPT_DIR,
    "ibvs_result_xy_rz.png",
)


class IBVSController(Node):
    DOF_NAMES = ["x", "y", "Rz"]

    def __init__(self):
        super().__init__("ibvs_controller")

        # =====================================================================
        # Load Jacobian
        # =====================================================================

        self.get_logger().info(
            f"Loading Jacobian from {J_HAT_PATH}"
        )

        if not os.path.isfile(J_HAT_PATH):
            raise FileNotFoundError(
                f"Jacobian not found: {J_HAT_PATH}. "
                "Run the [x, y, Rz] estimator first."
            )

        self.J = np.asarray(
            np.load(J_HAT_PATH),
            dtype=float,
        )

        if self.J.shape != (NUM_S_VALUES, 3):
            raise ValueError(
                f"Expected J shape ({NUM_S_VALUES}, 3) with columns "
                f"[x, y, Rz], but received {self.J.shape}"
            )

        if not np.all(np.isfinite(self.J)):
            raise ValueError(
                "The Jacobian contains NaN or infinite values"
            )

        self.rank_j = int(
            np.linalg.matrix_rank(self.J)
        )

        self.cond_j = float(
            np.linalg.cond(self.J)
        )

        self.get_logger().info(
            f"Jacobian columns={self.DOF_NAMES}, "
            f"shape={self.J.shape}, "
            f"rank={self.rank_j}, "
            f"condition={self.cond_j:.2f}"
        )

        if self.rank_j < 3:
            raise ValueError(
                f"Jacobian rank must be 3, got {self.rank_j}"
            )

        if self.cond_j > 100.0:
            self.get_logger().warn(
                "High Jacobian condition number: "
                f"{self.cond_j:.2f}"
            )

        self.error_projector = (
            self.J
            @ np.linalg.pinv(
                self.J,
                rcond=1e-5,
            )
        )

        # =====================================================================
        # Load target
        # =====================================================================

        self.get_logger().info(
            f"Loading target from {S_TARGET_PATH}"
        )

        if not os.path.isfile(S_TARGET_PATH):
            raise FileNotFoundError(
                f"Target file not found: {S_TARGET_PATH}"
            )

        self.s_target = np.asarray(
            np.load(S_TARGET_PATH),
            dtype=float,
        ).reshape(-1)

        if self.s_target.size != NUM_S_VALUES:
            raise ValueError(
                f"Expected {NUM_S_VALUES} target feature values, "
                f"got {self.s_target.size}"
            )

        if not np.all(np.isfinite(self.s_target)):
            raise ValueError(
                "s_target contains NaN or infinite values"
            )

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

        self.pub = self.create_publisher(
            PoseStamped,
            TARGET_POSE_TOPIC,
            command_qos,
        )

        self.keypoint_sub = self.create_subscription(
            Float64MultiArray,
            KEYPOINT_TOPIC,
            self._keypoints_cb,
            measurement_qos,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            CURRENT_POSE_TOPIC,
            self._pose_cb,
            measurement_qos,
        )

        # =====================================================================
        # Measurements
        # =====================================================================

        self.s_current = None
        self.last_keypoint_time = None

        self.actual_xyz = None
        self.actual_q = None
        self.last_pose_time = None

        # =====================================================================
        # Command state
        # =====================================================================

        self.cmd_xyz = None
        self.cmd_q = None

        self.start_xyz = None
        self.start_q = None

        self.cmd_yaw_offset = 0.0

        self.start_time = None
        self.finished = False

        # =====================================================================
        # Logs
        # =====================================================================

        self.reachable_error_history = deque(
            maxlen=STAGNATION_SAMPLES
        )

        self.last_blocked_dofs = []

        self.log_time = []
        self.log_error = []
        self.log_rmse = []
        self.log_reachable_error = []
        self.log_unreachable_error = []
        self.log_du = []
        self.log_z_drift = []
        self.log_tracking_xy = []
        self.log_tracking_rz = []

        self.timer = self.create_timer(
            1.0 / RATE_HZ,
            self._control_loop,
        )

        self.get_logger().info(
            "Improved IBVS ready: "
            "DoFs=[x, y, Rz], "
            f"feature vector size={NUM_S_VALUES}, "
            "Z fixed, "
            f"gain={CONTROL_GAIN}, "
            f"damping={DAMPING}"
        )

    # =========================================================================
    # ROS callbacks
    # =========================================================================

    def _keypoints_cb(
        self,
        msg: Float64MultiArray,
    ):
        values = np.asarray(
            msg.data,
            dtype=float,
        ).reshape(-1)

        if (
            values.size != NUM_S_VALUES
            or not np.all(np.isfinite(values))
        ):
            self.get_logger().warn(
                "Ignoring invalid keypoint message "
                f"(expected size {NUM_S_VALUES}, "
                f"got {values.size})",
                throttle_duration_sec=2.0,
            )
            return

        self.s_current = values
        self.last_keypoint_time = time.monotonic()

    def _pose_cb(
        self,
        msg: PoseStamped,
    ):
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

        self.actual_xyz = xyz
        self.actual_q = q / q_norm
        self.last_pose_time = time.monotonic()

    # =========================================================================
    # Pose helpers
    # =========================================================================

    @staticmethod
    def _relative_rotvec(
        current_q: np.ndarray,
        reference_q: np.ndarray,
    ) -> np.ndarray:
        return (
            R.from_quat(current_q)
            * R.from_quat(reference_q).inv()
        ).as_rotvec()

    def _publish_cmd(self):
        if (
            self.cmd_xyz is None
            or self.cmd_q is None
        ):
            return

        msg = PoseStamped()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.header.frame_id = FRAME_ID

        msg.pose.position.x = float(
            self.cmd_xyz[0]
        )
        msg.pose.position.y = float(
            self.cmd_xyz[1]
        )
        msg.pose.position.z = float(
            self.cmd_xyz[2]
        )

        msg.pose.orientation.x = float(
            self.cmd_q[0]
        )
        msg.pose.orientation.y = float(
            self.cmd_q[1]
        )
        msg.pose.orientation.z = float(
            self.cmd_q[2]
        )
        msg.pose.orientation.w = float(
            self.cmd_q[3]
        )

        self.pub.publish(msg)

    def _x_bounds(self):
        return (
            self.start_xyz[0] - POSE_CLAMP_X,
            self.start_xyz[0] + POSE_CLAMP_X,
        )

    def _y_bounds(self):
        return (
            self.start_xyz[1] - POSE_CLAMP_Y,
            self.start_xyz[1] + POSE_CLAMP_Y,
        )

    def _apply_xy_clamp(self):
        x_low, x_high = self._x_bounds()
        y_low, y_high = self._y_bounds()

        requested_x = float(
            self.cmd_xyz[0]
        )

        requested_y = float(
            self.cmd_xyz[1]
        )

        self.cmd_xyz[0] = float(
            np.clip(
                requested_x,
                x_low,
                x_high,
            )
        )

        self.cmd_xyz[1] = float(
            np.clip(
                requested_y,
                y_low,
                y_high,
            )
        )

        self.cmd_xyz[2] = self.start_xyz[2]

        if abs(self.cmd_xyz[0] - requested_x) > 1e-9:
            self.get_logger().warn(
                f"X clamp: requested={requested_x:.4f}, "
                f"range=[{x_low:.4f}, {x_high:.4f}]",
                throttle_duration_sec=1.0,
            )

        if abs(self.cmd_xyz[1] - requested_y) > 1e-9:
            self.get_logger().warn(
                f"Y clamp: requested={requested_y:.4f}, "
                f"range=[{y_low:.4f}, {y_high:.4f}]",
                throttle_duration_sec=1.0,
            )

    # =========================================================================
    # Control limits
    # =========================================================================

    def _dof_is_blocked(
        self,
        dof_index: int,
        increment: float,
    ) -> bool:
        if dof_index == 0:
            low, high = self._x_bounds()
            current = self.actual_xyz[0]

            return (
                (
                    current <= low + 1e-5
                    and increment < 0.0
                )
                or
                (
                    current >= high - 1e-5
                    and increment > 0.0
                )
            )

        if dof_index == 1:
            low, high = self._y_bounds()
            current = self.actual_xyz[1]

            return (
                (
                    current <= low + 1e-5
                    and increment < 0.0
                )
                or
                (
                    current >= high - 1e-5
                    and increment > 0.0
                )
            )

        if dof_index == 2:
            current_yaw = float(
                self._relative_rotvec(
                    self.actual_q,
                    self.start_q,
                )[2]
            )

            return (
                (
                    current_yaw <= -YAW_CLAMP + 1e-5
                    and increment < 0.0
                )
                or
                (
                    current_yaw >= YAW_CLAMP - 1e-5
                    and increment > 0.0
                )
            )

        raise ValueError(
            f"Invalid DoF index: {dof_index}"
        )

    # =========================================================================
    # Damped normalised pseudoinverse
    # =========================================================================

    def _calculate_damped_command(
        self,
        feature_error: np.ndarray,
        available,
    ) -> np.ndarray:
        J_available = self.J[
            :,
            available,
        ]

        scales = CONTROL_SCALES[
            available
        ]

        scale_matrix = np.diag(
            scales
        )

        J_normalised = (
            J_available
            @ scale_matrix
        )

        hessian = (
            J_normalised.T
            @ J_normalised
            + (DAMPING ** 2)
            * np.eye(
                len(available),
                dtype=float,
            )
        )

        gradient = (
            J_normalised.T
            @ feature_error
        )

        try:
            normalised_du = np.linalg.solve(
                hessian,
                gradient,
            )
        except np.linalg.LinAlgError:
            normalised_du = (
                np.linalg.pinv(
                    hessian,
                    rcond=1e-6,
                )
                @ gradient
            )

        physical_du = (
            CONTROL_GAIN
            * (
                scale_matrix
                @ normalised_du
            )
        )

        return np.asarray(
            physical_du,
            dtype=float,
        ).reshape(-1)

    def _compute_command(
        self,
        feature_error: np.ndarray,
    ):
        available = [0, 1, 2]
        blocked = []

        for _ in range(4):
            full_du = np.zeros(
                3,
                dtype=float,
            )

            if not available:
                return (
                    full_du,
                    [
                        self.DOF_NAMES[index]
                        for index in blocked
                    ],
                )

            du_available = (
                self._calculate_damped_command(
                    feature_error,
                    available,
                )
            )

            full_du[available] = du_available

            newly_blocked = [
                index
                for index in available
                if self._dof_is_blocked(
                    index,
                    full_du[index],
                )
            ]

            if not newly_blocked:
                return (
                    full_du,
                    [
                        self.DOF_NAMES[index]
                        for index in blocked
                    ],
                )

            for index in newly_blocked:
                if index not in blocked:
                    blocked.append(index)

            available = [
                index
                for index in available
                if index not in newly_blocked
            ]

        return (
            np.zeros(
                3,
                dtype=float,
            ),
            [
                self.DOF_NAMES[index]
                for index in blocked
            ],
        )

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
        applied_dx,
        applied_dy,
        applied_dyaw,
        z_drift,
        tracking_xy,
        tracking_rz,
    ):
        self.log_time.append(
            float(elapsed)
        )

        self.log_error.append(
            float(error_norm)
        )

        self.log_rmse.append(
            float(error_rmse)
        )

        self.log_reachable_error.append(
            float(reachable_norm)
        )

        self.log_unreachable_error.append(
            float(unreachable_norm)
        )

        self.log_du.append(
            [
                float(applied_dx),
                float(applied_dy),
                float(applied_dyaw),
            ]
        )

        self.log_z_drift.append(
            float(z_drift)
        )

        self.log_tracking_xy.append(
            float(tracking_xy)
        )

        self.log_tracking_rz.append(
            float(tracking_rz)
        )

    # =========================================================================
    # Main loop
    # =========================================================================

    def _control_loop(self):
        if self.finished:
            return

        if (
            self.count_publishers(
                TARGET_POSE_TOPIC
            )
            > 1
        ):
            self.get_logger().error(
                "Another target_pose publisher is active"
            )
            self._finish(
                "publisher conflict"
            )
            return

        if self.pub.get_subscription_count() < 1:
            self.get_logger().warn(
                "Waiting for controller subscriber on "
                f"{TARGET_POSE_TOPIC}",
                throttle_duration_sec=2.0,
            )
            return

        if self.s_current is None:
            self.get_logger().info(
                f"Waiting for {KEYPOINT_TOPIC}",
                throttle_duration_sec=2.0,
            )
            return

        now = time.monotonic()

        if (
            self.last_keypoint_time is None
            or now - self.last_keypoint_time > KEYPOINT_TIMEOUT
        ):
            self.get_logger().warn(
                "Keypoints are stale; holding command",
                throttle_duration_sec=1.0,
            )
            self._publish_cmd()
            return

        if (
            self.actual_xyz is None
            or self.actual_q is None
        ):
            self.get_logger().info(
                f"Waiting for {CURRENT_POSE_TOPIC}",
                throttle_duration_sec=2.0,
            )
            return

        if (
            self.last_pose_time is None
            or now - self.last_pose_time > POSE_TIMEOUT
        ):
            self.get_logger().warn(
                "Current pose is stale; holding command",
                throttle_duration_sec=1.0,
            )
            self._publish_cmd()
            return

        # =====================================================================
        # Initialise
        # =====================================================================

        if self.cmd_xyz is None:
            self.cmd_xyz = (
                self.actual_xyz.copy()
            )

            self.cmd_q = (
                self.actual_q.copy()
            )

            self.start_xyz = (
                self.actual_xyz.copy()
            )

            self.start_q = (
                self.actual_q.copy()
            )

            self.cmd_yaw_offset = 0.0
            self.start_time = now

            self.get_logger().info(
                "Command initialised:\n"
                f"  xyz={np.round(self.cmd_xyz, 5)}\n"
                f"  q={np.round(self.cmd_q, 6)}\n"
                f"  s_target={np.round(self.s_target, 5)} (m)\n"
                f"  s_current={np.round(self.s_current, 5)} (m)"
            )

            self._publish_cmd()
            return

        elapsed = (
            now
            - self.start_time
        )

        if elapsed > MAX_DURATION:
            self._finish(
                "timeout"
            )
            return

        # =====================================================================
        # Tracking and Z drift
        # =====================================================================

        z_drift = float(
            self.actual_xyz[2]
            - self.start_xyz[2]
        )

        if abs(z_drift) > Z_DRIFT_WARNING:
            self.get_logger().warn(
                f"Actual Z drift={z_drift * 1000:.1f} mm",
                throttle_duration_sec=1.0,
            )

        if abs(z_drift) > Z_DRIFT_ABORT:
            self.get_logger().error(
                "Z drift is too large for the [x, y, Rz] Jacobian"
            )
            self._finish(
                "excessive Z drift"
            )
            return

        tracking_xy = float(
            np.linalg.norm(
                self.actual_xyz[:2]
                - self.cmd_xyz[:2]
            )
        )

        tracking_rotvec = (
            self._relative_rotvec(
                self.cmd_q,
                self.actual_q,
            )
        )

        tracking_rz = float(
            abs(tracking_rotvec[2])
        )

        if tracking_xy > TRACKING_TOL_XY:
            self.get_logger().warn(
                f"Large XY tracking error={tracking_xy * 1000:.1f} mm",
                throttle_duration_sec=1.0,
            )

        if tracking_rz > TRACKING_TOL_RZ:
            self.get_logger().warn(
                f"Large Rz tracking error={np.rad2deg(tracking_rz):.2f} deg",
                throttle_duration_sec=1.0,
            )

        # =====================================================================
        # Feature error
        # =====================================================================

        feature_error = (
            ERROR_SIGN
            * (
                self.s_target
                - self.s_current
            )
        )

        error_norm = float(
            np.linalg.norm(
                feature_error
            )
        )

        error_rmse = float(
            np.sqrt(
                np.mean(
                    feature_error ** 2
                )
            )
        )

        reachable_error = (
            self.error_projector
            @ feature_error
        )

        unreachable_error = (
            feature_error
            - reachable_error
        )

        reachable_norm = float(
            np.linalg.norm(
                reachable_error
            )
        )

        unreachable_norm = float(
            np.linalg.norm(
                unreachable_error
            )
        )

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
                0.0,
                0.0,
                0.0,
                z_drift,
                tracking_xy,
                tracking_rz,
            )

            self.get_logger().info(
                f"Converged: ||e||={error_norm * 1000:.2f} mm"
            )

            self._finish(
                "convergence"
            )
            return

        if reachable_norm < REACHABLE_DEADBAND_M:
            self._append_logs(
                elapsed,
                error_norm,
                error_rmse,
                reachable_norm,
                unreachable_norm,
                0.0,
                0.0,
                0.0,
                z_drift,
                tracking_xy,
                tracking_rz,
            )

            self.get_logger().warn(
                "The reachable component of the error is already small. "
                "Remaining error cannot be corrected using only [x, y, Rz]."
            )

            self._finish(
                "best reachable solution"
            )
            return

        # =====================================================================
        # Wait for robot tracking
        # =====================================================================

        if (
            tracking_xy > TRACKING_GATE_XY
            or tracking_rz > TRACKING_GATE_RZ
        ):
            self.get_logger().warn(
                "Waiting for robot tracking: "
                f"XY={tracking_xy * 1000:.1f} mm, "
                f"Rz={np.rad2deg(tracking_rz):.2f} deg",
                throttle_duration_sec=1.0,
            )

            self._append_logs(
                elapsed,
                error_norm,
                error_rmse,
                reachable_norm,
                unreachable_norm,
                0.0,
                0.0,
                0.0,
                z_drift,
                tracking_xy,
                tracking_rz,
            )

            self._publish_cmd()
            return

        # =====================================================================
        # Stagnation detection
        # =====================================================================

        self.reachable_error_history.append(
            reachable_norm
        )

        if (
            len(self.reachable_error_history)
            == STAGNATION_SAMPLES
        ):
            improvement = (
                self.reachable_error_history[0]
                - self.reachable_error_history[-1]
            )

            if improvement < STAGNATION_MIN_IMPROVEMENT_M:
                self._append_logs(
                    elapsed,
                    error_norm,
                    error_rmse,
                    reachable_norm,
                    unreachable_norm,
                    0.0,
                    0.0,
                    0.0,
                    z_drift,
                    tracking_xy,
                    tracking_rz,
                )

                self.get_logger().warn(
                    "Reachable error stagnated: "
                    f"improvement={improvement * 1000:.3f} mm "
                    f"over {STAGNATION_WINDOW_SEC:.1f} s"
                )

                self._finish(
                    "reachable-error stagnation"
                )
                return

        # =====================================================================
        # Calculate command
        # =====================================================================

        du_raw, blocked_dofs = (
            self._compute_command(
                feature_error
            )
        )

        if blocked_dofs != self.last_blocked_dofs:
            if blocked_dofs:
                self.get_logger().warn(
                    "Saturated DoFs removed: "
                    f"{blocked_dofs}"
                )

            self.last_blocked_dofs = (
                blocked_dofs.copy()
            )

        requested_dx = float(
            np.clip(
                du_raw[0],
                -MAX_DU_X,
                MAX_DU_X,
            )
        )

        requested_dy = float(
            np.clip(
                du_raw[1],
                -MAX_DU_Y,
                MAX_DU_Y,
            )
        )

        requested_dyaw = float(
            np.clip(
                du_raw[2],
                -MAX_DU_RZ,
                MAX_DU_RZ,
            )
        )

        # =====================================================================
        # Generate next command from previous commanded target
        # =====================================================================
        # THIS IS THE IMPORTANT FIX:
        #
        # Before, the code did:
        #
        #     self.cmd_xyz = self.actual_xyz.copy()
        #     self.cmd_xyz[0] += requested_dx
        #     self.cmd_xyz[1] += requested_dy
        #
        # That means every tick was anchored to the real measured robot pose.
        # If the Franka was slightly slow, the intended movement was forgotten.
        #
        # Now, the new command is anchored to the previous commanded target.
        # The actual robot pose is still used above for safety/tracking checks.

        previous_cmd_xyz = (
            self.cmd_xyz.copy()
        )

        previous_cmd_yaw = float(
            self.cmd_yaw_offset
        )

        requested_total_yaw = float(
            np.clip(
                previous_cmd_yaw
                + requested_dyaw,
                -YAW_CLAMP,
                YAW_CLAMP,
            )
        )

        self.cmd_xyz = (
            previous_cmd_xyz.copy()
        )

        self.cmd_xyz[0] += requested_dx
        self.cmd_xyz[1] += requested_dy

        self._apply_xy_clamp()

        base_rz = R.from_rotvec(
            [
                0.0,
                0.0,
                requested_total_yaw,
            ]
        )

        self.cmd_q = (
            base_rz
            * R.from_quat(
                self.start_q
            )
        ).as_quat()

        self.cmd_yaw_offset = (
            requested_total_yaw
        )

        applied_dx = float(
            self.cmd_xyz[0]
            - previous_cmd_xyz[0]
        )

        applied_dy = float(
            self.cmd_xyz[1]
            - previous_cmd_xyz[1]
        )

        applied_dyaw = float(
            requested_total_yaw
            - previous_cmd_yaw
        )

        self._publish_cmd()

        self.get_logger().info(
            f"applied_du=["
            f"{applied_dx * 1000:.3f} mm (x), "
            f"{applied_dy * 1000:.3f} mm (y), "
            f"{np.rad2deg(applied_dyaw):.4f} deg (Rz)] "
            f"blocked={blocked_dofs}"
        )

        self._append_logs(
            elapsed,
            error_norm,
            error_rmse,
            reachable_norm,
            unreachable_norm,
            applied_dx,
            applied_dy,
            applied_dyaw,
            z_drift,
            tracking_xy,
            tracking_rz,
        )

    # =========================================================================
    # Save results
    # =========================================================================

    def _finish(
        self,
        reason: str,
    ):
        if self.finished:
            return

        self.finished = True
        self.timer.cancel()

        self._publish_cmd()

        self.get_logger().info(
            f"Stopping IBVS: {reason}"
        )

        if not self.log_time:
            self.get_logger().warn(
                "No samples were recorded. "
                "No result plot was generated."
            )
            raise SystemExit

        time_data = np.asarray(
            self.log_time,
            dtype=float,
        )

        error_data = np.asarray(
            self.log_error,
            dtype=float,
        )

        rmse_data = np.asarray(
            self.log_rmse,
            dtype=float,
        )

        reachable_data = np.asarray(
            self.log_reachable_error,
            dtype=float,
        )

        unreachable_data = np.asarray(
            self.log_unreachable_error,
            dtype=float,
        )

        du_data = np.asarray(
            self.log_du,
            dtype=float,
        )

        z_drift_data = np.asarray(
            self.log_z_drift,
            dtype=float,
        )

        tracking_xy_data = np.asarray(
            self.log_tracking_xy,
            dtype=float,
        )

        tracking_rz_data = np.asarray(
            self.log_tracking_rz,
            dtype=float,
        )

        np.savez(
            RESULT_DATA_PATH,
            reason=np.asarray(
                [reason]
            ),
            dof_names=np.asarray(
                self.DOF_NAMES
            ),
            jacobian=self.J,
            jacobian_rank=np.asarray(
                [self.rank_j]
            ),
            jacobian_condition=np.asarray(
                [self.cond_j]
            ),
            time=time_data,
            error_norm=error_data,
            rmse=rmse_data,
            reachable_error=reachable_data,
            unreachable_error=unreachable_data,
            applied_du=du_data,
            z_drift=z_drift_data,
            tracking_xy=tracking_xy_data,
            tracking_rz=tracking_rz_data,
        )

        self.get_logger().info(
            f"Results saved to {RESULT_DATA_PATH}"
        )

        figure, axes = plt.subplots(
            5,
            1,
            figsize=(11, 16),
        )

        axes[0].plot(
            time_data,
            error_data * 1000.0,
            label="Total error",
        )

        axes[0].plot(
            time_data,
            reachable_data * 1000.0,
            label="Reachable error",
        )

        axes[0].plot(
            time_data,
            unreachable_data * 1000.0,
            label="Unreachable error",
        )

        axes[0].axhline(
            DEADBAND_M * 1000.0,
            linestyle="--",
            label="Total deadband",
        )

        axes[0].axhline(
            REACHABLE_DEADBAND_M * 1000.0,
            linestyle=":",
            label="Reachable deadband",
        )

        axes[0].set_title(
            f"[x, y, Rz] improved IBVS: {reason}"
        )

        axes[0].set_ylabel(
            "Error norm (mm)"
        )

        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(
            time_data,
            rmse_data * 1000.0,
            label="RMSE",
        )

        axes[1].set_ylabel(
            "RMSE (mm)"
        )

        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(
            time_data,
            du_data[:, 0] * 1000.0,
            label="dx (mm)",
        )

        axes[2].plot(
            time_data,
            du_data[:, 1] * 1000.0,
            label="dy (mm)",
        )

        axes[2].plot(
            time_data,
            np.rad2deg(
                du_data[:, 2]
            ),
            label="dRz (deg)",
        )

        axes[2].set_ylabel(
            "Applied increment"
        )

        axes[2].grid(True)
        axes[2].legend()

        axes[3].plot(
            time_data,
            z_drift_data * 1000.0,
            label="Actual Z drift",
        )

        axes[3].axhline(
            Z_DRIFT_WARNING * 1000.0,
            linestyle="--",
            label="Warning",
        )

        axes[3].axhline(
            -Z_DRIFT_WARNING * 1000.0,
            linestyle="--",
        )

        axes[3].axhline(
            Z_DRIFT_ABORT * 1000.0,
            linestyle=":",
            label="Abort",
        )

        axes[3].axhline(
            -Z_DRIFT_ABORT * 1000.0,
            linestyle=":",
        )

        axes[3].set_ylabel(
            "Z drift (mm)"
        )

        axes[3].grid(True)
        axes[3].legend()

        axes[4].plot(
            time_data,
            tracking_xy_data * 1000.0,
            label="XY tracking error (mm)",
        )

        axes[4].plot(
            time_data,
            np.rad2deg(
                tracking_rz_data
            ),
            label="Rz tracking error (deg)",
        )

        axes[4].axhline(
            TRACKING_GATE_XY * 1000.0,
            linestyle="--",
            label="XY command gate",
        )

        axes[4].axhline(
            np.rad2deg(
                TRACKING_GATE_RZ
            ),
            linestyle=":",
            label="Rz command gate",
        )

        axes[4].set_ylabel(
            "Tracking error"
        )

        axes[4].set_xlabel(
            "Time (s)"
        )

        axes[4].grid(True)
        axes[4].legend()

        plt.tight_layout()

        plt.savefig(
            RESULT_PLOT_PATH,
            dpi=150,
        )

        plt.close(
            figure
        )

        self.get_logger().info(
            f"Plot saved to {RESULT_PLOT_PATH}"
        )

        raise SystemExit


def main(args=None):
    rclpy.init(args=args)

    node = IBVSController()

    try:
        rclpy.spin(node)

    except SystemExit:
        pass

    except KeyboardInterrupt:
        node.get_logger().warn(
            "IBVS cancelled by user"
        )

        if not node.finished:
            try:
                node._finish(
                    "cancelled by user"
                )
            except SystemExit:
                pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()