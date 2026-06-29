#!/usr/bin/env python3
"""
Jacobian estimator for controlled DoFs:
    [x, y, Rz]
Estimated Jacobian:
    J = [J_x, J_y, J_Rz]
Shape:
    J.shape == (NUM_S_VALUES, 3)
The estimator commands:
    +x, -x
    +y, -y
    +Rz, -Rz
Rz is a rotation around the Z axis of the base frame.
Z translation is never commanded. Any measured Z displacement is logged
as drift and excluded from the controlled-motion vector.
The Jacobian is estimated from the ACTUAL measured displacements:
    delta_u = [dx, dy, dRz]
using least squares:
    J = delta_S @ pinv(delta_U)
--------------------------------------------------------------------------
QTM MIGRATION NOTE (replaces ArUco/RealSense pixel-corner features):
  s is now [x1, y1, x2, y2, x3, y3] -- world-frame X,Y (METERS) for the
  3 QTM rigid bodies on the deformable rod, published by rod_perception.py
  on /NS2/rod_keypoints. NUM_S_VALUES = 6 (was 8 for 4 ArUco corners in
  pixels).
  Frame alignment: QTM's world x/y axes are NOT assumed to be aligned
  with the FR3 base frame's x/y axes. This is fine -- the least-squares
  fit J = S @ pinv(U) absorbs whatever fixed rotation/offset exists
  between the QTM world frame and the robot base frame into the
  coefficients of J. This holds as long as the QTM rig and the robot
  base remain RIGIDLY FIXED relative to each other between calibration
  and later control. If the mocap cameras or robot mount get bumped,
  J must be re-estimated.
  Units: all feature-space tolerances below are now METERS, not pixels.
--------------------------------------------------------------------------
PERTURBATION SIZE NOTE:
  delta_t and delta_r defaults were widened (~3x) from the original
  calibration values so the six calibration moves are easy to see on
  video. 35 mm translation / ~8.6 deg rotation is still small relative
  to the FR3 workspace, but double-check against your actual rig
  (cable slack, rod tension, workspace limits) before running on
  hardware. Use --delta_t / --delta_r on the command line to override
  without editing this file.
--------------------------------------------------------------------------


python3 jacobian_estimator_code.py --delta_t 0.025 --delta_r 0.1 --home_xy_tolerance 0.012 --feature_return_tolerance 0.006 --max_return_extension 15
"""
import argparse
import os
import time
from collections import deque
from typing import Optional, Tuple
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

# Number of scalar values in the feature vector s.
# QTM: 3 rigid bodies x (x, y) = 6.  (Was 8 for 4 ArUco corners x (u, v).)
NUM_S_VALUES = 6

def smoothstep(alpha: float) -> float:
    """
    Smooth interpolation from 0 to 1.
    The position and orientation velocities are zero at the beginning
    and at the end of the movement.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)

class JacobianEstimatorNode(Node):
    BASE_FRAME = "NS2_base"
    TARGET_POSE_TOPIC = (
        "/NS2/my_cartesian_impedance_controller/target_pose"
    )
    CURRENT_POSE_TOPIC = (
        "/NS2/franka_robot_state_broadcaster/current_pose"
    )
    KEYPOINT_TOPIC = "/NS2/rod_keypoints"
    # Jacobian column order.
    DOF_NAMES = ["x", "y", "Rz"]
    def __init__(
        self,
        delta_t: float,
        delta_r: float,
        wait: float,
        output: str,
        move_duration: float,
        home_hold: float,
        average_samples: int,
        minimum_translation: float,
        minimum_rotation_deg: float,
        home_xy_tolerance: float,
        home_yaw_tolerance_deg: float,
        feature_return_tolerance: float,
        z_drift_warning: float,
        z_drift_abort: float,
        max_return_extension: float,
    ):
        super().__init__("jacobian_estimator")
        # --------------------------------------------------------------
        # Parameter validation
        # --------------------------------------------------------------
        if delta_t <= 0.0:
            raise ValueError("delta_t must be positive")
        if delta_r <= 0.0:
            raise ValueError("delta_r must be positive")
        if wait <= 0.0:
            raise ValueError("wait must be positive")
        if move_duration <= 0.0:
            raise ValueError("move_duration must be positive")
        if home_hold <= 0.0:
            raise ValueError("home_hold must be positive")
        if average_samples < 1:
            raise ValueError(
                "average_samples must be at least 1"
            )
        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------
        self.delta_t = float(delta_t)
        self.delta_r = float(delta_r)
        self.wait = float(wait)
        self.output = os.path.abspath(
            os.path.expanduser(output)
        )
        self.move_duration = float(move_duration)
        self.home_hold = float(home_hold)
        self.average_samples = int(average_samples)
        self.minimum_translation = float(
            minimum_translation
        )
        self.minimum_rotation = np.deg2rad(
            float(minimum_rotation_deg)
        )
        self.home_xy_tolerance = float(
            home_xy_tolerance
        )
        self.home_yaw_tolerance = np.deg2rad(
            float(home_yaw_tolerance_deg)
        )
        self.feature_return_tolerance = float(
            feature_return_tolerance
        )
        self.z_drift_warning = float(
            z_drift_warning
        )
        self.z_drift_abort = float(
            z_drift_abort
        )
        self.max_return_extension = float(
            max_return_extension
        )
        # --------------------------------------------------------------
        # ROS QoS
        # --------------------------------------------------------------
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
        # --------------------------------------------------------------
        # Publisher and subscribers
        # --------------------------------------------------------------
        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.TARGET_POSE_TOPIC,
            command_qos,
        )
        self.keypoint_sub = self.create_subscription(
            Float64MultiArray,
            self.KEYPOINT_TOPIC,
            self._keypoint_cb,
            measurement_qos,
        )
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.CURRENT_POSE_TOPIC,
            self._pose_cb,
            measurement_qos,
        )
        # --------------------------------------------------------------
        # Latest robot and image measurements
        # --------------------------------------------------------------
        self.actual_xyz: Optional[np.ndarray] = None
        self.actual_q: Optional[np.ndarray] = None
        self.last_pose_time: Optional[float] = None
        self.latest_s: Optional[np.ndarray] = None
        self.last_keypoint_time: Optional[float] = None
        self.keypoint_history = deque(
            maxlen=max(
                100,
                4 * self.average_samples,
            )
        )
        # --------------------------------------------------------------
        # Calibration home state
        # --------------------------------------------------------------
        self.home_xyz: Optional[np.ndarray] = None
        self.home_q: Optional[np.ndarray] = None
        self.s0: Optional[np.ndarray] = None
        # --------------------------------------------------------------
        # Jacobian samples
        # --------------------------------------------------------------
        # Each sample is:
        #
        #   [dx, dy, dRz]
        #
        self.delta_u_samples = []
        # Z translation is ignored but recorded.
        self.delta_z_samples = []
        # Feature (s) changes -- QTM world-frame x,y per marker, metres.
        self.delta_s_samples = []
        self.sample_labels = []
        # --------------------------------------------------------------
        # State machine
        # --------------------------------------------------------------
        self.state = "WAIT_INPUTS"
        self.phase = ""
        self.dof_index = 0
        self.sign_index = 0
        self.signs = [+1.0, -1.0]
        self.finished = False
        # --------------------------------------------------------------
        # Current smooth movement
        # --------------------------------------------------------------
        self.segment_start_xyz: Optional[np.ndarray] = None
        self.segment_start_q: Optional[np.ndarray] = None
        self.segment_target_xyz: Optional[np.ndarray] = None
        self.segment_target_q: Optional[np.ndarray] = None
        self.segment_start_time = 0.0
        self.settle_deadline = 0.0
        self.return_started = 0.0
        self.return_deadline = 0.0
        self.timer = self.create_timer(
            0.05,
            self._tick,
        )
        self.get_logger().info(
            "[x, y, Rz] Jacobian estimator ready:\n"
            f"  DoFs={self.DOF_NAMES}\n"
            f"  feature vector size={NUM_S_VALUES} "
            "(QTM world x,y per marker, metres)\n"
            f"  requested delta_t="
            f"{self.delta_t:.4f} m\n"
            f"  requested delta_r="
            f"{self.delta_r:.4f} rad "
            f"({np.rad2deg(self.delta_r):.2f} deg)\n"
            "  rotation axis=base-frame Z\n"
            "  Z translation disabled; "
            "measured Z drift will only be logged\n"
            f"  settle_wait={self.wait:.1f} s\n"
            f"  move_duration="
            f"{self.move_duration:.1f} s"
        )
    # ==================================================================
    # ROS callbacks
    # ==================================================================
    def _keypoint_cb(
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
        now = time.monotonic()
        self.latest_s = values
        self.last_keypoint_time = now
        self.keypoint_history.append(
            (
                now,
                values.copy(),
            )
        )
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
    # ==================================================================
    # Input validation
    # ==================================================================
    def _missing_inputs(self):
        """
        Return a list describing the ROS inputs that are missing.
        """
        now = time.monotonic()
        missing = []
        if (
            self.actual_xyz is None
            or self.actual_q is None
        ):
            missing.append(
                f"pose on {self.CURRENT_POSE_TOPIC}"
            )
        elif self.last_pose_time is None:
            missing.append("pose timestamp")
        elif now - self.last_pose_time >= 1.0:
            missing.append(
                "fresh pose; age="
                f"{now - self.last_pose_time:.2f} s"
            )
        if self.latest_s is None:
            missing.append(
                f"keypoints on {self.KEYPOINT_TOPIC}"
            )
        elif self.last_keypoint_time is None:
            missing.append("keypoint timestamp")
        elif now - self.last_keypoint_time >= 1.0:
            missing.append(
                "fresh keypoints; age="
                f"{now - self.last_keypoint_time:.2f} s"
            )
        subscription_count = (
            self.pose_pub.get_subscription_count()
        )
        if subscription_count < 1:
            missing.append(
                "controller subscriber on "
                f"{self.TARGET_POSE_TOPIC}"
            )
        return missing
    def _mean_recent_keypoints(
        self,
    ) -> Optional[np.ndarray]:
        if (
            len(self.keypoint_history)
            < self.average_samples
        ):
            return None
        samples = [
            sample
            for _, sample in list(
                self.keypoint_history
            )[-self.average_samples:]
        ]
        return np.mean(
            np.asarray(
                samples,
                dtype=float,
            ),
            axis=0,
        )
    # ==================================================================
    # Pose helpers
    # ==================================================================
    def _publish_pose(
        self,
        xyz: np.ndarray,
        q: np.ndarray,
    ):
        msg = PoseStamped()
        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )
        msg.header.frame_id = self.BASE_FRAME
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self.pose_pub.publish(msg)
    def _publish_home(self):
        if (
            self.home_xyz is not None
            and self.home_q is not None
        ):
            self._publish_pose(
                self.home_xyz,
                self.home_q,
            )
    @staticmethod
    def _relative_rotvec(
        current_q: np.ndarray,
        reference_q: np.ndarray,
    ) -> np.ndarray:
        """
        Rotation from reference_q to current_q expressed in the
        base/world frame.
        If:
            current = Rz_base * reference
        then:
            current * reference^-1 = Rz_base
        """
        return (
            R.from_quat(current_q)
            * R.from_quat(reference_q).inv()
        ).as_rotvec()
    def _actual_control_delta(
        self,
    ) -> np.ndarray:
        """
        Return the actual controlled displacement from home:
            [dx, dy, dRz]
        """
        dxyz = (
            self.actual_xyz
            - self.home_xyz
        )
        drot = self._relative_rotvec(
            self.actual_q,
            self.home_q,
        )
        return np.array(
            [
                dxyz[0],
                dxyz[1],
                drot[2],
            ],
            dtype=float,
        )
    def _ignored_z_delta(self) -> float:
        """
        Return the measured Z drift.
        Z is not part of the controlled vector.
        """
        return float(
            self.actual_xyz[2]
            - self.home_xyz[2]
        )
    def _controlled_home_error(
        self,
    ) -> Tuple[float, float, float]:
        """
        Return:
            xy_error_m
            yaw_error_rad
            z_drift_m
        """
        dxyz = (
            self.actual_xyz
            - self.home_xyz
        )
        drot = self._relative_rotvec(
            self.actual_q,
            self.home_q,
        )
        xy_error = float(
            np.linalg.norm(
                [
                    dxyz[0],
                    dxyz[1],
                ]
            )
        )
        yaw_error = float(
            abs(drot[2])
        )
        z_drift = float(
            dxyz[2]
        )
        return (
            xy_error,
            yaw_error,
            z_drift,
        )
    def _requested_target(
        self,
        dof_index: int,
        sign: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a target corresponding to:
            dof_index 0 -> X
            dof_index 1 -> Y
            dof_index 2 -> Rz around base-frame Z
        """
        xyz = self.home_xyz.copy()
        q = self.home_q.copy()
        if dof_index == 0:
            # Translation along base-frame X.
            xyz[0] += (
                sign * self.delta_t
            )
        elif dof_index == 1:
            # Translation along base-frame Y.
            xyz[1] += (
                sign * self.delta_t
            )
        elif dof_index == 2:
            # Rotation around base-frame Z.
            base_rz = R.from_rotvec(
                [
                    0.0,
                    0.0,
                    sign * self.delta_r,
                ]
            )
            # Pre-multiplication means that Rz is expressed
            # relative to the base/world frame.
            q = (
                base_rz
                * R.from_quat(self.home_q)
            ).as_quat()
        else:
            raise ValueError(
                f"Invalid DoF index: {dof_index}"
            )
        # Z is not controlled.
        xyz[2] = self.home_xyz[2]
        return xyz, q
    # ==================================================================
    # Smooth trajectory
    # ==================================================================
    def _begin_segment(
        self,
        start_xyz: np.ndarray,
        start_q: np.ndarray,
        target_xyz: np.ndarray,
        target_q: np.ndarray,
        next_phase: str,
    ):
        self.segment_start_xyz = (
            start_xyz.copy()
        )
        self.segment_start_q = (
            start_q.copy()
        )
        self.segment_target_xyz = (
            target_xyz.copy()
        )
        self.segment_target_q = (
            target_q.copy()
        )
        self.segment_start_time = (
            time.monotonic()
        )
        self.phase = next_phase
    def _execute_segment(self) -> bool:
        elapsed = (
            time.monotonic()
            - self.segment_start_time
        )
        alpha = smoothstep(
            elapsed / self.move_duration
        )
        xyz = (
            self.segment_start_xyz
            + alpha
            * (
                self.segment_target_xyz
                - self.segment_start_xyz
            )
        )
        # Z stays fixed throughout the complete movement.
        xyz[2] = self.home_xyz[2]
        start_rotation = R.from_quat(
            self.segment_start_q
        )
        target_rotation = R.from_quat(
            self.segment_target_q
        )
        relative_rotation = (
            target_rotation
            * start_rotation.inv()
        )
        q = (
            R.from_rotvec(
                alpha
                * relative_rotation.as_rotvec()
            )
            * start_rotation
        ).as_quat()
        self._publish_pose(
            xyz,
            q,
        )
        return (
            elapsed
            >= self.move_duration
        )
    # ==================================================================
    # Sample validation
    # ==================================================================
    def _validate_sample(
        self,
        actual_du: np.ndarray,
        dof_index: int,
        sign: float,
    ):
        value = float(
            actual_du[dof_index]
        )
        label = (
            f"{'+' if sign > 0 else '-'}"
            f"{self.DOF_NAMES[dof_index]}"
        )
        if dof_index < 2:
            # X or Y translation.
            minimum = self.minimum_translation
            scale = 1000.0
            unit = "mm"
        else:
            # Rz rotation.
            minimum = self.minimum_rotation
            scale = 180.0 / np.pi
            unit = "deg"
        if sign * value <= 0.0:
            self._abort(
                f"{label} actual motion has "
                "the wrong sign: "
                f"{value * scale:.3f} {unit}"
            )
            return
        if abs(value) < minimum:
            self._abort(
                f"{label} actual motion is too small: "
                f"{abs(value) * scale:.3f} {unit}; "
                f"minimum is "
                f"{minimum * scale:.3f} {unit}"
            )
    # ==================================================================
    # State machine
    # ==================================================================
    def _tick(self):
        if self.finished:
            return
        missing_inputs = self._missing_inputs()
        if missing_inputs:
            self.get_logger().warn(
                "Waiting for: "
                + "; ".join(missing_inputs),
                throttle_duration_sec=2.0,
            )
            return
        if self.count_publishers(
            self.TARGET_POSE_TOPIC
        ) > 1:
            self._abort(
                "Another target_pose publisher is active"
            )
            return
        # --------------------------------------------------------------
        # Capture the initial home pose
        # --------------------------------------------------------------
        if self.state == "WAIT_INPUTS":
            self.home_xyz = (
                self.actual_xyz.copy()
            )
            self.home_q = (
                self.actual_q.copy()
            )
            self.get_logger().info(
                "Home pose captured:\n"
                f"  xyz="
                f"{np.round(self.home_xyz, 5)}\n"
                f"  q="
                f"{np.round(self.home_q, 6)}"
            )
            self._publish_home()
            self.settle_deadline = (
                time.monotonic()
                + self.home_hold
            )
            self.state = "CAPTURE_HOME"
            return
        # --------------------------------------------------------------
        # Capture initial image features
        # --------------------------------------------------------------
        if self.state == "CAPTURE_HOME":
            self._publish_home()
            if (
                time.monotonic()
                < self.settle_deadline
            ):
                return
            s_home = (
                self._mean_recent_keypoints()
            )
            if s_home is None:
                return
            self.s0 = s_home.copy()
            self.get_logger().info(
                "Calibration state captured: "
                f"s0={np.round(self.s0, 5)} (m)"
            )
            self.state = "ESTIMATING"
            self.phase = "START_SAMPLE"
            self.dof_index = 0
            self.sign_index = 0
            return
        # --------------------------------------------------------------
        # Six calibration movements
        # --------------------------------------------------------------
        if self.state == "ESTIMATING":
            if (
                self.dof_index
                >= len(self.DOF_NAMES)
            ):
                self.state = "FINAL_HOME"
                self.return_started = (
                    time.monotonic()
                )
                self.return_deadline = (
                    self.return_started
                    + self.home_hold
                )
                self._publish_home()
                return
            dof_name = self.DOF_NAMES[
                self.dof_index
            ]
            sign = self.signs[
                self.sign_index
            ]
            sign_text = (
                "+"
                if sign > 0.0
                else "-"
            )
            # ----------------------------------------------------------
            # Start perturbation
            # ----------------------------------------------------------
            if self.phase == "START_SAMPLE":
                target_xyz, target_q = (
                    self._requested_target(
                        self.dof_index,
                        sign,
                    )
                )
                self.get_logger().info(
                    f"DoF {self.dof_index} "
                    f"({dof_name}): commanding "
                    f"{sign_text} perturbation"
                )
                # Start from the currently measured pose to avoid
                # an instantaneous command jump.
                self._begin_segment(
                    self.actual_xyz,
                    self.actual_q,
                    target_xyz,
                    target_q,
                    "EXEC_SAMPLE",
                )
                return
            # ----------------------------------------------------------
            # Execute perturbation
            # ----------------------------------------------------------
            if self.phase == "EXEC_SAMPLE":
                if self._execute_segment():
                    self.settle_deadline = (
                        time.monotonic()
                        + self.wait
                    )
                    self.phase = "SETTLE_SAMPLE"
                return
            # ----------------------------------------------------------
            # Capture perturbation
            # ----------------------------------------------------------
            if self.phase == "SETTLE_SAMPLE":
                self._publish_pose(
                    self.segment_target_xyz,
                    self.segment_target_q,
                )
                if (
                    time.monotonic()
                    < self.settle_deadline
                ):
                    return
                s_sample = (
                    self._mean_recent_keypoints()
                )
                if s_sample is None:
                    return
                actual_du = (
                    self._actual_control_delta()
                )
                ignored_dz = (
                    self._ignored_z_delta()
                )
                self._validate_sample(
                    actual_du,
                    self.dof_index,
                    sign,
                )
                if self.finished:
                    return
                if (
                    abs(ignored_dz)
                    > self.z_drift_warning
                ):
                    self.get_logger().warn(
                        f"Ignored Z drift during "
                        f"{sign_text}{dof_name}: "
                        f"{ignored_dz * 1000:.2f} mm"
                    )
                if (
                    abs(ignored_dz)
                    > self.z_drift_abort
                ):
                    self._abort(
                        f"Z drift during "
                        f"{sign_text}{dof_name} "
                        "is too large: "
                        f"{ignored_dz * 1000:.2f} mm"
                    )
                    return
                delta_s = (
                    s_sample
                    - self.s0
                )
                self.delta_u_samples.append(
                    actual_du.copy()
                )
                self.delta_z_samples.append(
                    ignored_dz
                )
                self.delta_s_samples.append(
                    delta_s.copy()
                )
                self.sample_labels.append(
                    f"{sign_text}{dof_name}"
                )
                self.get_logger().info(
                    f"Captured "
                    f"{sign_text}{dof_name}:\n"
                    "  actual controlled du=["
                    f"{actual_du[0] * 1000:.3f} "
                    "mm (x), "
                    f"{actual_du[1] * 1000:.3f} "
                    "mm (y), "
                    f"{np.rad2deg(actual_du[2]):.4f} "
                    "deg (Rz)]\n"
                    f"  ignored dz="
                    f"{ignored_dz * 1000:.3f} mm\n"
                    f"  ||delta_s||="
                    f"{np.linalg.norm(delta_s) * 1000:.3f} mm"
                )
                # Return from the actual measured pose.
                self._begin_segment(
                    self.actual_xyz,
                    self.actual_q,
                    self.home_xyz,
                    self.home_q,
                    "EXEC_RETURN",
                )
                return
            # ----------------------------------------------------------
            # Execute return movement
            # ----------------------------------------------------------
            if self.phase == "EXEC_RETURN":
                if self._execute_segment():
                    self.return_started = (
                        time.monotonic()
                    )
                    self.return_deadline = (
                        self.return_started
                        + self.wait
                    )
                    self.phase = "VERIFY_RETURN"
                return
            # ----------------------------------------------------------
            # Verify return to home
            # ----------------------------------------------------------
            if self.phase == "VERIFY_RETURN":
                self._publish_home()
                if (
                    time.monotonic()
                    < self.return_deadline
                ):
                    return
                (
                    xy_error,
                    yaw_error,
                    z_drift,
                ) = self._controlled_home_error()
                s_home = (
                    self._mean_recent_keypoints()
                )
                if s_home is None:
                    return
                image_error = float(
                    np.linalg.norm(
                        s_home
                        - self.s0
                    )
                )
                pose_ok = (
                    xy_error
                    <= self.home_xy_tolerance
                    and yaw_error
                    <= self.home_yaw_tolerance
                )
                image_ok = (
                    image_error
                    <= self.feature_return_tolerance
                )
                z_ok = (
                    abs(z_drift)
                    <= self.z_drift_abort
                )
                if (
                    pose_ok
                    and image_ok
                    and z_ok
                ):
                    self.get_logger().info(
                        f"Returned home after "
                        f"{sign_text}{dof_name}: "
                        f"xy_error="
                        f"{xy_error * 1000:.2f} mm, "
                        f"yaw_error="
                        f"{np.rad2deg(yaw_error):.3f} deg, "
                        f"ignored z_drift="
                        f"{z_drift * 1000:.2f} mm, "
                        f"feature_error="
                        f"{image_error * 1000:.2f} mm"
                    )
                    self.sign_index += 1
                    if (
                        self.sign_index
                        >= len(self.signs)
                    ):
                        self.sign_index = 0
                        self.dof_index += 1
                    self.phase = "START_SAMPLE"
                    return
                elapsed = (
                    time.monotonic()
                    - self.return_started
                )
                if (
                    elapsed
                    >= self.wait
                    + self.max_return_extension
                ):
                    self._abort(
                        f"Return failed after "
                        f"{sign_text}{dof_name}: "
                        f"xy_error="
                        f"{xy_error * 1000:.2f} mm, "
                        f"yaw_error="
                        f"{np.rad2deg(yaw_error):.3f} deg, "
                        f"ignored z_drift="
                        f"{z_drift * 1000:.2f} mm, "
                        f"feature_error="
                        f"{image_error * 1000:.2f} mm"
                    )
                    return
                self.get_logger().warn(
                    f"Waiting for return after "
                    f"{sign_text}{dof_name}: "
                    f"xy_error="
                    f"{xy_error * 1000:.2f} mm, "
                    f"yaw_error="
                    f"{np.rad2deg(yaw_error):.3f} deg, "
                    f"ignored z_drift="
                    f"{z_drift * 1000:.2f} mm, "
                    f"feature_error="
                    f"{image_error * 1000:.2f} mm",
                    throttle_duration_sec=1.0,
                )
                return
            self._abort(
                f"Unknown phase: {self.phase}"
            )
            return
        # --------------------------------------------------------------
        # Final return to home
        # --------------------------------------------------------------
        if self.state == "FINAL_HOME":
            self._publish_home()
            if (
                time.monotonic()
                < self.return_deadline
            ):
                return
            (
                xy_error,
                yaw_error,
                z_drift,
            ) = self._controlled_home_error()
            s_final = (
                self._mean_recent_keypoints()
            )
            if s_final is None:
                return
            image_error = float(
                np.linalg.norm(
                    s_final
                    - self.s0
                )
            )
            if (
                xy_error
                <= self.home_xy_tolerance
                and yaw_error
                <= self.home_yaw_tolerance
                and image_error
                <= self.feature_return_tolerance
                and abs(z_drift)
                <= self.z_drift_abort
            ):
                self._finish_success()
                return
            elapsed = (
                time.monotonic()
                - self.return_started
            )
            if (
                elapsed
                >= self.home_hold
                + self.max_return_extension
            ):
                self._abort(
                    "Final return failed: "
                    f"xy_error="
                    f"{xy_error * 1000:.2f} mm, "
                    f"yaw_error="
                    f"{np.rad2deg(yaw_error):.3f} deg, "
                    f"ignored z_drift="
                    f"{z_drift * 1000:.2f} mm, "
                    f"feature_error="
                    f"{image_error * 1000:.2f} mm"
                )
                return
            self.get_logger().warn(
                "Waiting for final return",
                throttle_duration_sec=1.0,
            )
            return
        self._abort(
            f"Unknown state: {self.state}"
        )
    # ==================================================================
    # Save Jacobian
    # ==================================================================
    def _finish_success(self):
        if len(self.delta_u_samples) != 6:
            self._abort(
                "Expected 6 samples, got "
                f"{len(self.delta_u_samples)}"
            )
            return
        # U shape:
        #
        #   3 x 6
        #
        # Rows:
        #
        #   dx
        #   dy
        #   dRz
        #
        U = np.asarray(
            self.delta_u_samples,
            dtype=float,
        ).T
        # S shape:
        #
        #   NUM_S_VALUES x 6
        #
        S = np.asarray(
            self.delta_s_samples,
            dtype=float,
        ).T
        DZ = np.asarray(
            self.delta_z_samples,
            dtype=float,
        )
        rank_u = int(
            np.linalg.matrix_rank(U)
        )
        cond_u = float(
            np.linalg.cond(U)
        )
        if rank_u < 3:
            self._abort(
                "Actual controlled-motion matrix "
                f"rank={rank_u}; expected 3. "
                "The robot did not produce three "
                "independent movements."
            )
            return
        J = (
            S
            @ np.linalg.pinv(
                U,
                rcond=1e-6,
            )
        )
        rank_j = int(
            np.linalg.matrix_rank(J)
        )
        cond_j = float(
            np.linalg.cond(J)
        )
        if rank_j < 3:
            self._abort(
                f"Estimated Jacobian rank={rank_j}; "
                "expected 3"
            )
            return
        output_dir = (
            os.path.dirname(self.output)
            or "."
        )
        os.makedirs(
            output_dir,
            exist_ok=True,
        )
        np.save(
            self.output,
            J,
        )
        root, extension = os.path.splitext(
            self.output
        )
        if extension.lower() == ".npy":
            j_hat_path = (
                root + "_hat.npy"
            )
        else:
            j_hat_path = (
                self.output + "_hat.npy"
            )
        np.save(
            j_hat_path,
            J,
        )
        s0_path = os.path.join(
            output_dir,
            "s0.npy",
        )
        np.save(
            s0_path,
            self.s0,
        )
        u0_path = os.path.join(
            output_dir,
            "u0.npy",
        )
        np.save(
            u0_path,
            np.concatenate(
                [
                    self.home_xyz,
                    self.home_q,
                ]
            ),
        )
        samples_path = os.path.join(
            output_dir,
            "jacobian_samples_xy_rz.npz",
        )
        np.savez(
            samples_path,
            labels=np.asarray(
                self.sample_labels
            ),
            delta_u_controlled=U,
            delta_z_ignored=DZ,
            delta_s=S,
        )
        self.get_logger().info(
            "\n"
            + "=" * 68
            + "\n[x, y, Rz] JACOBIAN ESTIMATION COMPLETE (QTM features)"
            + "\n"
            + "=" * 68
            + f"\nColumns: {self.DOF_NAMES}"
            + f"\nFeature vector size: {NUM_S_VALUES} "
            "(3 QTM bodies x,y, metres)"
            + f"\nU shape: {U.shape}, "
            f"rank={rank_u}, "
            f"condition={cond_u:.2f}"
            + f"\nJ shape: {J.shape}, "
            f"rank={rank_j}, "
            f"condition={cond_j:.2f}"
            + f"\nMaximum ignored |dz|: "
            f"{np.max(np.abs(DZ)) * 1000:.2f} mm"
            + f"\nJ saved to:       {self.output}"
            + f"\nJ_hat saved to:   {j_hat_path}"
            + f"\ns0 saved to:      {s0_path}"
            + f"\nu0 saved to:      {u0_path}"
            + f"\nsamples saved to: {samples_path}"
            + "\n"
            + "=" * 68
        )
        if cond_u > 100.0:
            self.get_logger().warn(
                "Controlled-motion matrix is "
                "poorly conditioned"
            )
        if cond_j > 100.0:
            self.get_logger().warn(
                "Feature Jacobian is poorly conditioned"
            )
        if (
            np.max(np.abs(DZ))
            > self.z_drift_warning
        ):
            self.get_logger().warn(
                "Significant Z coupling was ignored. "
                "Use targets recorded at approximately "
                "the same Z position."
            )
        self.finished = True
        self.timer.cancel()
        self._publish_home()
        raise SystemExit
    # ==================================================================
    # Abort
    # ==================================================================
    def _abort(
        self,
        reason: str,
    ):
        if self.finished:
            return
        self.finished = True
        self.timer.cancel()
        if (
            self.home_xyz is not None
            and self.home_q is not None
        ):
            for _ in range(10):
                self._publish_home()
                time.sleep(0.02)
        self.get_logger().error(
            reason
            + ". No Jacobian was saved."
        )
        raise SystemExit

def main(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Feature-space Jacobian estimator "
            "for DoFs [x, y, Rz] using QTM mocap features"
        )
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        default=0.035,
        help=(
            "Requested X/Y translation "
            "perturbation in metres "
            "(widened from 0.012 m so moves are visible on video)"
        ),
    )
    parser.add_argument(
        "--delta_r",
        type=float,
        default=0.15,
        help=(
            "Requested Rz perturbation "
            "around base-frame Z in radians "
            "(widened from 0.06 rad so rotation is visible on video)"
        ),
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=5.0,
        help=(
            "Settle time after each "
            "perturbation in seconds"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="J.npy",
        help="Output path for J.npy",
    )
    parser.add_argument(
        "--move_duration",
        type=float,
        default=5.0,
        help=(
            "Duration of each smooth "
            "movement in seconds"
        ),
    )
    parser.add_argument(
        "--home_hold",
        type=float,
        default=3.0,
        help=(
            "Time to hold the home pose "
            "at the start and end"
        ),
    )
    parser.add_argument(
        "--average_samples",
        type=int,
        default=15,
        help=(
            "Number of recent keypoint "
            "samples to average"
        ),
    )
    parser.add_argument(
        "--minimum_translation",
        type=float,
        default=0.0002,
        help=(
            "Minimum accepted actual X or Y "
            "movement in metres"
        ),
    )
    parser.add_argument(
        "--minimum_rotation_deg",
        type=float,
        default=0.05,
        help=(
            "Minimum accepted actual Rz "
            "movement in degrees"
        ),
    )
    parser.add_argument(
        "--home_xy_tolerance",
        type=float,
        default=0.006,
        help=(
            "Maximum accepted return XY "
            "position error in metres"
        ),
    )
    parser.add_argument(
        "--home_yaw_tolerance_deg",
        type=float,
        default=2.0,
        help=(
            "Maximum accepted return Rz "
            "error in degrees"
        ),
    )
    parser.add_argument(
        "--feature_return_tolerance",
        type=float,
        default=0.004,
        help=(
            "Maximum accepted feature-space return "
            "error norm in METRES "
            "(QTM x,y per marker; was pixels for ArUco)"
        ),
    )
    parser.add_argument(
        "--z_drift_warning",
        type=float,
        default=0.005,
        help=(
            "Z drift warning threshold "
            "in metres"
        ),
    )
    parser.add_argument(
        "--z_drift_abort",
        type=float,
        default=0.020,
        help=(
            "Z drift abort threshold "
            "in metres"
        ),
    )
    parser.add_argument(
        "--max_return_extension",
        type=float,
        default=8.0,
        help=(
            "Extra time allowed for "
            "return-to-home in seconds"
        ),
    )
    parsed, _ = parser.parse_known_args()
    rclpy.init(args=args)
    node = JacobianEstimatorNode(
        delta_t=parsed.delta_t,
        delta_r=parsed.delta_r,
        wait=parsed.wait,
        output=parsed.output,
        move_duration=parsed.move_duration,
        home_hold=parsed.home_hold,
        average_samples=parsed.average_samples,
        minimum_translation=(
            parsed.minimum_translation
        ),
        minimum_rotation_deg=(
            parsed.minimum_rotation_deg
        ),
        home_xy_tolerance=(
            parsed.home_xy_tolerance
        ),
        home_yaw_tolerance_deg=(
            parsed.home_yaw_tolerance_deg
        ),
        feature_return_tolerance=(
            parsed.feature_return_tolerance
        ),
        z_drift_warning=(
            parsed.z_drift_warning
        ),
        z_drift_abort=(
            parsed.z_drift_abort
        ),
        max_return_extension=(
            parsed.max_return_extension
        ),
    )
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        node.get_logger().warn(
            "Jacobian estimation cancelled"
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()