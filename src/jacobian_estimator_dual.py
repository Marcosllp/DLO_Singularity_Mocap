#!/usr/bin/env python3
"""
Dual-robot Jacobian estimator for controlled DoFs:

    [NS1.x, NS1.y, NS1.Rz, NS2.x, NS2.y, NS2.Rz]

Estimated combined Jacobian:

    J = [J_NS1x, J_NS1y, J_NS1Rz, J_NS2x, J_NS2y, J_NS2Rz]

Shape:

    J.shape == (NUM_S_VALUES, 6)   # (6, 6) for QTM features

Both robots act on the SAME deformable rod, observed through the SAME
shared feature vector s (3 QTM rigid bodies x,y in metres). The combined
Jacobian captures how each robot's individual motion affects the shared
rod shape.

--------------------------------------------------------------------------
WHY THE 12 CALIBRATION MOVES ARE SEQUENTIAL, NOT SIMULTANEOUS:

  To separate "what does NS1.x alone do to s" from "what does NS2.x alone
  do to s," each calibration sample must isolate ONE robot's motion at a
  time. If both robots moved together on every sample, every observation
  would only ever show the *combined* effect, which is mathematically
  underdetermined -- you cannot solve for 6 independent Jacobian columns
  from samples that never vary the two robots' motions independently.

  So: only one robot physically displaces per move (12 moves total, 6
  DoFs x 2 signs). HOWEVER, both robots' state machines, home-holding
  loops, and publishers run concurrently throughout the whole script --
  the idle robot continuously republishes its own home pose every tick
  rather than sitting inert. On video this reads as one continuous
  two-robot sequence with both arms live the entire time, while each
  individual sample stays a clean, separable, single-robot measurement.

  Calibration order (alternating by robot for a continuous two-arm feel):

    NS1.x+   NS1.x-   NS2.x+   NS2.x-
    NS1.y+   NS1.y-   NS2.y+   NS2.y-
    NS1.Rz+  NS1.Rz-  NS2.Rz+  NS2.Rz-

--------------------------------------------------------------------------
QTM FEATURES (unchanged from single-robot version):

  s = [x1, y1, x2, y2, x3, y3] -- world-frame X,Y (METRES) for the 3 QTM
  rigid bodies on the deformable rod, published by rod_perception.py on
  /rod_keypoints. NUM_S_VALUES = 6.

  Frame alignment: QTM's world x/y axes are NOT assumed to be aligned
  with either robot's base frame. The least-squares fit J = S @ pinv(U)
  absorbs whatever fixed rotation/offset exists between the QTM world
  frame and each robot base frame into the corresponding columns of J.
  This holds as long as the QTM rig and both robot bases remain RIGIDLY
  FIXED relative to each other between calibration and later control.

--------------------------------------------------------------------------
PERTURBATION SIZE:

  Same widened defaults as the single-robot version (delta_t=0.035 m,
  delta_r=0.15 rad) so moves are visible on video. Override with
  --delta_t / --delta_r if your rod/cabling can't tolerate this at both
  end-effectors simultaneously being active in the same run.

--------------------------------------------------------------------------
EXAMPLE:

  
python3 jacobian_estimator_dual.py --delta_t 0.025 --delta_r 0.1 --home_xy_tolerance 0.012 --feature_return_tolerance 0.006 --max_return_extension 15
--------------------------------------------------------------------------
"""
import argparse
import os
import time
from collections import deque
from dataclasses import dataclass, field
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

# Number of scalar values in the shared feature vector s.
# QTM: 3 rigid bodies x (x, y) = 6.
NUM_S_VALUES = 6

# Combined DoF order for the 6x6 Jacobian.
ROBOT_NAMES = ["NS1", "NS2"]
DOF_NAMES_PER_ROBOT = ["x", "y", "Rz"]

# Full column order: NS1.x, NS1.y, NS1.Rz, NS2.x, NS2.y, NS2.Rz
COMBINED_DOF_LABELS = [
    f"{robot}.{dof}"
    for robot in ROBOT_NAMES
    for dof in DOF_NAMES_PER_ROBOT
]


def smoothstep(alpha: float) -> float:
    """
    Smooth interpolation from 0 to 1.

    The position and orientation velocities are zero at the beginning
    and at the end of the movement.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


@dataclass
class RobotState:
    """
    Per-robot ROS interfaces and pose state. One instance per arm
    (NS1, NS2). Both instances stay "live" for the whole run: each
    robot continuously republishes its own home pose every tick,
    whether or not it is the one actively perturbing.
    """

    name: str
    base_frame: str
    target_pose_topic: str
    current_pose_topic: str

    pose_pub: object = None
    pose_sub: object = None

    actual_xyz: Optional[np.ndarray] = None
    actual_q: Optional[np.ndarray] = None
    last_pose_time: Optional[float] = None

    home_xyz: Optional[np.ndarray] = None
    home_q: Optional[np.ndarray] = None

    # Active calibration segment (only populated while this robot is
    # the one currently perturbing).
    segment_start_xyz: Optional[np.ndarray] = None
    segment_start_q: Optional[np.ndarray] = None
    segment_target_xyz: Optional[np.ndarray] = None
    segment_target_q: Optional[np.ndarray] = None
    segment_start_time: float = 0.0


class DualJacobianEstimatorNode(Node):

    KEYPOINT_TOPIC = "/NS2/rod_keypoints"

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
        keypoint_topic: str,
    ):
        super().__init__("jacobian_estimator_dual")

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
            raise ValueError("average_samples must be at least 1")

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------
        self.delta_t = float(delta_t)
        self.delta_r = float(delta_r)
        self.wait = float(wait)
        self.output = os.path.abspath(os.path.expanduser(output))
        self.move_duration = float(move_duration)
        self.home_hold = float(home_hold)
        self.average_samples = int(average_samples)
        self.minimum_translation = float(minimum_translation)
        self.minimum_rotation = np.deg2rad(float(minimum_rotation_deg))
        self.home_xy_tolerance = float(home_xy_tolerance)
        self.home_yaw_tolerance = np.deg2rad(float(home_yaw_tolerance_deg))
        self.feature_return_tolerance = float(feature_return_tolerance)
        self.z_drift_warning = float(z_drift_warning)
        self.z_drift_abort = float(z_drift_abort)
        self.max_return_extension = float(max_return_extension)
        self.KEYPOINT_TOPIC = keypoint_topic

        # --------------------------------------------------------------
        # ROS QoS (BEST_EFFORT throughout, matching rod_perception.py
        # and the franka pose broadcasters -- see prior single-robot
        # QoS-mismatch debugging in this project).
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
        # Per-robot state (NS1, NS2)
        # --------------------------------------------------------------
        self.robots = {
            "NS1": RobotState(
                name="NS1",
                base_frame="NS1_base",
                target_pose_topic=(
                    "/NS1/my_cartesian_impedance_controller/target_pose"
                ),
                current_pose_topic=(
                    "/NS1/franka_robot_state_broadcaster/current_pose"
                ),
            ),
            "NS2": RobotState(
                name="NS2",
                base_frame="NS2_base",
                target_pose_topic=(
                    "/NS2/my_cartesian_impedance_controller/target_pose"
                ),
                current_pose_topic=(
                    "/NS2/franka_robot_state_broadcaster/current_pose"
                ),
            ),
        }

        for robot in self.robots.values():
            robot.pose_pub = self.create_publisher(
                PoseStamped,
                robot.target_pose_topic,
                command_qos,
            )
            robot.pose_sub = self.create_subscription(
                PoseStamped,
                robot.current_pose_topic,
                self._make_pose_cb(robot.name),
                measurement_qos,
            )

        # --------------------------------------------------------------
        # Shared keypoint (rod feature) subscription
        # --------------------------------------------------------------
        self.latest_s: Optional[np.ndarray] = None
        self.last_keypoint_time: Optional[float] = None
        self.keypoint_history = deque(
            maxlen=max(100, 4 * self.average_samples)
        )

        self.keypoint_sub = self.create_subscription(
            Float64MultiArray,
            self.KEYPOINT_TOPIC,
            self._keypoint_cb,
            measurement_qos,
        )

        # --------------------------------------------------------------
        # Calibration home state (shared s0)
        # --------------------------------------------------------------
        self.s0: Optional[np.ndarray] = None

        # --------------------------------------------------------------
        # Jacobian samples
        # --------------------------------------------------------------
        # Each sample is the full 6-vector:
        #   [dNS1x, dNS1y, dNS1Rz, dNS2x, dNS2y, dNS2Rz]
        # with the four non-perturbed entries close to zero.
        self.delta_u_samples = []
        self.delta_z_samples = []  # dict per sample: {"NS1": dz, "NS2": dz}
        self.delta_s_samples = []
        self.sample_labels = []

        # --------------------------------------------------------------
        # Calibration plan: 12 moves, alternating robot per DoF pair.
        # Each entry: (robot_name, dof_index_within_robot, sign)
        # --------------------------------------------------------------
        self.plan = []
        for dof_index in range(3):
            for robot_name in ROBOT_NAMES:
                for sign in (+1.0, -1.0):
                    self.plan.append((robot_name, dof_index, sign))
        self.plan_index = 0

        # --------------------------------------------------------------
        # State machine
        # --------------------------------------------------------------
        self.state = "WAIT_INPUTS"
        self.phase = ""
        self.finished = False

        self.settle_deadline = 0.0
        self.return_started = 0.0
        self.return_deadline = 0.0

        self.timer = self.create_timer(0.05, self._tick)

        self.get_logger().info(
            "Dual-robot [x, y, Rz] x2 Jacobian estimator ready:\n"
            f"  robots={ROBOT_NAMES}\n"
            f"  combined DoFs={COMBINED_DOF_LABELS}\n"
            f"  feature vector size={NUM_S_VALUES} "
            "(QTM world x,y per marker, metres)\n"
            f"  requested delta_t={self.delta_t:.4f} m\n"
            f"  requested delta_r={self.delta_r:.4f} rad "
            f"({np.rad2deg(self.delta_r):.2f} deg)\n"
            "  rotation axis=base-frame Z (per robot)\n"
            "  Z translation disabled per robot; "
            "measured Z drift only logged\n"
            f"  settle_wait={self.wait:.1f} s\n"
            f"  move_duration={self.move_duration:.1f} s\n"
            f"  total calibration moves={len(self.plan)} "
            "(6 DoFs x 2 signs, one robot moves at a time, "
            "both robots' home-hold loops run continuously)"
        )

    # ==================================================================
    # ROS callbacks
    # ==================================================================

    def _make_pose_cb(self, robot_name: str):
        def _cb(msg: PoseStamped):
            robot = self.robots[robot_name]
            pose = msg.pose
            xyz = np.array(
                [pose.position.x, pose.position.y, pose.position.z],
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
            robot.actual_xyz = xyz
            robot.actual_q = q / q_norm
            robot.last_pose_time = time.monotonic()

        return _cb

    def _keypoint_cb(self, msg: Float64MultiArray):
        values = np.asarray(msg.data, dtype=float).reshape(-1)
        if values.size != NUM_S_VALUES or not np.all(np.isfinite(values)):
            self.get_logger().warn(
                "Ignoring invalid keypoint message "
                f"(expected size {NUM_S_VALUES}, got {values.size})",
                throttle_duration_sec=2.0,
            )
            return
        now = time.monotonic()
        self.latest_s = values
        self.last_keypoint_time = now
        self.keypoint_history.append((now, values.copy()))

    # ==================================================================
    # Input validation
    # ==================================================================

    def _missing_inputs(self):
        now = time.monotonic()
        missing = []

        for robot in self.robots.values():
            if robot.actual_xyz is None or robot.actual_q is None:
                missing.append(f"pose on {robot.current_pose_topic}")
            elif robot.last_pose_time is None:
                missing.append(f"{robot.name} pose timestamp")
            elif now - robot.last_pose_time >= 1.0:
                missing.append(
                    f"{robot.name} fresh pose; age="
                    f"{now - robot.last_pose_time:.2f} s"
                )

            subscription_count = robot.pose_pub.get_subscription_count()
            if subscription_count < 1:
                missing.append(
                    f"controller subscriber on {robot.target_pose_topic}"
                )

        if self.latest_s is None:
            missing.append(f"keypoints on {self.KEYPOINT_TOPIC}")
        elif self.last_keypoint_time is None:
            missing.append("keypoint timestamp")
        elif now - self.last_keypoint_time >= 1.0:
            missing.append(
                "fresh keypoints; age="
                f"{now - self.last_keypoint_time:.2f} s"
            )

        return missing

    def _mean_recent_keypoints(self) -> Optional[np.ndarray]:
        if len(self.keypoint_history) < self.average_samples:
            return None
        samples = [
            sample
            for _, sample in list(self.keypoint_history)[
                -self.average_samples:
            ]
        ]
        return np.mean(np.asarray(samples, dtype=float), axis=0)

    # ==================================================================
    # Pose helpers (per robot)
    # ==================================================================

    def _publish_pose(self, robot: RobotState, xyz: np.ndarray, q: np.ndarray):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = robot.base_frame
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        robot.pose_pub.publish(msg)

    def _publish_home(self, robot: RobotState):
        if robot.home_xyz is not None and robot.home_q is not None:
            self._publish_pose(robot, robot.home_xyz, robot.home_q)

    def _publish_all_homes(self):
        """
        Republish BOTH robots' home poses. Called every tick regardless
        of which robot is actively perturbing, so the idle robot stays
        actively commanded (visibly "live") rather than going silent.
        """
        for robot in self.robots.values():
            self._publish_home(robot)

    @staticmethod
    def _relative_rotvec(
        current_q: np.ndarray, reference_q: np.ndarray
    ) -> np.ndarray:
        return (
            R.from_quat(current_q) * R.from_quat(reference_q).inv()
        ).as_rotvec()

    def _actual_control_delta(self, robot: RobotState) -> np.ndarray:
        """Return [dx, dy, dRz] for one robot relative to its home."""
        dxyz = robot.actual_xyz - robot.home_xyz
        drot = self._relative_rotvec(robot.actual_q, robot.home_q)
        return np.array([dxyz[0], dxyz[1], drot[2]], dtype=float)

    def _ignored_z_delta(self, robot: RobotState) -> float:
        return float(robot.actual_xyz[2] - robot.home_xyz[2])

    def _controlled_home_error(
        self, robot: RobotState
    ) -> Tuple[float, float, float]:
        dxyz = robot.actual_xyz - robot.home_xyz
        drot = self._relative_rotvec(robot.actual_q, robot.home_q)
        xy_error = float(np.linalg.norm([dxyz[0], dxyz[1]]))
        yaw_error = float(abs(drot[2]))
        z_drift = float(dxyz[2])
        return xy_error, yaw_error, z_drift

    def _requested_target(
        self, robot: RobotState, dof_index: int, sign: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        xyz = robot.home_xyz.copy()
        q = robot.home_q.copy()

        if dof_index == 0:
            xyz[0] += sign * self.delta_t
        elif dof_index == 1:
            xyz[1] += sign * self.delta_t
        elif dof_index == 2:
            base_rz = R.from_rotvec([0.0, 0.0, sign * self.delta_r])
            q = (base_rz * R.from_quat(robot.home_q)).as_quat()
        else:
            raise ValueError(f"Invalid DoF index: {dof_index}")

        xyz[2] = robot.home_xyz[2]
        return xyz, q

    # ==================================================================
    # Smooth trajectory (per robot)
    # ==================================================================

    def _begin_segment(
        self,
        robot: RobotState,
        start_xyz: np.ndarray,
        start_q: np.ndarray,
        target_xyz: np.ndarray,
        target_q: np.ndarray,
    ):
        robot.segment_start_xyz = start_xyz.copy()
        robot.segment_start_q = start_q.copy()
        robot.segment_target_xyz = target_xyz.copy()
        robot.segment_target_q = target_q.copy()
        robot.segment_start_time = time.monotonic()

    def _execute_segment(self, robot: RobotState) -> bool:
        elapsed = time.monotonic() - robot.segment_start_time
        alpha = smoothstep(elapsed / self.move_duration)

        xyz = robot.segment_start_xyz + alpha * (
            robot.segment_target_xyz - robot.segment_start_xyz
        )
        xyz[2] = robot.home_xyz[2]

        start_rotation = R.from_quat(robot.segment_start_q)
        target_rotation = R.from_quat(robot.segment_target_q)
        relative_rotation = target_rotation * start_rotation.inv()

        q = (
            R.from_rotvec(alpha * relative_rotation.as_rotvec())
            * start_rotation
        ).as_quat()

        self._publish_pose(robot, xyz, q)
        return elapsed >= self.move_duration

    # ==================================================================
    # Sample validation
    # ==================================================================

    def _validate_sample(
        self,
        actual_du_robot: np.ndarray,
        dof_index: int,
        sign: float,
        label: str,
    ):
        value = float(actual_du_robot[dof_index])

        if dof_index < 2:
            minimum = self.minimum_translation
            scale = 1000.0
            unit = "mm"
        else:
            minimum = self.minimum_rotation
            scale = 180.0 / np.pi
            unit = "deg"

        if sign * value <= 0.0:
            self._abort(
                f"{label} actual motion has the wrong sign: "
                f"{value * scale:.3f} {unit}"
            )
            return False

        if abs(value) < minimum:
            self._abort(
                f"{label} actual motion is too small: "
                f"{abs(value) * scale:.3f} {unit}; "
                f"minimum is {minimum * scale:.3f} {unit}"
            )
            return False

        return True

    # ==================================================================
    # State machine
    # ==================================================================

    def _tick(self):
        if self.finished:
            return

        missing_inputs = self._missing_inputs()
        if missing_inputs:
            self.get_logger().warn(
                "Waiting for: " + "; ".join(missing_inputs),
                throttle_duration_sec=2.0,
            )
            return

        for robot in self.robots.values():
            if self.count_publishers(robot.target_pose_topic) > 1:
                self._abort(
                    f"Another target_pose publisher is active on "
                    f"{robot.target_pose_topic}"
                )
                return

        # --------------------------------------------------------------
        # Capture both robots' initial home poses
        # --------------------------------------------------------------
        if self.state == "WAIT_INPUTS":
            for robot in self.robots.values():
                robot.home_xyz = robot.actual_xyz.copy()
                robot.home_q = robot.actual_q.copy()
                self.get_logger().info(
                    f"{robot.name} home pose captured:\n"
                    f"  xyz={np.round(robot.home_xyz, 5)}\n"
                    f"  q={np.round(robot.home_q, 6)}"
                )

            self._publish_all_homes()
            self.settle_deadline = time.monotonic() + self.home_hold
            self.state = "CAPTURE_HOME"
            return

        # --------------------------------------------------------------
        # Capture initial shared rod features
        # --------------------------------------------------------------
        if self.state == "CAPTURE_HOME":
            self._publish_all_homes()

            if time.monotonic() < self.settle_deadline:
                return

            s_home = self._mean_recent_keypoints()
            if s_home is None:
                return

            self.s0 = s_home.copy()
            self.get_logger().info(
                f"Calibration state captured: s0={np.round(self.s0, 5)} (m)"
            )

            self.state = "ESTIMATING"
            self.phase = "START_SAMPLE"
            self.plan_index = 0
            return

        # --------------------------------------------------------------
        # 12 calibration movements (one robot moves per sample; the
        # other robot's home pose is republished every tick throughout)
        # --------------------------------------------------------------
        if self.state == "ESTIMATING":

            if self.plan_index >= len(self.plan):
                self.state = "FINAL_HOME"
                self.return_started = time.monotonic()
                self.return_deadline = self.return_started + self.home_hold
                self._publish_all_homes()
                return

            robot_name, dof_index, sign = self.plan[self.plan_index]
            active_robot = self.robots[robot_name]
            idle_robot = self.robots[
                "NS2" if robot_name == "NS1" else "NS1"
            ]
            dof_name = DOF_NAMES_PER_ROBOT[dof_index]
            sign_text = "+" if sign > 0.0 else "-"
            label = f"{sign_text}{robot_name}.{dof_name}"

            # Keep the idle robot actively holding home every tick,
            # regardless of phase, so both arms stay live on video.
            self._publish_home(idle_robot)

            # ------------------------------------------------------
            # Start perturbation
            # ------------------------------------------------------
            if self.phase == "START_SAMPLE":
                target_xyz, target_q = self._requested_target(
                    active_robot, dof_index, sign
                )
                self.get_logger().info(
                    f"Move {self.plan_index + 1}/{len(self.plan)} "
                    f"({label}): commanding perturbation"
                )
                self._begin_segment(
                    active_robot,
                    active_robot.actual_xyz,
                    active_robot.actual_q,
                    target_xyz,
                    target_q,
                )
                self.phase = "EXEC_SAMPLE"
                return

            # ------------------------------------------------------
            # Execute perturbation
            # ------------------------------------------------------
            if self.phase == "EXEC_SAMPLE":
                if self._execute_segment(active_robot):
                    self.settle_deadline = time.monotonic() + self.wait
                    self.phase = "SETTLE_SAMPLE"
                return

            # ------------------------------------------------------
            # Capture perturbation
            # ------------------------------------------------------
            if self.phase == "SETTLE_SAMPLE":
                self._publish_pose(
                    active_robot,
                    active_robot.segment_target_xyz,
                    active_robot.segment_target_q,
                )

                if time.monotonic() < self.settle_deadline:
                    return

                s_sample = self._mean_recent_keypoints()
                if s_sample is None:
                    return

                actual_du_active = self._actual_control_delta(active_robot)
                ignored_dz_active = self._ignored_z_delta(active_robot)

                ok = self._validate_sample(
                    actual_du_active, dof_index, sign, label
                )
                if not ok or self.finished:
                    return

                if abs(ignored_dz_active) > self.z_drift_warning:
                    self.get_logger().warn(
                        f"Ignored Z drift during {label}: "
                        f"{ignored_dz_active * 1000:.2f} mm"
                    )

                if abs(ignored_dz_active) > self.z_drift_abort:
                    self._abort(
                        f"Z drift during {label} is too large: "
                        f"{ignored_dz_active * 1000:.2f} mm"
                    )
                    return

                # Also record idle robot's (should be ~0) drift purely
                # for logging/inspection -- not used for validation.
                idle_dz = self._ignored_z_delta(idle_robot)

                delta_s = s_sample - self.s0

                # Build the full 6-vector combined delta_u: zeros for
                # the robot that did not move, actual_du for the one
                # that did.
                combined_du = np.zeros(6, dtype=float)
                offset = 0 if robot_name == "NS1" else 3
                combined_du[offset:offset + 3] = actual_du_active

                self.delta_u_samples.append(combined_du.copy())
                self.delta_z_samples.append(
                    {robot_name: ignored_dz_active, idle_robot.name: idle_dz}
                )
                self.delta_s_samples.append(delta_s.copy())
                self.sample_labels.append(label)

                self.get_logger().info(
                    f"Captured {label}:\n"
                    f"  actual controlled du=["
                    f"{actual_du_active[0] * 1000:.3f} mm (x), "
                    f"{actual_du_active[1] * 1000:.3f} mm (y), "
                    f"{np.rad2deg(actual_du_active[2]):.4f} deg (Rz)]\n"
                    f"  ignored dz ({robot_name})="
                    f"{ignored_dz_active * 1000:.3f} mm\n"
                    f"  ||delta_s||={np.linalg.norm(delta_s) * 1000:.3f} mm"
                )

                self._begin_segment(
                    active_robot,
                    active_robot.actual_xyz,
                    active_robot.actual_q,
                    active_robot.home_xyz,
                    active_robot.home_q,
                )
                self.phase = "EXEC_RETURN"
                return

            # ------------------------------------------------------
            # Execute return movement
            # ------------------------------------------------------
            if self.phase == "EXEC_RETURN":
                if self._execute_segment(active_robot):
                    self.return_started = time.monotonic()
                    self.return_deadline = self.return_started + self.wait
                    self.phase = "VERIFY_RETURN"
                return

            # ------------------------------------------------------
            # Verify return to home
            # ------------------------------------------------------
            if self.phase == "VERIFY_RETURN":
                self._publish_home(active_robot)

                if time.monotonic() < self.return_deadline:
                    return

                xy_error, yaw_error, z_drift = self._controlled_home_error(
                    active_robot
                )

                s_home = self._mean_recent_keypoints()
                if s_home is None:
                    return

                image_error = float(np.linalg.norm(s_home - self.s0))

                pose_ok = (
                    xy_error <= self.home_xy_tolerance
                    and yaw_error <= self.home_yaw_tolerance
                )
                image_ok = image_error <= self.feature_return_tolerance
                z_ok = abs(z_drift) <= self.z_drift_abort

                if pose_ok and image_ok and z_ok:
                    self.get_logger().info(
                        f"Returned home after {label}: "
                        f"xy_error={xy_error * 1000:.2f} mm, "
                        f"yaw_error={np.rad2deg(yaw_error):.3f} deg, "
                        f"ignored z_drift={z_drift * 1000:.2f} mm, "
                        f"feature_error={image_error * 1000:.2f} mm"
                    )

                    self.plan_index += 1
                    self.phase = "START_SAMPLE"
                    return

                elapsed = time.monotonic() - self.return_started

                if elapsed >= self.wait + self.max_return_extension:
                    self._abort(
                        f"Return failed after {label}: "
                        f"xy_error={xy_error * 1000:.2f} mm, "
                        f"yaw_error={np.rad2deg(yaw_error):.3f} deg, "
                        f"ignored z_drift={z_drift * 1000:.2f} mm, "
                        f"feature_error={image_error * 1000:.2f} mm"
                    )
                    return

                self.get_logger().warn(
                    f"Waiting for return after {label}: "
                    f"xy_error={xy_error * 1000:.2f} mm, "
                    f"yaw_error={np.rad2deg(yaw_error):.3f} deg, "
                    f"ignored z_drift={z_drift * 1000:.2f} mm, "
                    f"feature_error={image_error * 1000:.2f} mm",
                    throttle_duration_sec=1.0,
                )
                return

            self._abort(f"Unknown phase: {self.phase}")
            return

        # --------------------------------------------------------------
        # Final return to home (both robots)
        # --------------------------------------------------------------
        if self.state == "FINAL_HOME":
            self._publish_all_homes()

            if time.monotonic() < self.return_deadline:
                return

            errors = {
                robot.name: self._controlled_home_error(robot)
                for robot in self.robots.values()
            }

            s_final = self._mean_recent_keypoints()
            if s_final is None:
                return

            image_error = float(np.linalg.norm(s_final - self.s0))

            all_pose_ok = all(
                xy <= self.home_xy_tolerance and yaw <= self.home_yaw_tolerance
                for xy, yaw, _ in errors.values()
            )
            all_z_ok = all(
                abs(z) <= self.z_drift_abort for _, _, z in errors.values()
            )
            image_ok = image_error <= self.feature_return_tolerance

            if all_pose_ok and all_z_ok and image_ok:
                self._finish_success()
                return

            elapsed = time.monotonic() - self.return_started

            if elapsed >= self.home_hold + self.max_return_extension:
                detail = "; ".join(
                    f"{name}: xy_error={xy * 1000:.2f} mm, "
                    f"yaw_error={np.rad2deg(yaw):.3f} deg, "
                    f"z_drift={z * 1000:.2f} mm"
                    for name, (xy, yaw, z) in errors.items()
                )
                self._abort(
                    "Final return failed: "
                    f"{detail}; feature_error={image_error * 1000:.2f} mm"
                )
                return

            self.get_logger().warn(
                "Waiting for final return (both robots)",
                throttle_duration_sec=1.0,
            )
            return

        self._abort(f"Unknown state: {self.state}")

    # ==================================================================
    # Save Jacobian
    # ==================================================================

    def _finish_success(self):
        expected_samples = len(self.plan)
        if len(self.delta_u_samples) != expected_samples:
            self._abort(
                f"Expected {expected_samples} samples, got "
                f"{len(self.delta_u_samples)}"
            )
            return

        # U shape: 6 x 12 (rows = combined DoFs, columns = samples)
        U = np.asarray(self.delta_u_samples, dtype=float).T

        # S shape: NUM_S_VALUES x 12
        S = np.asarray(self.delta_s_samples, dtype=float).T

        rank_u = int(np.linalg.matrix_rank(U))
        cond_u = float(np.linalg.cond(U))

        if rank_u < 6:
            self._abort(
                f"Combined controlled-motion matrix rank={rank_u}; "
                "expected 6. The two robots did not jointly produce "
                "six independent movements."
            )
            return

        J = S @ np.linalg.pinv(U, rcond=1e-6)

        rank_j = int(np.linalg.matrix_rank(J))
        cond_j = float(np.linalg.cond(J))

        if rank_j < min(NUM_S_VALUES, 6):
            self._abort(f"Estimated Jacobian rank={rank_j}; expected higher")
            return

        output_dir = os.path.dirname(self.output) or "."
        os.makedirs(output_dir, exist_ok=True)

        np.save(self.output, J)

        root, extension = os.path.splitext(self.output)
        if extension.lower() == ".npy":
            j_hat_path = root + "_hat.npy"
        else:
            j_hat_path = self.output + "_hat.npy"
        np.save(j_hat_path, J)

        s0_path = os.path.join(output_dir, "s0.npy")
        np.save(s0_path, self.s0)

        # u0: concatenated home pose for both robots
        # [NS1.xyz, NS1.q, NS2.xyz, NS2.q]
        u0_path = os.path.join(output_dir, "u0.npy")
        np.save(
            u0_path,
            np.concatenate(
                [
                    self.robots["NS1"].home_xyz,
                    self.robots["NS1"].home_q,
                    self.robots["NS2"].home_xyz,
                    self.robots["NS2"].home_q,
                ]
            ),
        )

        samples_path = os.path.join(
            output_dir, "jacobian_samples_dual_xy_rz.npz"
        )
        # Flatten per-sample z-drift dicts into two arrays for saving.
        dz_ns1 = np.array(
            [d.get("NS1", np.nan) for d in self.delta_z_samples]
        )
        dz_ns2 = np.array(
            [d.get("NS2", np.nan) for d in self.delta_z_samples]
        )
        np.savez(
            samples_path,
            labels=np.asarray(self.sample_labels),
            delta_u_controlled=U,
            delta_z_ignored_ns1=dz_ns1,
            delta_z_ignored_ns2=dz_ns2,
            delta_s=S,
            combined_dof_labels=np.asarray(COMBINED_DOF_LABELS),
        )

        max_dz = float(
            np.nanmax(np.abs(np.concatenate([dz_ns1, dz_ns2])))
        )

        self.get_logger().info(
            "\n"
            + "=" * 68
            + "\nDUAL-ROBOT [x,y,Rz]x2 JACOBIAN ESTIMATION COMPLETE "
            "(QTM features)"
            + "\n"
            + "=" * 68
            + f"\nCombined DoF columns: {COMBINED_DOF_LABELS}"
            + f"\nFeature vector size: {NUM_S_VALUES} "
            "(3 QTM bodies x,y, metres)"
            + f"\nU shape: {U.shape}, rank={rank_u}, condition={cond_u:.2f}"
            + f"\nJ shape: {J.shape}, rank={rank_j}, condition={cond_j:.2f}"
            + f"\nMaximum ignored |dz| (either robot): {max_dz * 1000:.2f} mm"
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
                "Combined controlled-motion matrix is poorly conditioned"
            )
        if cond_j > 100.0:
            self.get_logger().warn("Feature Jacobian is poorly conditioned")
        if max_dz > self.z_drift_warning:
            self.get_logger().warn(
                "Significant Z coupling was ignored on at least one "
                "robot. Use targets recorded at approximately the "
                "same Z position for both arms."
            )

        self.finished = True
        self.timer.cancel()
        self._publish_all_homes()
        raise SystemExit

    # ==================================================================
    # Abort
    # ==================================================================

    def _abort(self, reason: str):
        if self.finished:
            return
        self.finished = True
        self.timer.cancel()

        for robot in self.robots.values():
            if robot.home_xyz is not None and robot.home_q is not None:
                for _ in range(10):
                    self._publish_home(robot)
                    time.sleep(0.02)

        self.get_logger().error(reason + ". No Jacobian was saved.")
        raise SystemExit


def main(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Dual-robot feature-space Jacobian estimator for combined "
            "DoFs [NS1.x,y,Rz, NS2.x,y,Rz] using shared QTM mocap "
            "features"
        )
    )

    parser.add_argument(
        "--delta_t",
        type=float,
        default=0.035,
        help="Requested X/Y translation perturbation in metres (per robot)",
    )
    parser.add_argument(
        "--delta_r",
        type=float,
        default=0.15,
        help=(
            "Requested Rz perturbation around base-frame Z in radians "
            "(per robot)"
        ),
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=5.0,
        help="Settle time after each perturbation in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="J_dual.npy",
        help="Output path for the combined 6x6 J_dual.npy",
    )
    parser.add_argument(
        "--move_duration",
        type=float,
        default=5.0,
        help="Duration of each smooth movement in seconds",
    )
    parser.add_argument(
        "--home_hold",
        type=float,
        default=3.0,
        help="Time to hold home pose at the start and end",
    )
    parser.add_argument(
        "--average_samples",
        type=int,
        default=15,
        help="Number of recent keypoint samples to average",
    )
    parser.add_argument(
        "--minimum_translation",
        type=float,
        default=0.0002,
        help="Minimum accepted actual X or Y movement in metres",
    )
    parser.add_argument(
        "--minimum_rotation_deg",
        type=float,
        default=0.05,
        help="Minimum accepted actual Rz movement in degrees",
    )
    parser.add_argument(
        "--home_xy_tolerance",
        type=float,
        default=0.006,
        help="Maximum accepted return XY position error in metres",
    )
    parser.add_argument(
        "--home_yaw_tolerance_deg",
        type=float,
        default=2.0,
        help="Maximum accepted return Rz error in degrees",
    )
    parser.add_argument(
        "--feature_return_tolerance",
        type=float,
        default=0.004,
        help=(
            "Maximum accepted feature-space return error norm in metres"
        ),
    )
    parser.add_argument(
        "--z_drift_warning",
        type=float,
        default=0.005,
        help="Z drift warning threshold in metres",
    )
    parser.add_argument(
        "--z_drift_abort",
        type=float,
        default=0.020,
        help="Z drift abort threshold in metres",
    )
    parser.add_argument(
        "--max_return_extension",
        type=float,
        default=8.0,
        help="Extra time allowed for return-to-home in seconds",
    )
    parser.add_argument(
        "--keypoint_topic",
        type=str,
        default="/NS2/rod_keypoints",
        help="Topic publishing the shared rod keypoints",
    )

    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)

    node = DualJacobianEstimatorNode(
        delta_t=parsed.delta_t,
        delta_r=parsed.delta_r,
        wait=parsed.wait,
        output=parsed.output,
        move_duration=parsed.move_duration,
        home_hold=parsed.home_hold,
        average_samples=parsed.average_samples,
        minimum_translation=parsed.minimum_translation,
        minimum_rotation_deg=parsed.minimum_rotation_deg,
        home_xy_tolerance=parsed.home_xy_tolerance,
        home_yaw_tolerance_deg=parsed.home_yaw_tolerance_deg,
        feature_return_tolerance=parsed.feature_return_tolerance,
        z_drift_warning=parsed.z_drift_warning,
        z_drift_abort=parsed.z_drift_abort,
        max_return_extension=parsed.max_return_extension,
        keypoint_topic=parsed.keypoint_topic,
    )

    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        node.get_logger().warn("Dual Jacobian estimation cancelled")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
