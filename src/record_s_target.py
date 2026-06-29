#!/usr/bin/env python3
"""
record_s_target.py
------------------
Move the robot (and/or the rod) to the TARGET configuration manually,
then run this script. It listens to /NS2/tip_keypoints, averages N
samples, and saves s_target.npy. No Jacobian required.
QTM MIGRATION NOTE:
  s is now [x1, y1, x2, y2, x3, y3] -- world-frame X,Y (METERS) for the
  3 QTM rigid bodies on the rod (marker_left, marker_middle, marker_right),
  published by rod_perception.py. NUM_S_VALUES = 6 (was 8 for 4 ArUco
  corners in pixels).
"""
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
# Number of scalar values in the feature vector s.
# QTM: 3 rigid bodies x (x, y) = 6.  (Was 8 for 4 ArUco corners x (u, v).)
NUM_S_VALUES = 6
S_TARGET_PATH = os.path.expanduser("~/mocap_ros2_ws/src/s_target.npy")
N_SAMPLES = 30  # number of frames to average (~0.3 s at rod_perception's 100 Hz)

class STargetRecorder(Node):
    def __init__(self):
        super().__init__("s_target_recorder")
        self.samples = []
        # Must match rod_perception.py's publisher QoS (BEST_EFFORT) --
        # a default-QoS (RELIABLE) subscriber cannot receive from a
        # BEST_EFFORT publisher; ROS2 will silently drop every message
        # and only print a one-time "incompatible QoS" warning.
        measurement_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.sub = self.create_subscription(
            Float64MultiArray,
            "/NS2/rod_keypoints",
            self._cb,
            measurement_qos,
        )
        self.get_logger().info(
            f"Waiting for {N_SAMPLES} frames on /NS2/rod_keypoints "
            f"(expecting size {NUM_S_VALUES}: "
            "[x_left, y_left, x_mid, y_mid, x_right, y_right], metres) ..."
        )
    def _cb(self, msg: Float64MultiArray):
        arr = np.array(msg.data, dtype=float)
        if arr.size != NUM_S_VALUES:
            self.get_logger().warn(
                f"Unexpected feature size: {arr.size} "
                f"(expected {NUM_S_VALUES}), skipping."
            )
            return
        if not np.all(np.isfinite(arr)):
            self.get_logger().warn("Non-finite values in s, skipping.")
            return
        self.samples.append(arr)
        self.get_logger().info(
            f"  Sample {len(self.samples)}/{N_SAMPLES}: "
            f"{np.round(arr, 4)} (m)"
        )
        if len(self.samples) >= N_SAMPLES:
            s_target = np.mean(self.samples, axis=0)
            output_dir = os.path.dirname(S_TARGET_PATH) or "."
            os.makedirs(output_dir, exist_ok=True)
            np.save(S_TARGET_PATH, s_target)
            self.get_logger().info(f"s_target saved to {S_TARGET_PATH}")
            self.get_logger().info(f"   s_target = {np.round(s_target, 5)} (m)")
            raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    node = STargetRecorder()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()