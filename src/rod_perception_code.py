#!/usr/bin/env python3
"""
rod_perception.py

Replaces aruco_perception.py for the dual-FR3 IBVS migration.

Subscribes to the qualisys_driver "all bodies" outputs:
  - /PoseAllBodies                  (geometry_msgs/PoseArray)
  - /VisualizationPoseArrayMarkers  (visualization_msgs/MarkerArray)  -> carries body names in marker.text

Builds s = [x1, y1, x2, y2, x3, y3] (world-frame X,Y, meters) for the
3 rigid bodies tracked on the elastic rod, in a FIXED, name-resolved order,
and publishes it on /NS2/rod_keypoints as a Float64MultiArray.

Why name-resolved instead of index-resolved:
QTM streams 6DOF bodies in project-file order via Get6DOFBody(i, ...).
That order is stable as long as nobody edits the QTM project's body list,
but a silent index swap (e.g. someone reorders bodies in QTM, or a body
drops out and QTM re-indexes) would feed the wrong rigid body into the
wrong slot of s with no error -- just a wrong Jacobian column. Resolving
by name costs ~nothing and turns that failure mode into a loud warning
instead of a silent wrong-axis bug.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


# ---------------------------------------------------------------------------
# EDIT THIS: map QTM rigid body names (exactly as defined in your QTM
# project file) to fixed slots 0, 1, 2 in s. Order here = order in s.
# Example slot 0 -> rod end 1, slot 1 -> rod middle, slot 2 -> rod end 2.
# ---------------------------------------------------------------------------
BODY_NAME_TO_SLOT = {
    "marker_left": 0,
    "marker_middle": 1,
    "marker_right": 2,
}
NUM_BODIES = len(BODY_NAME_TO_SLOT)

STALE_WARN_SEC = 0.5  # warn if marker names haven't refreshed in this long


class RodPerception(Node):

    def __init__(self):
        super().__init__("rod_perception")

        self.declare_parameter("output_topic", "/NS2/rod_keypoints")
        self.declare_parameter("pose_topic", "/PoseAllBodies")
        self.declare_parameter("marker_topic", "/VisualizationPoseArrayMarkers")

        output_topic = self.get_parameter("output_topic").value
        pose_topic = self.get_parameter("pose_topic").value
        marker_topic = self.get_parameter("marker_topic").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Names arrive on the marker array (marker.text == QTM body name).
        # We cache the names in array order from the latest marker array
        # message, then zip them positionally against the next pose array
        # (both lists are built in the same loop, same order, on the
        # driver's side -- see _marker_cb / _pose_cb for why we match by
        # position rather than by the numeric marker.id).
        self._latest_names_in_order = []
        self._latest_marker_stamp = None

        self.pose_sub = self.create_subscription(
            PoseArray, pose_topic, self._pose_cb, qos
        )
        self.marker_sub = self.create_subscription(
            MarkerArray, marker_topic, self._marker_cb, qos
        )

        self.s_pub = self.create_publisher(Float64MultiArray, output_topic, qos)

        self._warned_missing = set()

        self.get_logger().info(
            f"rod_perception ready. Expecting bodies: {list(BODY_NAME_TO_SLOT.keys())}"
            f" -> publishing s on {output_topic}"
        )

    def _marker_cb(self, msg: MarkerArray):
        # Names by POSITION in this message's marker list, not by the
        # literal marker.id value. QTM body ids (e.g. 4,5,6 if your
        # project has other bodies defined) don't have to start at 0,
        # but /PoseAllBodies has no id field at all -- it's a plain list
        # built in the same loop, same order, on the same driver tick.
        # So position-in-list is the only thing that reliably lines up
        # between the two topics; the numeric id does not.
        names_in_order = [m.text for m in msg.markers]
        self._latest_names_in_order = names_in_order
        self._latest_marker_stamp = self.get_clock().now()

    def _pose_cb(self, msg: PoseArray):
        if not self._latest_names_in_order:
            self.get_logger().warn(
                "No body names received yet on marker topic; "
                "cannot resolve rigid bodies by name.",
                throttle_duration_sec=2.0,
            )
            return

        if len(self._latest_names_in_order) != len(msg.poses):
            self.get_logger().warn(
                "Marker count mismatch between "
                f"/VisualizationPoseArrayMarkers ({len(self._latest_names_in_order)}) "
                f"and /PoseAllBodies ({len(msg.poses)}); skipping this frame.",
                throttle_duration_sec=2.0,
            )
            return

        # name -> 3D position (meters) for this frame, matched by position
        name_to_xy = {}
        for name, pose in zip(self._latest_names_in_order, msg.poses):
            if not name:
                continue
            name_to_xy[name] = (pose.position.x, pose.position.y)

        s = [0.0] * (2 * NUM_BODIES)
        ok = True

        for name, slot in BODY_NAME_TO_SLOT.items():
            xy = name_to_xy.get(name)
            if xy is None:
                if name not in self._warned_missing:
                    self.get_logger().warn(
                        f"Rigid body '{name}' not found in this frame "
                        f"(occluded / not tracked?). Skipping publish."
                    )
                    self._warned_missing.add(name)
                ok = False
                continue
            else:
                self._warned_missing.discard(name)

            x, y = xy
            if not (math.isfinite(x) and math.isfinite(y)):
                self.get_logger().warn(f"Non-finite pose for '{name}', skipping publish.")
                ok = False
                continue

            s[2 * slot] = x
            s[2 * slot + 1] = y

        if not ok:
            return  # don't publish a partially-filled / stale s vector

        out = Float64MultiArray()
        out.layout.dim.append(
            MultiArrayDimension(label="s", size=len(s), stride=len(s))
        )
        out.data = s
        self.s_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RodPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
