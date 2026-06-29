#include "qualisys_mocap_ros2/qualisys_driver_all_bodies.hpp"
#include <algorithm>
#include <cmath>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
namespace qualisys
{
double QualisysDriverAllBodies::deg2rad = M_PI / 180.0;
///////////////////////////////////////////////////////////////////////////////
QualisysDriverAllBodies::
QualisysDriverAllBodies(
    const std::string & node_name)
: Node(node_name),
  base_port_(22222),
  publish_tf_(false)
{
    timer_ = create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(
            &QualisysDriverAllBodies::timerCallback,
            this));
}
///////////////////////////////////////////////////////////////////////////////
bool QualisysDriverAllBodies::init()
{
    declare_parameter(
        "server_address",
        "192.168.254.1");
    declare_parameter(
        "server_base_port",
        22222);
    declare_parameter(
        "publish_tf",
        false);
    get_parameter(
        "server_address",
        server_address_);
    get_parameter(
        "server_base_port",
        base_port_);
    get_parameter(
        "publish_tf",
        publish_tf_);
    RCLCPP_INFO_STREAM(
        get_logger(),
        "Connecting to Qualisys Motion Tracking System at "
        << server_address_
        << ":"
        << base_port_);
    if (!port_protocol_.Connect(
            (char *)server_address_.data(),
            base_port_,
            0,
            1,
            19,
            false))
    {
        RCLCPP_FATAL_STREAM(
            get_logger(),
            "Could not connect to "
            << server_address_
            << ":"
            << base_port_);
        return false;
    }
    RCLCPP_INFO_STREAM(
        get_logger(),
        "Connected to "
        << server_address_
        << ":"
        << base_port_);
    bool data_available;
    port_protocol_.Read6DOFSettings(
        data_available);
    pose_all_bodies_publisher_ =
        create_publisher<
            geometry_msgs::msg::PoseArray>(
                "/PoseAllBodies",
                10);
    marker_pose_array_publisher_ =
        create_publisher<
            visualization_msgs::msg::MarkerArray>(
                "/VisualizationPoseArrayMarkers",
                10);
    if (publish_tf_)
    {
        tf_broadcaster_ =
            std::make_unique<
                tf2_ros::TransformBroadcaster>(*this);
    }
    return true;
}
///////////////////////////////////////////////////////////////////////////////
void QualisysDriverAllBodies::disconnect()
{
    RCLCPP_INFO_STREAM(
        get_logger(),
        "Disconnecting from "
        << server_address_
        << ":"
        << base_port_);
    port_protocol_.StreamFramesStop();
    port_protocol_.Disconnect();
}
///////////////////////////////////////////////////////////////////////////////
void QualisysDriverAllBodies::handlePacketData(
    CRTPacket * prt_packet)
{
    int body_count =
        prt_packet->Get6DOFBodyCount();
    visualization_msgs::msg::MarkerArray
        marker_array_msg;
    geometry_msgs::msg::PoseArray
        pose_array_msg;
    pose_array_msg.header.stamp = now();
    pose_array_msg.header.frame_id = "map";
    float x, y, z;
    float rotationMatrix[9];
    for (int i = 0; i < body_count; ++i)
    {
        prt_packet->Get6DOFBody(
            i,
            x,
            y,
            z,
            rotationMatrix);
        std::string body_name(
            port_protocol_.Get6DOFBodyName(i));
        if (!std::isfinite(x) ||
            !std::isfinite(y) ||
            !std::isfinite(z))
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                3000,
                "Rigid body translation contains invalid values");
            continue;
        }
        bool rotation_valid = true;
        for (int j = 0; j < 9; ++j)
        {
            if (!std::isfinite(rotationMatrix[j]))
            {
                rotation_valid = false;
                break;
            }
        }
        if (!rotation_valid)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                3000,
                "Rigid body rotation matrix contains invalid values");
            continue;
        }
        tf2::Matrix3x3 R;
        R.setValue(
            rotationMatrix[0],
            rotationMatrix[3],
            rotationMatrix[6],
            rotationMatrix[1],
            rotationMatrix[4],
            rotationMatrix[7],
            rotationMatrix[2],
            rotationMatrix[5],
            rotationMatrix[8]);
        tf2::Quaternion q;
        R.getRotation(q);
        geometry_msgs::msg::Pose pose;
        pose.position.x = x / 1000.0;
        pose.position.y = y / 1000.0;
        pose.position.z = z / 1000.0;
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();
        pose_array_msg.poses.push_back(
            pose);
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = now();
        marker.ns = "qualisys_bodies";
        marker.id = i;
        marker.text = body_name;
        marker.type =
            visualization_msgs::msg::Marker::SPHERE;
        marker.action =
            visualization_msgs::msg::Marker::ADD;
        marker.pose = pose;
        marker.scale.x = 0.02;
        marker.scale.y = 0.02;
        marker.scale.z = 0.02;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        marker_array_msg.markers.push_back(
            marker);
        if (publish_tf_ &&
            tf_broadcaster_)
        {
            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header.stamp = now();
            tf_msg.header.frame_id = "map";
            tf_msg.child_frame_id = body_name;
            tf_msg.transform.translation.x =
                pose.position.x;
            tf_msg.transform.translation.y =
                pose.position.y;
            tf_msg.transform.translation.z =
                pose.position.z;
            tf_msg.transform.rotation =
                pose.orientation;
            tf_broadcaster_->sendTransform(
                tf_msg);
        }
    }
    pose_all_bodies_publisher_->publish(
        pose_array_msg);
    marker_pose_array_publisher_->publish(
        marker_array_msg);
}
///////////////////////////////////////////////////////////////////////////////
void QualisysDriverAllBodies::run()
{
    CRTPacket * prt_packet =
        port_protocol_.GetRTPacket();
    CRTPacket::EPacketType e_type;
    port_protocol_.GetCurrentFrame(
        CRTProtocol::cComponent6d);
    if (!port_protocol_.ReceiveRTPacket(
            e_type,
            true))
    {
        return;
    }
    switch (e_type)
    {
        case CRTPacket::PacketError:
            RCLCPP_ERROR_STREAM_THROTTLE(
                get_logger(),
                *get_clock(),
                1000,
                "Error while streaming frames: "
                << port_protocol_
                       .GetRTPacket()
                       ->GetErrorString());
            break;
        case CRTPacket::PacketNoMoreData:
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                1000,
                "No more data");
            break;
        case CRTPacket::PacketData:
            handlePacketData(
                prt_packet);
            break;
        default:
            RCLCPP_ERROR_THROTTLE(
                get_logger(),
                *get_clock(),
                1000,
                "Unknown CRTPacket type");
            break;
    }
}
///////////////////////////////////////////////////////////////////////////////
void QualisysDriverAllBodies::timerCallback()
{
    run();
}
///////////////////////////////////////////////////////////////////////////////
} // namespace qualisys