#ifndef QUALISYS_DRIVER_ALL_BODIES_HPP
#define QUALISYS_DRIVER_ALL_BODIES_HPP

#include <cmath>
#include <memory>
#include <sstream>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

// SDK Qualisys
//#include <qualisys/Subject.h>
#include <qualisys/RTProtocol.h>

namespace qualisys
{

class QualisysDriverAllBodies : public rclcpp::Node
{
public:

    /**
     * @brief Constructor
     */
    explicit QualisysDriverAllBodies(
        const std::string & node_name = "qualisys_driver_all_bodies");

    /**
     * @brief Destructor
     */
    ~QualisysDriverAllBodies()
    {
        disconnect();
    }

    /**
     * @brief Initialize connection and publishers
     * @return true if initialization succeeded
     */
    bool init();

    /**
     * @brief Acquire and publish one frame
     */
    void run();

    /**
     * @brief Disconnect from QTM server
     */
    void disconnect();

private:

    // Disable copy
    QualisysDriverAllBodies(
        const QualisysDriverAllBodies &) = delete;

    QualisysDriverAllBodies &
    operator=(const QualisysDriverAllBodies &) = delete;

    /**
     * @brief Process incoming packet
     */
    void handlePacketData(
        CRTPacket * prt_packet);

    /**
     * @brief Optional timer callback
     */
    void timerCallback();

    // Unit converter
    static double deg2rad;

    // =========================
    // Qualisys configuration
    // =========================

    std::string server_address_;
    int base_port_;

    CRTProtocol port_protocol_;

    // =========================
    // ROS2 parameters
    // =========================

    bool publish_tf_;

    // =========================
    // Publishers
    // =========================

    rclcpp::Publisher<
        geometry_msgs::msg::PoseArray>::SharedPtr
            pose_all_bodies_publisher_;

    rclcpp::Publisher<
        visualization_msgs::msg::MarkerArray>::SharedPtr
            marker_pose_array_publisher_;

    // =========================
    // TF broadcaster
    // =========================

    std::unique_ptr<
        tf2_ros::TransformBroadcaster>
            tf_broadcaster_;

    // =========================
    // Timer (optional)
    // =========================

    rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace qualisys

#endif // QUALISYS_DRIVER_ALL_BODIES_HPP
