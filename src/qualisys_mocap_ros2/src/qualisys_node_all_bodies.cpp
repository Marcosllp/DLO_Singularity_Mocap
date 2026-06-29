#include <rclcpp/rclcpp.hpp>
#include "qualisys_mocap_ros2/qualisys_driver_all_bodies.hpp"

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    auto node =
        std::make_shared<qualisys::QualisysDriverAllBodies>(
            "qualisys_driver");

    if (!node->init())
    {
        RCLCPP_ERROR(node->get_logger(), "Init failed");
        return -1;
    }

    RCLCPP_INFO(node->get_logger(), "Qualisys driver running");

    rclcpp::spin(node);

    node->disconnect();

    rclcpp::shutdown();
    return 0;
}