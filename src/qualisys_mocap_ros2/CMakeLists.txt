cmake_minimum_required(VERSION 3.8)
project(qualisys_mocap_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

##############################################################################
# ROS2
##############################################################################

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

##############################################################################
# Includes
##############################################################################

include_directories(
  include
)

##############################################################################
# Qualisys SDK library
##############################################################################

add_library(qualisys_sdk
  src/qualisys/Network.cpp
  src/qualisys/RTPacket.cpp
  src/qualisys/RTProtocol.cpp
  src/qualisys/Markup.cpp
)

target_include_directories(qualisys_sdk
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/qualisys
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

##############################################################################
# ROS2 Driver library
##############################################################################

add_library(qualisys_driver
  src/qualisys_driver_all_bodies.cpp
)

target_include_directories(qualisys_driver
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

ament_target_dependencies(qualisys_driver
  rclcpp
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
)

target_link_libraries(qualisys_driver
  qualisys_sdk
)

##############################################################################
# Node executable
##############################################################################

add_executable(qualisys_node_all_bodies
  src/qualisys_node_all_bodies.cpp
)

target_link_libraries(qualisys_node_all_bodies
  qualisys_driver
)

ament_target_dependencies(qualisys_node_all_bodies
  rclcpp
)

##############################################################################
# Install
##############################################################################

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  TARGETS
    qualisys_sdk
    qualisys_driver
    qualisys_node_all_bodies

  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

##############################################################################
# Export
##############################################################################

ament_export_include_directories(include)

ament_export_dependencies(
  rclcpp
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
)

ament_package()
