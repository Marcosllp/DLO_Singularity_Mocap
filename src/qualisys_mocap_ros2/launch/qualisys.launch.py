from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # Path vers le fichier YAML installé (IMPORTANT)
    config_file = os.path.join(
        get_package_share_directory('qualisys_mocap_ros2'),
        'config',
        'qualisys.yaml'
    )

    return LaunchDescription([
        Node(
            package='qualisys_mocap_ros2',
            executable='qualisys_node_all_bodies',
            name='qualisys_node',
            output='screen',

            # paramètres ROS 2 (recommandé: YAML)
            parameters=[config_file],

            # optionnel mais utile pour debug
            arguments=['--ros-args', '--log-level', 'info'],
        )
    ])
