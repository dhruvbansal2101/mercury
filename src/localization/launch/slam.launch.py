from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'map_frame': 'map',
            'scan_topic': '/scan',
            'mode': 'mapping',
            'resolution': 0.05,
            'max_laser_range': 10.0,
            'minimum_travel_distance': 0.02,
            'minimum_travel_heading': 0.02,
            'scan_queue_size': 20,
            'throttle_scans': 1,
            'map_update_interval': 0.3,
            'transform_timeout': 0.2,
            'tf_buffer_duration': 30.0
        }]
    )

    return LaunchDescription([
        slam_node
    ])