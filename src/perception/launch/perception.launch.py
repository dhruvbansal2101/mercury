from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    lane_costmap_node = Node(
        package='perception',
        executable='lane_costmap',
        name='lane_costmap',
        output='screen',
        parameters=[
            {
                "use_sim_time": True,
                "image_topic": "/camera/image_raw",
                "camera_info_topic": "/camera/camera_info",
                "costmap_topic": "/perception/road_costmap",
                "costmap_frame": "map",
                "bev_width": 800,
                "bev_height": 600,
                "resolution": 0.007,
                "undistort": True,
                "show_debug": True,
            }
        ]
    )

    return LaunchDescription([
        lane_costmap_node
    ])