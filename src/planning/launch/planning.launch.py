from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_planning = get_package_share_directory('planning')

    params = os.path.join(pkg_planning, 'config', 'nav2_params.yaml')
    global_costmap = os.path.join(pkg_planning, 'config', 'global_costmap.yaml')
    local_costmap = os.path.join(pkg_planning, 'config', 'local_costmap.yaml')

    planner = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        parameters=[params, global_costmap],
        output='screen'
    )

    controller = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        parameters=[params, local_costmap],
        output='screen'
    )

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        parameters=[params],
        output='screen'
    )

    behavior = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        parameters=[params],
        output='screen'
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': [
                'planner_server',
                'controller_server',
                'bt_navigator',
                'behavior_server'
            ]
        }]
    )

    return LaunchDescription([
        planner,
        controller,
        bt_navigator,
        behavior,
        lifecycle_manager
    ])