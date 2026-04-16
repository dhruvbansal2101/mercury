from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    declare_xacro_file_arg = DeclareLaunchArgument(
        'xacro_file',
        description='Path to the xacro file'
    )

    description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('description'),
                'launch',
                'description.launch.py'
            ])
        ),
        launch_arguments={
            'xacro_file': LaunchConfiguration('xacro_file')
        }.items()
    )

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('localization'),
                'launch',
                'localization.launch.py'
            ])
        )
    )

    planning = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('planning'),
                'launch',
                'planning.launch.py'
            ])
        )
    )

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('perception'),
                'launch',
                'perception.launch.py'
            ])
        )
    )

    # ── Lane costmap node ──────────────────────────────────────────────────
    # Projects camera lane boundaries into a persistent OccupancyGrid
    # (/perception/road_costmap, map frame) so Nav2's global planner
    # treats outside-lane areas as lethal obstacles.
    # Parameters below must match the camera URDF sensor config and the
    # global_costmap.yaml map extent / origin.
    

    # ── Goal decomposer ────────────────────────────────────────────────────
    goal_decomposer = Node(
        package='bringup',
        executable='goal_decomposer',
        name='goal_decomposer',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'path_sample_dist': 2.0,
            'gate_dist': 1.0,
            'min_goal_dist': 0.5,
            'plan_retry_delay_sec': 4.0,
            'nav2_settle_sec': 1.5,
        }]
    )

   

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        parameters=[{'use_sim_time': True}],
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('bringup'),
            'config',
            'bringup.rviz'
        ])],
        output='screen'
    )

    return LaunchDescription([
        declare_xacro_file_arg,
        description,
        localization,
        planning,
        perception,
        goal_decomposer,
        rviz_node,
    ])