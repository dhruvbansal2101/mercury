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

    goal_decomposer = Node(
        package='bringup',
        executable='goal_decomposer',
        name='goal_decomposer',
        output='screen',
        parameters=[{
            'use_sim_time': True,

            # ── Gate / waypoint spacing ───────────────────────────────
            # Sample a gate plane every 2.0m along the planned path.
            # Gates are placed ON the lane centre (planner follows costmap).
            # Increase for faster but coarser waypoint tracking.
            'path_sample_dist': 2.0,

            # Gate plane is 1.0m before the waypoint centre.
            # Robot "crosses" the gate when it passes this imaginary
            # finish line — it does NOT need to reach the exact point.
            # Increase if gates are not triggering (robot passing too fast).
            # Decrease if gates trigger too early.
            'gate_dist': 1.0,

            # Ignore new goals within 0.5m of current goal (debounce)
            'min_goal_dist': 0.5,

            # Wait this long before retrying the planner
            'plan_retry_delay_sec': 4.0,

            # Wait this long after canceling Nav2 before re-sending
            'nav2_settle_sec': 1.5,
        }]
    )

    lane_assist = Node(
        package='bringup',
        executable='lane_assist_node',
        name='lane_assist_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'Kp': 0.18,
            'Kd': 0.08,
            'max_correction': 0.3,
            'image_half_width': 320.0,
            'dead_band_px': 25.0,
            'timeout_sec': 0.5,
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
        lane_assist,
        rviz_node,
    ])