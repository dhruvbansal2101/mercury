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

    goal_decomposer = Node( package='bringup', executable='goal_decomposer', name='goal_decomposer', output='screen', parameters=[{
        'use_sim_time': True,
        # Sample a waypoint every 1.5m along the planned path.
        # Lower = more waypoints = smoother curve following but more stop/start.
        # Higher = fewer waypoints = faster but may cut corners.
        'path_sample_dist': 1.5,
        # Robot "passes" a waypoint when it crosses a plane this far
        # ahead of the waypoint. 0.8m = generous gate for smooth flow.
        # If robot keeps overshooting, reduce to 0.5m.
        'gate_dist': 0.8,
        # Skip a waypoint if Nav2 can't reach it (obstacle blocking exact point).
        'skip_on_failure': True,
        # Max time (seconds) to spend on a single waypoint before skipping.
        'wp_timeout_sec': 20.0,
        # Publish sampled waypoints as a path on /goal_decomposer/debug_path
        # so you can visualise them in RViz.
        'publish_debug_path': True,
    }] )

    lane_assist = Node(
        package='bringup',
        executable='lane_assist_node',
        name='lane_assist_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            # Reduced Kp since the EMA in lane_detection now smooths the
            # error — less proportional gain needed to avoid oscillation.
            'Kp': 0.18,
            # Kd helps damp the derivative spike when switching between
            # 'both lanes' and 'single lane' mode on curves.
            'Kd': 0.08,
            'max_correction': 0.3,
            'image_half_width': 320.0,
            # Slightly larger dead-band so small residual EMA errors on
            # straights don't cause constant micro-corrections.
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
        rviz_node
    ])