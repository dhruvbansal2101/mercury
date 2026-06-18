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

    lane_bev_carrot_node = Node(
        package='perception',
        executable='lane_bev_carrot',
        name='lane_bev_carrot',
        output='screen',
        parameters=[{
            'use_sim_time':          True,
            'carrot_dist_m':         4.8,
            'goal_tolerance':        0.8,
            'publish_rate':          2.0,
            'camera_hfov':           1.047,
            'image_width':           640,
            'image_height':          480,
            'min_proj_m':            0.3,
            'max_proj_m':            6.0,
            'n_bev_samples':         50,
            'fit_cache_sec':         1.0,
            'no_carrot_stop_streak': 3,
            'safe_cost_max': 50,
            'safety_radius':         0.6,
            'max_carrot_dist_m': 6.0,
        }]
    )

    recovery_node = Node(
        package='recovery_manager',
        executable='recovery_node',
        name='recovery_node',
        output='screen'
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
        lane_bev_carrot_node, 
        recovery_node,  
        rviz_node,
    ])