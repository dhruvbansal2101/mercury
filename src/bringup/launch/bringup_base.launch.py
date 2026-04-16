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
    lane_costmap = Node(
        package='perception',
        executable='lane_costmap',
        name='lane_costmap',
        output='screen',
        parameters=[{
            'use_sim_time': True,

            # ── Map extent — must match global_costmap.yaml ──────────────
            'map_width_m':    70.0,
            'map_height_m':   70.0,
            'resolution':      0.10,   # 0.10 m/cell → 700×700 grid
            'map_origin_x':  -35.0,
            'map_origin_y':  -35.0,

            # ── Camera sensor (matches URDF robot_sensors.xacro) ─────────
            'camera_hfov':   1.047,    # rad  (from sensor config)
            'image_width':   640,
            'image_height':  480,

            # ── Detection tuning ─────────────────────────────────────────
            'roi_top_frac':  0.35,     # ignore top 35% (sky, far distance)
            'sample_rows':   8,        # rows to project per frame
            'white_v_min':   170,      # HSV value threshold for white lanes
            'white_s_max':   60,       # HSV saturation threshold

            # ── Projection extent ─────────────────────────────────────────
            # Pixels projected outside each boundary → marked lethal (100).
            # 48 px × ~0.005 m/px (close range) ≈ 0.25 m outside the line.
            'obstacle_pixels_outside': 48,
            # Pixels projected inside boundary → confirmed free (0).
            'free_pixels_inside':      32,

            # ── Performance ───────────────────────────────────────────────
            'publish_rate':    5.0,    # Hz  (StaticLayer re-reads each update)
            'process_every_n': 3,      # process 1-in-3 camera frames (~10 Hz)
        }]
    )

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

    # ── Lane assist node ───────────────────────────────────────────────────
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
        lane_costmap,       # ← new: builds lane boundary costmap for Nav2
        goal_decomposer,
        lane_assist,
        rviz_node,
    ])