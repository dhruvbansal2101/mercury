from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true'
    )
    use_sim = LaunchConfiguration('use_sim_time')

    lane_detection_node = Node(
        package='perception',
        executable='lane_detection',
        name='lane_detection_node',
        output='screen',
        parameters=[{
            'use_sim_time':          True,
            'image_topic':           '/camera/image_raw',
            'show_debug':            True,

            'roi_top_frac':          0.35,
            'white_v_min':           170,
            'white_s_max':           60,
            'close_kw':              5,
            'close_kh':              25,
            'canny_low':             30.0,
            'canny_high':            100.0,
            'hough_threshold':       15,
            'hough_min_len':         20.0,
            'hough_max_gap':         40.0,
            'min_slope_abs':         0.2,
            'max_slope_abs':         4.0,

            # ── KEY TUNING PARAMS ───────────────────────────────────────────
            # Half the real lane width in pixels.
            # HOW TO MEASURE: pause sim, look at a [both] frame where both
            # white lane lines are clearly visible, read L:xxx and R:xxx from
            # the debug overlay. lane_half_width_px = (R - L) / 2.
            # Start with 160 for a 640px image.
            'lane_half_width_px':    160.0,

            # Auto-cal OFF — red obstacles were registering as white and
            # corrupting the calibrated lane width (was showing cal=1077px).
            'use_auto_cal':          False,

            # Reject [both] detections where sep > this as obstacle-polluted.
            # = lane_half_width_px * 2 * 1.4  →  160*2*1.4 = 448 ≈ 450
            'max_valid_sep_px':      450.0,

            # Fragment threshold
            'min_lane_sep_px':       120.0,

            'ema_alpha':             0.30,
            'drift_gain':            0.8,
        }]
    )

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

     # ── Lane assist node ───────────────────────────────────────────────────
    lane_assist = Node(
        package='perception',
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

    return LaunchDescription([
        use_sim_time_arg,
        lane_detection_node,
        lane_costmap,
        lane_assist,
    ])