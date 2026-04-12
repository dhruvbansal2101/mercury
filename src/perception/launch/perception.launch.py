from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

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

    return LaunchDescription([
        lane_detection_node,
    ])