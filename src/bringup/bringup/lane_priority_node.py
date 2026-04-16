"""
lane_priority_node.py
---------------------
Two-mode velocity arbitrator that decides who drives the robot:
  LANE mode  — lane_detection drives. Constant forward speed, PD steering
               from /lane_center_error. Nav2 cmd_vel is IGNORED.
               The robot naturally follows every track curve because the
               camera keeps it centred in the lane.

  NAV2 mode  — Nav2 drives. Its cmd_vel passes through with a gentle lane
               correction blended in (configurable weight). Activated when
               the local costmap reports obstacles within lookahead distance.

Mode transitions:
  LANE → NAV2 : any cell within the lookahead cone exceeds cost threshold
  NAV2 → LANE : obstacle-free for at least hysteresis_sec seconds
                (prevents rapid toggling)

Topic chain (replaces lane_assist_node in the pipeline):
  Nav2 → /cmd_vel → [lane_priority_node] → /cmd_vel_nav → twist_to_stamped

Additional inputs:
  /lane_center_error          (Float32)       — px offset from lane centre
  /lane_visible               (Bool)          — lane found this frame
  /local_costmap/costmap      (OccupancyGrid) — obstacle map
  /diff_drive_controller/odom (Odometry)      — robot heading for cone check

Diagnostic outputs:
  /lane_priority/mode              (String)  — "lane" or "nav2"
  /lane_priority/obstacle_weight   (Float32) — 0.0 = pure lane, 1.0 = pure Nav2
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Duration

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool, Float32, String


# ── Costmap constants ──────────────────────────────────────────────────────────
LETHAL_COST   = 100   # OccupancyGrid cell value for lethal obstacle
INSCRIBED_COST = 99   # just inside inflation radius


class LanePriorityNode(Node):

    MODE_LANE = 'lane'
    MODE_NAV2 = 'nav2'

    def __init__(self):
        super().__init__('lane_priority_node')

        # ── Parameters ────────────────────────────────────────────────────────
        # Lane-following forward speed in LANE mode (m/s).
        # Set to 0 if you want the robot to stop when lane is lost.
        self.declare_parameter('base_speed',              0.30)

        # How far ahead (m) to scan the costmap for obstacles.
        self.declare_parameter('obstacle_lookahead_m',    2.0)

        # OccupancyGrid cell cost above which we treat a cell as an obstacle.
        # 50 = halfway to lethal; 65–70 is a reasonable conservative threshold.
        self.declare_parameter('obstacle_cost_threshold', 65)

        # Number of cells either side of the heading axis to check
        # (robot half-width in costmap cells ≈ 0.20m / 0.05m = 4 cells).
        self.declare_parameter('lateral_check_cells',      4)

        # Seconds of obstacle-free costmap before reverting to LANE mode.
        self.declare_parameter('hysteresis_sec',           2.0)

        # In NAV2 mode: how much lane correction to blend into Nav2 steering.
        # 0.0 = pure Nav2, 1.0 = pure lane correction.
        self.declare_parameter('lane_weight_in_nav2',      0.25)

        # PD controller for lane correction (in both modes).
        self.declare_parameter('Kp',                       0.18)
        self.declare_parameter('Kd',                       0.08)
        self.declare_parameter('max_correction',           0.50)
        self.declare_parameter('image_half_width',        320.0)
        self.declare_parameter('dead_band_px',            20.0)

        # Seconds before a stale lane message causes fallback to straight.
        self.declare_parameter('lane_timeout_sec',         0.5)

        # ── Read parameters ───────────────────────────────────────────────────
        p = self.get_parameter
        self._base_speed        = p('base_speed').value
        self._lookahead_m       = p('obstacle_lookahead_m').value
        self._cost_thresh       = p('obstacle_cost_threshold').value
        self._lat_cells         = p('lateral_check_cells').value
        self._hysteresis_sec    = p('hysteresis_sec').value
        self._lane_nav2_weight  = p('lane_weight_in_nav2').value
        self._Kp                = p('Kp').value
        self._Kd                = p('Kd').value
        self._max_corr          = p('max_correction').value
        self._half_w            = p('image_half_width').value
        self._dead_band         = p('dead_band_px').value
        self._lane_timeout      = p('lane_timeout_sec').value

        # ── Internal state ────────────────────────────────────────────────────
        self._mode              = self.MODE_LANE
        self._nav2_cmd          = Twist()           # latest Nav2 /cmd_vel
        self._lane_error        = 0.0               # latest px offset
        self._prev_lane_error   = 0.0
        self._lane_visible      = False
        self._last_lane_stamp   = None              # rclpy.time.Time

        self._robot_x           = 0.0
        self._robot_y           = 0.0
        self._robot_yaw         = 0.0

        self._costmap_data      = None              # flat list[int]
        self._costmap_info      = None              # MapMetaData

        self._obstacle_detected = False
        self._last_clear_time   = self.get_clock().now()  # last time we saw clear costmap

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Twist,        '/cmd_vel',
                                 self._nav2_cmd_cb,   10)
        self.create_subscription(Float32,      '/lane_center_error',
                                 self._lane_error_cb, 10)
        self.create_subscription(Bool,         '/lane_visible',
                                 self._lane_vis_cb,   10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap',
                                 self._costmap_cb,     1)   # latched, depth=1
        self.create_subscription(Odometry,     '/diff_drive_controller/odom',
                                 self._odom_cb,        10)

        # ── Publishers ────────────────────────────────────────────────────────
        self._cmd_pub    = self.create_publisher(Twist,   '/cmd_vel_nav',                   10)
        self._mode_pub   = self.create_publisher(String,  '/lane_priority/mode',             5)
        self._weight_pub = self.create_publisher(Float32, '/lane_priority/obstacle_weight',  5)

        # ── Control loop at 20 Hz ─────────────────────────────────────────────
        self.create_timer(0.05,  self._control_loop)
        # Mode diagnostics at 2 Hz
        self.create_timer(0.5,   self._publish_diagnostics)

        self.get_logger().info(
            f'LanePriorityNode started | '
            f'base_speed={self._base_speed}m/s  '
            f'lookahead={self._lookahead_m}m  '
            f'cost_thresh={self._cost_thresh}  '
            f'hysteresis={self._hysteresis_sec}s')

    # ─────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ─────────────────────────────────────────────────────────────────────────
    def _nav2_cmd_cb(self, msg: Twist):
        self._nav2_cmd = msg

    def _lane_error_cb(self, msg: Float32):
        self._lane_error      = msg.data
        self._last_lane_stamp = self.get_clock().now()

    def _lane_vis_cb(self, msg: Bool):
        self._lane_visible = msg.data

    def _odom_cb(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # Yaw from quaternion (Z-up convention)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _costmap_cb(self, msg: OccupancyGrid):
        self._costmap_info = msg.info
        self._costmap_data = msg.data   # flat tuple/list row-major

    # ─────────────────────────────────────────────────────────────────────────
    # Obstacle detection — scan a forward cone in the local costmap
    # ─────────────────────────────────────────────────────────────────────────
    def _obstacles_ahead(self) -> bool:
        """
        Return True if any costmap cell within the forward lookahead cone
        has a cost >= _cost_thresh.

        The local costmap is a rolling window centered on the robot (in odom
        frame). We compute the robot's cell position from the costmap origin
        and scan forward along the robot's heading.
        """
        if self._costmap_data is None or self._costmap_info is None:
            return False

        info   = self._costmap_info
        res    = info.resolution           # metres per cell
        ox     = info.origin.position.x
        oy     = info.origin.position.y
        w      = info.width                # cells
        h      = info.height               # cells
        data   = self._costmap_data

        # Robot cell in costmap grid
        rx_c = int((self._robot_x - ox) / res)
        ry_c = int((self._robot_y - oy) / res)

        cos_y = math.cos(self._robot_yaw)
        sin_y = math.sin(self._robot_yaw)
        steps = max(1, int(self._lookahead_m / res))  # forward steps

        for step in range(2, steps + 1):          # skip 1 (robot footprint)
            # Centre of the scan column at this step
            cx = rx_c + int(round(step * cos_y))
            cy = ry_c + int(round(step * sin_y))

            # Sweep laterally across the robot width
            for lat in range(-self._lat_cells, self._lat_cells + 1):
                # Perpendicular offset: rotate (lat, 0) by heading
                px = cx + int(round(lat * (-sin_y)))
                py = cy + int(round(lat *   cos_y))

                if 0 <= px < w and 0 <= py < h:
                    cost = data[py * w + px]
                    if cost >= self._cost_thresh:
                        return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Lane PD correction (shared between both modes)
    # ─────────────────────────────────────────────────────────────────────────
    def _lane_correction(self) -> float:
        """
        Returns angular velocity correction from the lane PD controller.
        Returns 0.0 when lane is stale or inside the dead-band.
        """
        if self._last_lane_stamp is None:
            return 0.0

        age   = self.get_clock().now() - self._last_lane_stamp
        stale = age > Duration(seconds=self._lane_timeout)

        if not self._lane_visible or stale:
            self._prev_lane_error = 0.0
            return 0.0

        err = self._lane_error
        if abs(err) < self._dead_band:
            self._prev_lane_error = err
            return 0.0

        half_w     = max(self._half_w, 1.0)
        norm       = err / half_w
        deriv      = (err - self._prev_lane_error) / half_w
        correction = self._Kp * norm - self._Kd * deriv
        correction = max(-self._max_corr, min(self._max_corr, correction))

        self._prev_lane_error = err
        return correction

    # ─────────────────────────────────────────────────────────────────────────
    # Mode switching with hysteresis
    # ─────────────────────────────────────────────────────────────────────────
    def _update_mode(self):
        obs = self._obstacles_ahead()
        now = self.get_clock().now()

        if obs:
            self._obstacle_detected = True
            self._last_clear_time   = now          # reset clear timer

        else:
            clear_for = (now - self._last_clear_time).nanoseconds / 1e9
            if clear_for >= self._hysteresis_sec:
                self._obstacle_detected = False

        prev = self._mode
        self._mode = self.MODE_NAV2 if self._obstacle_detected else self.MODE_LANE

        if self._mode != prev:
            self.get_logger().info(
                f'[LanePriority] Mode → {self._mode.upper()}  '
                f'(obstacle={self._obstacle_detected})')

    # ─────────────────────────────────────────────────────────────────────────
    # Control loop — 20 Hz
    # ─────────────────────────────────────────────────────────────────────────
    def _control_loop(self):
        self._update_mode()

        out = Twist()

        if self._mode == self.MODE_LANE:
            # ── LANE MODE ────────────────────────────────────────────────────
            # Constant forward speed; lane detection provides all steering.
            # Nav2 cmd_vel is completely ignored — the robot follows the track
            # naturally because the camera keeps it centred in the lane.
            out.linear.x  = self._base_speed
            out.angular.z = self._lane_correction()

            # If lane is lost (camera sees nothing), go straight at low speed
            # so the robot doesn't sit still indefinitely.
            if not self._lane_visible:
                out.angular.z = 0.0

        else:
            # ── NAV2 MODE ────────────────────────────────────────────────────
            # Nav2 handles linear and angular velocity for obstacle avoidance.
            # Blend in a fraction of lane correction to help keep lane centre.
            lane_corr = self._lane_correction()
            nav2_ang  = self._nav2_cmd.angular.z

            out.linear.x  = self._nav2_cmd.linear.x
            out.angular.z = ((1.0 - self._lane_nav2_weight) * nav2_ang
                             + self._lane_nav2_weight        * lane_corr)

        self._cmd_pub.publish(out)

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics — 2 Hz
    # ─────────────────────────────────────────────────────────────────────────
    def _publish_diagnostics(self):
        mode_msg      = String()
        mode_msg.data = self._mode
        self._mode_pub.publish(mode_msg)

        weight        = Float32()
        weight.data   = 1.0 if self._mode == self.MODE_NAV2 else 0.0
        self._weight_pub.publish(weight)

        self.get_logger().debug(
            f'[LanePriority] mode={self._mode}  '
            f'lane_err={self._lane_error:.1f}px  '
            f'lane_vis={self._lane_visible}  '
            f'obstacle={self._obstacle_detected}')


def main(args=None):
    rclpy.init(args=args)
    node = LanePriorityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()