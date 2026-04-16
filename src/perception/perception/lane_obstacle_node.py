"""
lane_obstacle_node.py  (v3 — fixed width + correct projection)
--------------------------------------------------------------
KEY FIXES over v2:
  1. Lane width was wrong — using 0.65m but real track is ~2.5m wide
     → use lane_half_width_m = 1.2 (tune this to your actual track)

  2. Wall points were converging to a V-shape because lateral offset
     was not constant — fixed by using constant lateral, varying forward

  3. Added debug logging so you can see exactly where walls are being placed

  4. Wall extends from wall_start_m to obs_range_m ahead at CONSTANT
     lateral positions — creating two parallel lines, not a V
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool


class LaneObstacleNode(Node):

    def __init__(self):
        super().__init__('lane_obstacle_node')

        # ── Parameters ───────────────────────────────────────────────────────
        # CRITICAL: measure actual lane half-width in Gazebo!
        # Click on one white line, note X. Click other white line, note X.
        # lane_half_width_m = (right_x - left_x) / 2
        # From your Gazebo screenshots: lane looks ~2.5m wide → half = 1.2m
        self.declare_parameter('lane_half_width_m',   1.2)

        # Image half-width in pixels (must match lane_detection param)
        self.declare_parameter('image_half_width_px', 160.0)

        # How far ahead to place wall obstacles (metres)
        self.declare_parameter('obstacle_range_m',    3.5)

        # Where to START placing walls (avoid robot footprint)
        self.declare_parameter('wall_start_m',        0.5)

        # Number of points along each wall
        self.declare_parameter('num_wall_points',     25)

        # How many adjacent rays to fill per wall point (wall thickness)
        self.declare_parameter('rays_per_point',       4)

        self.declare_parameter('publish_rate',        10.0)

        self.lane_half_w   = self.get_parameter('lane_half_width_m').value
        self.img_half_w_px = self.get_parameter('image_half_width_px').value
        self.obs_range     = self.get_parameter('obstacle_range_m').value
        self.wall_start    = self.get_parameter('wall_start_m').value
        self.num_pts       = int(self.get_parameter('num_wall_points').value)
        self.rays_per_pt   = int(self.get_parameter('rays_per_point').value)
        pub_rate           = self.get_parameter('publish_rate').value

        self._lane_error      = 0.0
        self._lane_visible    = False
        self._last_error_time = 0.0
        self._log_counter     = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            Float32, '/lane_center_error', self._error_cb, 10)
        self.create_subscription(
            Bool, '/lane_visible', self._visible_cb, 10)

        self._pub = self.create_publisher(LaserScan, '/lane_obstacles', 10)
        self.create_timer(1.0 / pub_rate, self._publish_cb)

        self.get_logger().info(
            f'LaneObstacleNode v3  '
            f'lane_half={self.lane_half_w}m  '
            f'range={self.wall_start}m → {self.obs_range}m  '
            f'pts={self.num_pts}')

    def _error_cb(self, msg: Float32):
        self._lane_error = msg.data
        self._last_error_time = self.get_clock().now().nanoseconds / 1e9

    def _visible_cb(self, msg: Bool):
        self._lane_visible = msg.data

    def _publish_cb(self):
        now     = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9

        # 720 rays = 0.5 degree resolution
        n_rays    = 720
        angle_min = -math.pi
        angle_max =  math.pi
        angle_inc = (angle_max - angle_min) / n_rays
        max_range = self.obs_range + 2.0

        ranges = [max_range] * n_rays

        stale = (now_sec - self._last_error_time) > 0.5

        if self._lane_visible and not stale:
            # ── Convert pixel error to world lateral offset ───────────────────
            # error = img_cx - lane_cx
            # positive error → lane centre is to the LEFT (+Y in base_link)
            # Normalise: lateral offset of lane centre from robot centre
            #   = (error / img_half_px) * lane_width_m / 2
            # BUT: error is in pixels offset from image centre, and
            # image_half_width represents half the lane width in pixels
            # So: lane_cx_lateral_m = (error/img_half_px) * lane_half_w
            lane_cx_lat_m = (self._lane_error / self.img_half_w_px) * self.lane_half_w

            # ── Wall lateral positions in base_link frame ─────────────────────
            # In base_link: +X = forward, +Y = LEFT, -Y = RIGHT
            # Left wall is at:  lane_cx_lat_m + lane_half_w  (positive Y)
            # Right wall is at: lane_cx_lat_m - lane_half_w  (negative Y)
            left_wall_y  = lane_cx_lat_m + self.lane_half_w
            right_wall_y = lane_cx_lat_m - self.lane_half_w

            # Log every 2 seconds
            self._log_counter += 1
            if self._log_counter >= 20:
                self._log_counter = 0
                self.get_logger().info(
                    f'Lane walls: error={self._lane_error:.1f}px  '
                    f'lane_cx_lat={lane_cx_lat_m:.2f}m  '
                    f'left_wall_Y={left_wall_y:.2f}m  '
                    f'right_wall_Y={right_wall_y:.2f}m')

            # ── Place wall points along forward direction ──────────────────────
            # Each wall point: (fwd_distance, wall_Y)
            # In base_link polar: angle = atan2(Y, X), dist = hypot(X, Y)
            #
            # CRITICAL: Y is CONSTANT per wall (parallel lines)
            #           X varies from wall_start to obs_range
            step = (self.obs_range - self.wall_start) / max(self.num_pts - 1, 1)

            for i in range(self.num_pts):
                fwd_x = self.wall_start + i * step  # constant forward sweep

                for wall_y in [left_wall_y, right_wall_y]:
                    # Polar coords in base_link
                    angle = math.atan2(wall_y, fwd_x)
                    dist  = math.hypot(fwd_x, wall_y)

                    if dist >= max_range or dist < 0.1:
                        continue

                    # Find ray index
                    ray_idx = int(round((angle - angle_min) / angle_inc))
                    ray_idx = max(0, min(n_rays - 1, ray_idx))

                    # Fill this ray and neighbors for wall thickness
                    half = self.rays_per_pt // 2
                    for delta in range(-half, half + 1):
                        ni = ray_idx + delta
                        if 0 <= ni < n_rays:
                            if dist < ranges[ni]:
                                ranges[ni] = dist

        # ── Build and publish scan ────────────────────────────────────────────
        scan                 = LaserScan()
        scan.header.stamp    = now.to_msg()
        scan.header.frame_id = 'base_link'
        scan.angle_min       = angle_min
        scan.angle_max       = angle_max
        scan.angle_increment = angle_inc
        scan.time_increment  = 0.0
        scan.scan_time       = 1.0 / 10.0
        scan.range_min       = 0.1
        scan.range_max       = max_range
        scan.ranges          = ranges
        scan.intensities     = []

        self._pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = LaneObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()