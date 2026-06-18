"""
recovery_node.py  v2
---------------------
Standalone recovery node, fully decoupled from lane_bev_carrot_node.py.

WHY SEPARATE
~~~~~~~~~~~~
Recovery logic was previously embedded in the lane-following node and kept
fighting it for control of /goal_pose, causing messy reversing, no
orientation settling, and risky backward driving into unseen obstacles.
This node owns recovery start-to-finish and the lane node's logic is
untouched except for one small change: the lane node now subscribes to
/recovery_active and stays silent while recovery is running, so the two
nodes never race on /goal_pose.

HOW THE TWO NODES COEXIST
~~~~~~~~~~~~~~~~~~~~~~~~~~
* lane_bev_carrot_node.py publishes carrots to /goal_pose as before, but
  pauses publishing whenever /recovery_active is True.
* This node independently watches /final_goal and map TF to know the
  robot's distance to the goal over time.
* If the robot hasn't gotten meaningfully closer to the goal in
  STUCK_WINDOW_SEC seconds, this node declares "stuck", asserts ownership
  via /recovery_active=True, runs recovery, then releases ownership with
  /recovery_active=False so the lane node resumes automatically.
* After recovery ends, a cooldown suppresses stuck detection for a few
  seconds so the robot's own settling/turn-out motion can't instantly
  re-trigger another recovery.

RECOVERY STATE MACHINE
~~~~~~~~~~~~~~~~~~~~~~~
  IDLE -> CHECK_REAR -> BACKING_UP -> SPINNING -> EVALUATING -> SETTLING -> IDLE
  (EVALUATING can escalate back to CHECK_REAR up to max_escalations times)
"""

import math
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import rclpy.qos
from rclpy.node import Node
from rclpy.action import ActionClient

import tf2_ros
import tf_transformations

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from nav2_msgs.action import BackUp, Spin


_R_OPT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


def _qrot(q):
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*(qy*qy+qz*qz),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),  1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)


class RS:
    IDLE        = 'IDLE'
    CHECK_REAR  = 'CHECK_REAR'
    BACKING_UP  = 'BACKING_UP'
    SPINNING    = 'SPINNING'
    EVALUATING  = 'EVALUATING'
    SETTLING    = 'SETTLING'


class RecoveryNode(Node):

    def __init__(self):
        super().__init__('recovery_node')
        self.get_logger().info('=' * 60)
        self.get_logger().info('RecoveryNode v2 - initialising')
        self.get_logger().info('=' * 60)

        # -- Action clients --
        self._backup_client = ActionClient(self, BackUp, '/backup')
        self._spin_client   = ActionClient(self, Spin,   '/spin')

        # -- Parameters --
        self.declare_parameter('stuck_window_sec',          10.0)
        self.declare_parameter('stuck_move_m',               0.30)  # physical displacement gate
        self.declare_parameter('stuck_progress_m',           0.25)  # legacy, unused
        self.declare_parameter('goal_tolerance',              0.5)

        self.declare_parameter('safe_cost_max',              50)
        self.declare_parameter('min_clear_m',                 0.9)
        self.declare_parameter('safety_radius',               0.30)
        self.declare_parameter('pothole_radius_m',            0.9)
        self.declare_parameter('pothole_cost_max',           50)

        self.declare_parameter('rear_check_radius_m',         0.6)
        self.declare_parameter('rear_check_min_clear_m',      0.5)

        self.declare_parameter('backup_dist_m',               0.6)
        self.declare_parameter('backup_speed',                0.12)
        self.declare_parameter('backup_escalation_extra_m',   0.3)
        self.declare_parameter('max_escalations',             1)
        self.declare_parameter('min_acceptable_clearance_m',  0.6)
        self.declare_parameter('backup_corridor_step_m',      0.25)
        self.declare_parameter('backup_min_dist_m',           0.2)

        self.declare_parameter('recovery_carrot_dist_m',      1.5)
        self.declare_parameter('publish_rate_hz',             8.0)
        self.declare_parameter('goal_align_weight',           1.0)  # prefer headings toward the goal
        self.declare_parameter('clearance_weight',            0.3)  # secondary: prefer open headings

        self.declare_parameter('settle_min_sec',               1.0)
        self.declare_parameter('settle_timeout_sec',           6.0)
        self.declare_parameter('settle_heading_tol_deg',      20.0)

        # Cooldown after a recovery before stuck detection can fire again.
        self.declare_parameter('recovery_cooldown_sec',        8.0)

        p = lambda n: self.get_parameter(n).value
        self._stuck_window      = float(p('stuck_window_sec'))
        self._stuck_move_m      = float(p('stuck_move_m'))
        self._stuck_progress    = float(p('stuck_progress_m'))
        self._goal_tol          = float(p('goal_tolerance'))

        self._safe_cost_max     = int(p('safe_cost_max'))
        self._min_clear_m       = float(p('min_clear_m'))
        self._safety_r          = float(p('safety_radius'))
        self._pothole_r         = float(p('pothole_radius_m'))
        self._pothole_cost_max  = int(p('pothole_cost_max'))

        self._rear_check_r      = float(p('rear_check_radius_m'))
        self._rear_min_clear    = float(p('rear_check_min_clear_m'))

        self._backup_dist_base  = float(p('backup_dist_m'))
        self._backup_speed      = float(p('backup_speed'))
        self._backup_extra      = float(p('backup_escalation_extra_m'))
        self._max_escalations   = int(p('max_escalations'))
        self._min_accept_clear  = float(p('min_acceptable_clearance_m'))
        self._backup_step       = float(p('backup_corridor_step_m'))
        self._backup_min_dist   = float(p('backup_min_dist_m'))

        self._recovery_carrot   = float(p('recovery_carrot_dist_m'))
        self._pub_rate          = float(p('publish_rate_hz'))
        self._goal_align_w      = float(p('goal_align_weight'))
        self._clear_w           = float(p('clearance_weight'))

        self._settle_min_sec    = float(p('settle_min_sec'))
        self._settle_timeout    = float(p('settle_timeout_sec'))
        self._settle_heading_tol = math.radians(float(p('settle_heading_tol_deg')))

        self._recovery_cooldown = float(p('recovery_cooldown_sec'))

        self.get_logger().info(
            f'Params | stuck_window={self._stuck_window}s  '
            f'stuck_progress={self._stuck_progress}m  '
            f'backup={self._backup_dist_base}m@{self._backup_speed}m/s  '
            f'max_escalations={self._max_escalations}  '
            f'min_accept_clear={self._min_accept_clear}m  '
            f'cooldown={self._recovery_cooldown}s')

        # -- Recovery state --
        self._state               = RS.IDLE
        self._escalation_count    = 0
        self._current_backup_dist = self._backup_dist_base
        self._scan_samples        = []
        self._best_overall        = None
        self._settle_target       = None
        self._settle_target_yaw   = None
        self._settle_start_time   = None
        self._recovery_active_pub_timer = None

        # -- Tracked goal / pose state --
        self._final_goal            = None
        self._pose_history          = []   # [(t_sec, x_map, y_map)]
        self._stuck_reported        = False
        self._goal_active           = False
        self._recovery_end_time     = None  # set on recovery end -> cooldown

        # -- Sensor caches (mirrors lane node's safety checks) --
        self._road_grid    = None
        self._road_info    = None
        self._pothole_grid = None
        self._pothole_info = None
        self._scan_pts_map = None
        self._scan_msg_raw  = None

        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        sq = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        lq = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(PoseStamped,   '/final_goal',
                                 self._goal_cb,    10)
        self.create_subscription(Odometry,      '/diff_drive_controller/odom',
                                 self._odom_cb,    10)
        self.create_subscription(OccupancyGrid, '/perception/road_costmap',
                                 self._road_cb,    lq)
        self.create_subscription(OccupancyGrid, '/perception/pothole_costmap',
                                 self._pothole_cb, lq)
        self.create_subscription(LaserScan,     '/scan',
                                 self._scan_cb,    sq)

        # Same output topic as the lane node. The lane node pauses while
        # /recovery_active is True, so there is no publisher race.
        self._pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self._active_pub = self.create_publisher(Bool, '/recovery_active', 10)

        self._robot_x = self._robot_y = self._robot_yaw = 0.0

        self.create_timer(1.0 / 5.0, self._monitor_tick)
        self.create_timer(1.0 / self._pub_rate, self._recovery_publish_tick)

        self.get_logger().info(
            'Ready. Watching /final_goal + map TF for stuck detection. '
            'Owns /goal_pose only while recovering.')

    # ====================================================================
    # Callbacks
    # ====================================================================

    def _goal_cb(self, msg: PoseStamped):
        gx, gy = msg.pose.position.x, msg.pose.position.y
        self.get_logger().info(f'[GOAL] tracking new goal -> ({gx:.3f},{gy:.3f})')
        self._final_goal = msg
        self._pose_history.clear()
        self._stuck_reported = False
        self._goal_active = False

    def _odom_cb(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self._robot_yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w])

    def _road_cb(self, msg: OccupancyGrid):
        self._road_info = msg.info
        self._road_grid = msg.data

    def _pothole_cb(self, msg: OccupancyGrid):
        self._pothole_info = msg.info
        self._pothole_grid = msg.data

    def _scan_cb(self, msg: LaserScan):
        self._scan_msg_raw = msg
        try:
            tf = self._tf_buf.lookup_transform(
                'map', msg.header.frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except tf2_ros.TransformException:
            return
        R = _qrot(tf.transform.rotation)
        t = tf.transform.translation
        pts = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min <= r <= msg.range_max:
                p = R @ np.array([r * math.cos(angle), r * math.sin(angle), 0.0])
                pts.append((p[0] + t.x, p[1] + t.y))
            angle += msg.angle_increment
        self._scan_pts_map = np.array(pts, dtype=np.float64) if pts else None

    # ====================================================================
    # Safety helpers - mirrors lane node's _is_safe()
    # ====================================================================

    def _road_cost(self, wx, wy) -> int:
        if self._road_grid is None:
            return -1
        info = self._road_info
        col = int((wx - info.origin.position.x) / info.resolution)
        row = int((wy - info.origin.position.y) / info.resolution)
        if not (0 <= col < info.width and 0 <= row < info.height):
            return -1
        return int(self._road_grid[row * info.width + col])

    def _pothole_cost(self, wx, wy) -> int:
        if self._pothole_grid is None:
            return -1
        info = self._pothole_info
        col = int((wx - info.origin.position.x) / info.resolution)
        row = int((wy - info.origin.position.y) / info.resolution)
        if not (0 <= col < info.width and 0 <= row < info.height):
            return -1
        return int(self._pothole_grid[row * info.width + col])

    def _is_safe(self, wx, wy) -> bool:
        for deg in (0, 45, 90, 135, 180, 225, 270, 315):
            a = math.radians(deg)
            c = self._road_cost(wx + self._safety_r * math.cos(a),
                                wy + self._safety_r * math.sin(a))
            if c != -1 and c >= self._safe_cost_max:
                return False
        c = self._road_cost(wx, wy)
        if c != -1 and c >= self._safe_cost_max:
            return False

        for deg in range(0, 360, 30):
            a = math.radians(deg)
            pc = self._pothole_cost(wx + self._pothole_r * math.cos(a),
                                    wy + self._pothole_r * math.sin(a))
            if pc != -1 and pc >= self._pothole_cost_max:
                return False
        if self._pothole_cost(wx, wy) >= self._pothole_cost_max:
            return False

        if self._scan_pts_map is not None and len(self._scan_pts_map) > 0:
            if np.any(np.hypot(self._scan_pts_map[:, 0] - wx,
                               self._scan_pts_map[:, 1] - wy) < self._min_clear_m):
                return False
        return True

    def _lateral_clearance(self, wx, wy) -> float:
        best = 2.0
        for r in np.arange(0.1, best, 0.1):
            for deg in (0, 45, 90, 135, 180, 225, 270, 315):
                a = math.radians(deg)
                c = self._road_cost(wx + r * math.cos(a), wy + r * math.sin(a))
                if c != -1 and c >= self._safe_cost_max:
                    best = min(best, r)
                    break
        return best

    def _is_on_road(self, wx, wy) -> bool:
        """Stricter than _is_safe: the cell must be KNOWN drivable road.
        Unknown / off-map (-1) is rejected, so recovery never steers off the
        mapped lane into open space that merely looks 'clear'."""
        c = self._road_cost(wx, wy)
        if c == -1 or c >= self._safe_cost_max:
            return False
        if self._pothole_cost(wx, wy) >= self._pothole_cost_max:
            return False
        return True

    def _safe_backup_distance(self, rx, ry, map_yaw, requested) -> float:
        """Largest backup distance (<= requested) whose whole rear corridor
        stays on the road and obstacle-free. 0.0 if even the first step is bad."""
        rear_yaw = map_yaw + math.pi
        step = max(0.1, self._backup_step)
        last_ok = 0.0
        d = step
        while d <= requested + 1e-6:
            px = rx + d * math.cos(rear_yaw)
            py = ry + d * math.sin(rear_yaw)
            if self._is_safe(px, py) and self._is_on_road(px, py):
                last_ok = d
                d += step
            else:
                break
        return last_ok

    def _rear_clearance(self, rx: float, ry: float, map_yaw: float) -> float:
        """
        Clearance directly behind the robot (map_yaw + 180 deg), checked
        against BOTH the static road costmap AND the live LaserScan.
        """
        rear_yaw = map_yaw + math.pi
        check_pt_x = rx + self._rear_check_r * math.cos(rear_yaw)
        check_pt_y = ry + self._rear_check_r * math.sin(rear_yaw)

        map_clear = self._is_safe(check_pt_x, check_pt_y)

        live_clear_dist = float('inf')
        if self._scan_pts_map is not None and len(self._scan_pts_map) > 0:
            dx = self._scan_pts_map[:, 0] - rx
            dy = self._scan_pts_map[:, 1] - ry
            dist = np.hypot(dx, dy)
            ang  = np.arctan2(dy, dx)
            ang_diff = np.abs(np.arctan2(np.sin(ang - rear_yaw), np.cos(ang - rear_yaw)))
            rear_mask = ang_diff < math.radians(45)
            if np.any(rear_mask):
                live_clear_dist = float(np.min(dist[rear_mask]))

        self.get_logger().debug(
            f'[REAR] map_clear={map_clear}  live_clear_dist={live_clear_dist:.2f}m  '
            f'required={self._rear_min_clear}m',
            throttle_duration_sec=1.0)

        if not map_clear:
            return 0.0
        return live_clear_dist

    # ====================================================================
    # Stuck detection
    # ====================================================================

    def _map_pose(self):
        """Return (x, y, yaw) of base_link in map frame, or None on TF failure."""
        try:
            tf = self._tf_buf.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except tf2_ros.TransformException:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return (t.x, t.y, yaw)

    def _monitor_tick(self):
        if self._final_goal is None:
            return

        gx = self._final_goal.pose.position.x
        gy = self._final_goal.pose.position.y

        pose = self._map_pose()
        if pose is None:
            return
        rx_map, ry_map, _ = pose

        dist = math.hypot(gx - rx_map, gy - ry_map)

        if dist <= self._goal_tol:
            if self._goal_active:
                self.get_logger().info(f'[MONITOR] goal reached (dist={dist:.2f}m)')
            self._goal_active = False
            self._pose_history.clear()
            self._stuck_reported = False
            return

        self._goal_active = True

        now = self.get_clock().now().nanoseconds / 1e9

        # Cooldown after a recovery.
        if self._recovery_end_time is not None and \
                (now - self._recovery_end_time) < self._recovery_cooldown:
            self._pose_history.clear()
            return

        # STUCK = the robot has physically barely moved over the window.
        # We do NOT use "distance to goal stopped dropping": on a curved lane
        # that distance naturally stays flat or rises while the robot drives
        # fine, which is what made recovery fire when it shouldn't.
        self._pose_history.append((now, rx_map, ry_map))
        cutoff = now - self._stuck_window
        self._pose_history = [e for e in self._pose_history if e[0] >= cutoff]

        if len(self._pose_history) < 5:
            return
        age = self._pose_history[-1][0] - self._pose_history[0][0]
        if age < self._stuck_window * 0.9:
            return

        x0, y0 = self._pose_history[0][1], self._pose_history[0][2]
        spread = max(math.hypot(x - x0, y - y0) for _, x, y in self._pose_history)

        self.get_logger().debug(
            f'[MONITOR] window={age:.0f}s spread={spread:.3f}m '
            f'(move_thresh={self._stuck_move_m}m) dist_to_goal={dist:.2f}m',
            throttle_duration_sec=5.0)

        if spread < self._stuck_move_m:
            if not self._stuck_reported and self._state == RS.IDLE:
                self.get_logger().warn(
                    f'[MONITOR] STUCK - moved only {spread:.3f}m in {age:.0f}s '
                    f'- starting recovery')
                self._stuck_reported = True
                self._pose_history.clear()
                self._start_recovery()
        else:
            if self._stuck_reported:
                self.get_logger().info('[MONITOR] movement resumed - clearing stuck flag')
            self._stuck_reported = False

    # ====================================================================
    # Recovery publish tick
    # ====================================================================

    def _recovery_publish_tick(self):
        active = self._state != RS.IDLE
        self._active_pub.publish(Bool(data=active))

        if self._state == RS.SETTLING:
            self._tick_settling()

    # ====================================================================
    # Phase: entry point
    # ====================================================================

    def _start_recovery(self):
        if self._state != RS.IDLE:
            self.get_logger().warn(
                f'[RECOVERY] start called but state={self._state} - ignored')
            return
        self._escalation_count    = 0
        self._current_backup_dist = self._backup_dist_base
        self._best_overall        = None
        self._state = RS.CHECK_REAR
        # Claim ownership immediately so the lane node goes silent this instant,
        # rather than waiting up to one publish-tick for the next active=True.
        self._active_pub.publish(Bool(data=True))
        self.get_logger().warn('>>> [RECOVERY] entering CHECK_REAR')
        self._do_check_rear()

    # ====================================================================
    # Phase: CHECK_REAR
    # ====================================================================

    def _do_check_rear(self):
        try:
            base_tf = self._tf_buf.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except tf2_ros.TransformException as e:
            self.get_logger().error(f'[CHECK_REAR] TF failed: {e} - skipping to SPIN')
            self._start_spin_phase()
            return

        bt = base_tf.transform.translation
        bq = base_tf.transform.rotation
        rx, ry = bt.x, bt.y
        _, _, map_yaw = tf_transformations.euler_from_quaternion(
            [bq.x, bq.y, bq.z, bq.w])

        rear_clear = self._rear_clearance(rx, ry, map_yaw)
        self.get_logger().info(
            f'[CHECK_REAR] rear_clearance={rear_clear:.2f}m  '
            f'required={self._rear_min_clear}m')

        if rear_clear >= self._rear_min_clear:
            self.get_logger().info('[CHECK_REAR] rear clear - proceeding to BACKING_UP')
            self._start_backup_phase()
        else:
            self.get_logger().warn(
                '[CHECK_REAR] rear BLOCKED - skipping backup, spinning in place instead')
            self._start_spin_phase()

    # ====================================================================
    # Phase: BACKING_UP
    # ====================================================================

    def _start_backup_phase(self):
        self._state = RS.BACKING_UP
        requested = self._current_backup_dist

        # Clamp the backup so we never reverse off the drivable road.
        pose = self._map_pose()
        if pose is not None:
            rx, ry, yaw = pose
            dist = self._safe_backup_distance(rx, ry, yaw, requested)
        else:
            dist = requested  # no TF: trust Nav2's local costmap to stop us

        if dist < self._backup_min_dist:
            self.get_logger().warn(
                f'[BACKING_UP] rear corridor leaves the road within '
                f'{self._backup_min_dist:.2f}m - skipping backup, spinning instead')
            self._start_spin_phase()
            return

        self.get_logger().warn(
            f'[BACKING_UP] sending BackUp goal dist={dist:.2f}m '
            f'(requested={requested:.2f}m) speed={self._backup_speed}m/s '
            f'(escalation={self._escalation_count})')

        if not self._backup_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('[BACKING_UP] /backup server unavailable - skip to SPIN')
            self._start_spin_phase()
            return

        goal          = BackUp.Goal()
        goal.target.x = dist
        goal.speed    = self._backup_speed

        fut = self._backup_client.send_goal_async(goal)
        fut.add_done_callback(self._backup_response_cb)

    def _backup_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('[BACKING_UP] goal REJECTED - skip to SPIN')
            self._start_spin_phase()
            return
        self.get_logger().info('[BACKING_UP] goal accepted')
        handle.get_result_async().add_done_callback(self._backup_done_cb)

    def _backup_done_cb(self, future):
        result = future.result()
        self.get_logger().info(f'[BACKING_UP] complete (status={result.status})')
        self._start_spin_phase()

    # ====================================================================
    # Phase: SPINNING
    # ====================================================================

    def _start_spin_phase(self):
        self._state        = RS.SPINNING
        self._scan_samples = []

        if not self._spin_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('[SPINNING] /spin server unavailable - using snapshot')
            self._do_evaluate()
            return

        goal            = Spin.Goal()
        goal.target_yaw = math.pi * 2.0

        self.get_logger().info('[SPINNING] sending 360 deg Spin goal')
        fut = self._spin_client.send_goal_async(
            goal, feedback_callback=self._spin_feedback_cb)
        fut.add_done_callback(self._spin_response_cb)

    def _spin_feedback_cb(self, feedback_msg):
        try:
            base_tf = self._tf_buf.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02))
        except tf2_ros.TransformException:
            return
        bq            = base_tf.transform.rotation
        _, _, map_yaw = tf_transformations.euler_from_quaternion(
            [bq.x, bq.y, bq.z, bq.w])
        bt     = base_tf.transform.translation
        rx, ry = bt.x, bt.y

        cx = rx + self._recovery_carrot * math.cos(map_yaw)
        cy = ry + self._recovery_carrot * math.sin(map_yaw)
        safe      = self._is_safe(cx, cy)
        on_road   = self._is_on_road(cx, cy)
        clearance = self._lateral_clearance(cx, cy) if safe else 0.0

        self._scan_samples.append((map_yaw, clearance, rx, ry, on_road, safe))
        self.get_logger().debug(
            f'[SPINNING] yaw={math.degrees(map_yaw):.1f} deg safe={safe} '
            f'on_road={on_road} clearance={clearance:.2f}m '
            f'samples={len(self._scan_samples)}')

    def _spin_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('[SPINNING] goal REJECTED')
            self._do_evaluate()
            return
        self.get_logger().info('[SPINNING] goal accepted - spinning 360 deg')
        handle.get_result_async().add_done_callback(self._spin_done_cb)

    def _spin_done_cb(self, future):
        result = future.result()
        self.get_logger().info(
            f'[SPINNING] complete (status={result.status}) '
            f'samples={len(self._scan_samples)}')
        self._do_evaluate()

    # ====================================================================
    # Phase: EVALUATING
    # ====================================================================

    def _do_evaluate(self):
        self._state = RS.EVALUATING

        gx = gy = None
        if self._final_goal is not None:
            gx = self._final_goal.pose.position.x
            gy = self._final_goal.pose.position.y

        # Score every ON-ROAD, obstacle-free heading from the spin.
        # Primary objective: point toward the goal. Secondary: pick an open lane.
        best = None  # (score, yaw, rx, ry, clearance, align)
        for (yaw, clearance, rx, ry, on_road, safe) in self._scan_samples:
            if not (safe and on_road):
                continue
            if gx is not None:
                bearing = math.atan2(gy - ry, gx - rx)
                align = math.cos(yaw - bearing)          # [-1, 1]
            else:
                align = 0.0
            score = self._goal_align_w * align + self._clear_w * (clearance / 2.0)
            if best is None or score > best[0]:
                best = (score, yaw, rx, ry, clearance, align)

        if best is not None:
            _, byaw, brx, bry, bcl, balign = best
            self.get_logger().info(
                f'[EVALUATING] round {self._escalation_count}: best on-road heading '
                f'yaw={math.degrees(byaw):.1f} deg align={balign:+.2f} '
                f'clearance={bcl:.2f}m (min_accept={self._min_accept_clear}m)')
            if self._best_overall is None or best[0] > self._best_overall[0]:
                self._best_overall = best
        else:
            self.get_logger().warn(
                f'[EVALUATING] round {self._escalation_count}: no on-road heading '
                f'found among {len(self._scan_samples)} samples')

        out_of_attempts = self._escalation_count >= self._max_escalations
        good_enough     = best is not None and best[4] >= self._min_accept_clear

        if good_enough:
            self.get_logger().info('[EVALUATING] acceptable on-road heading - committing')
            self._commit_escape(self._best_overall)
        elif out_of_attempts:
            if self._best_overall is not None:
                self.get_logger().warn(
                    '[EVALUATING] out of attempts - committing best on-road heading found')
                self._commit_escape(self._best_overall)
            else:
                # Nothing on-road anywhere: face the goal directly as last resort
                # so we still hand back to the lane node pointed at the goal.
                self.get_logger().warn(
                    '[EVALUATING] out of attempts and nothing on-road - facing goal')
                self._face_goal_fallback(gx, gy)
        else:
            self._escalation_count    += 1
            self._current_backup_dist += self._backup_extra
            self.get_logger().warn(
                f'[EVALUATING] no good on-road heading - escalating '
                f'(attempt {self._escalation_count}/{self._max_escalations})')
            self._state = RS.CHECK_REAR
            self._do_check_rear()

    def _commit_escape(self, best):
        # best = (score, yaw, rx, ry, clearance, align)
        _, yaw, rx, ry, clearance, _ = best
        self._enter_settling((clearance, yaw, rx, ry))

    def _face_goal_fallback(self, gx, gy):
        pose = self._map_pose()
        if pose is None:
            self.get_logger().error('[EVALUATING] no TF for face-goal - ending recovery')
            self._end_recovery()
            return
        rx, ry, yaw = pose
        if gx is not None:
            yaw = math.atan2(gy - ry, gx - rx)
        self._enter_settling((0.0, yaw, rx, ry))

    # ====================================================================
    # Phase: SETTLING
    # ====================================================================

    def _enter_settling(self, best):
        best_cl, best_yaw, rx, ry = best
        cx = rx + self._recovery_carrot * math.cos(best_yaw)
        cy = ry + self._recovery_carrot * math.sin(best_yaw)

        self._settle_target     = (cx, cy)
        self._settle_target_yaw = best_yaw
        self._settle_start_time = self.get_clock().now()
        self._state              = RS.SETTLING

        self.get_logger().warn(
            f'[SETTLING] escape carrot ({cx:.2f},{cy:.2f}) '
            f'heading={math.degrees(best_yaw):.1f} deg clearance={best_cl:.2f}m - '
            f'republishing until robot orients')

        self._publish_carrot(cx, cy, rx, ry)

    def _tick_settling(self):
        if self._settle_target is None:
            self.get_logger().warn('[SETTLING] no target - aborting to IDLE')
            self._end_recovery()
            return

        try:
            base_tf = self._tf_buf.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except tf2_ros.TransformException:
            return

        bt = base_tf.transform.translation
        bq = base_tf.transform.rotation
        rx, ry = bt.x, bt.y
        _, _, map_yaw = tf_transformations.euler_from_quaternion(
            [bq.x, bq.y, bq.z, bq.w])

        cx, cy = self._settle_target
        elapsed = (self.get_clock().now() - self._settle_start_time).nanoseconds / 1e9

        heading_err = abs(math.atan2(
            math.sin(self._settle_target_yaw - map_yaw),
            math.cos(self._settle_target_yaw - map_yaw)))
        dist_to_target = math.hypot(cx - rx, cy - ry)

        self.get_logger().debug(
            f'[SETTLING] elapsed={elapsed:.1f}s heading_err='
            f'{math.degrees(heading_err):.1f} deg dist_to_target={dist_to_target:.2f}m',
            throttle_duration_sec=0.5)

        facing  = heading_err <= self._settle_heading_tol
        reached = dist_to_target <= 0.3

        if elapsed >= self._settle_timeout:
            self.get_logger().warn(
                f'[SETTLING] timeout ({self._settle_timeout}s) - ending recovery anyway')
            self._end_recovery()
            return

        if elapsed >= self._settle_min_sec and (facing and reached):
            self.get_logger().info(
                f'[SETTLING] oriented (heading_err={math.degrees(heading_err):.1f} deg) '
                f'- ending recovery, handing back to lane node')
            self._end_recovery()
            return

        self._publish_carrot(cx, cy, rx, ry)

    # ====================================================================
    # End recovery
    # ====================================================================

    def _end_recovery(self):
        self._state             = RS.IDLE
        self._settle_target     = None
        self._settle_target_yaw = None
        self._scan_samples      = []
        self._best_overall      = None
        self._stuck_reported    = False
        self._pose_history.clear()
        # Start the cooldown and release ownership immediately so the lane
        # node resumes this instant rather than on the next publish-tick.
        self._recovery_end_time = self.get_clock().now().nanoseconds / 1e9
        self._active_pub.publish(Bool(data=False))
        self.get_logger().warn('[RECOVERY] complete - back to IDLE, lane node resumes control')

    # ====================================================================
    # Publish helper
    # ====================================================================

    def _publish_carrot(self, cx: float, cy: float, rx: float, ry: float):
        yaw = math.atan2(cy - ry, cx - rx)
        msg = PoseStamped()
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.header.frame_id    = 'map'
        msg.pose.position.x    = cx
        msg.pose.position.y    = cy
        msg.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.orientation.w = math.cos(yaw / 2)
        self._pub.publish(msg)
        self.get_logger().debug(
            f'[PUB] (recovery) -> /goal_pose ({cx:.3f},{cy:.3f}) yaw={math.degrees(yaw):.1f} deg')


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()