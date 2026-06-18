"""
recovery_node.py  v7
---------------------
Standalone recovery node, fully decoupled from lane_bev_carrot_node.py.

CHANGELOG vs v6
---------------
* MAKE ROOM TO SPIN with an OPEN-LOOP backup. nav2's BackUp/Spin behaviors both
  abort with "Collision Ahead" when the robot is pressed against the obstacle:
  the robot's footprint already overlaps a lethal cell, so the behavior's own
  collision check fails on the very first pose and it never moves (the log shows
  BackUp moving 0.03m then aborting, then Spin aborting too). nav2 simply will
  not reverse off an obstacle it is touching.
  The fix: when there is not enough clearance to spin, reverse OPEN-LOOP via
  /cmd_vel (Twist or TwistStamped), guarded by the LOCAL COSTMAP behind us
  (unknown = passable; this robot has no rear lidar). We back up one small step
  at a time, re-check spin clearance, and repeat - up to max_total_backup_m -
  until a footprint-sized disk is clear, THEN spin. If a spin still aborts we
  back up a bit more and retry (max_spin_retries). If we cannot make room (rear
  blocked AND can't spin) we face the goal and wait for the obstacle to clear.
  Set open_loop_backup=False to fall back to the nav2 BackUp action.
  IMPORTANT: set cmd_vel_topic / use_stamped_cmd_vel to match your controller,
  and make sure nothing else is driving /cmd_vel during recovery.

CHANGELOG vs v5
---------------
* WAIT-UNTIL-CLEAR handoff. After the recovery maneuver, do NOT hand control
  back to the lane node while the path toward the goal is still blocked (the
  goal can be BEHIND the obstacle, so the lane node would drive straight back
  into it). Instead enter WAIT_CLEAR: park the robot, keep /recovery_active
  True so the lane node stays silent, and wait until the path ahead clears
  (live-scan gated by default; wait_clear_use_costmap adds costmap gating)
  before handing control back. wait_clear_timeout_sec bounds the wait
  (0 = forever). Set wait_for_clear_path=False to restore immediate handback.

CHANGELOG vs v4
---------------
* Make ROOM before spinning (now actually implemented, see v7).
* Evaluation never points at the obstacle: it picks the best ON-ROAD,
  goal-ward heading; if nothing on-road is safe it faces the goal and waits
  rather than ramming the obstacle.

CHANGELOG vs v3
---------------
* Rear check no longer uses the lidar (no rear lidar on this robot). Rear is
  judged from the LOCAL COSTMAP only, UNKNOWN cells treated as passable, so the
  robot can reverse off an obstacle it is touching head-on.

CHANGELOG vs v2
---------------
* New /final_goal CANCELS any in-progress recovery immediately and hands
  control back to the lane node.
* Stuck detection requires BOTH low movement AND a confirmed obstacle ahead.
* Goal cooldown + recovery cooldown to avoid false / repeated triggers.
* Backup/spin effectiveness verification.
* Lane-aware escape headings (follow the road, don't cross it).
* Hard state reset on end / cancel.

The lane node is unchanged (v6): it stays alive and simply pauses publishing
while /recovery_active is True, resuming the instant it goes False.
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

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
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
    WAIT_CLEAR  = 'WAIT_CLEAR'   # hold position until path toward goal clears


class RecoveryNode(Node):

    def __init__(self):
        super().__init__('recovery_node')
        self.get_logger().info('=' * 60)
        self.get_logger().info('RecoveryNode v7 - initialising')
        self.get_logger().info('=' * 60)

        # -- Action clients --
        self._backup_client = ActionClient(self, BackUp, '/backup')
        self._spin_client   = ActionClient(self, Spin,   '/spin')

        # -- Parameters --
        self.declare_parameter('stuck_window_sec',          6.0)
        self.declare_parameter('stuck_move_m',               0.30)
        self.declare_parameter('stuck_progress_m',           0.25)  # legacy, unused
        self.declare_parameter('goal_tolerance',              0.5)
        self.declare_parameter('goal_settle_sec',             4.0)

        self.declare_parameter('safe_cost_max',              50)
        self.declare_parameter('min_clear_m',                 0.9)
        self.declare_parameter('safety_radius',               0.30)
        self.declare_parameter('pothole_radius_m',            0.9)
        self.declare_parameter('pothole_cost_max',           50)

        self.declare_parameter('rear_check_radius_m',         0.6)   # legacy, unused
        self.declare_parameter('rear_check_min_clear_m',      0.5)   # legacy, unused

        # Rear is judged from the LOCAL COSTMAP (no rear lidar on this robot).
        self.declare_parameter('local_costmap_topic',  '/local_costmap/costmap')
        self.declare_parameter('local_cost_max',        90)

        # Require a real obstacle ahead (not just low progress) before recovery.
        self.declare_parameter('require_obstacle',            True)
        self.declare_parameter('obstacle_check_dist_m',       1.2)
        self.declare_parameter('obstacle_check_cone_deg',    60.0)

        # Effectiveness verification.
        self.declare_parameter('min_effective_move_m',       0.10)
        self.declare_parameter('min_spin_rotation_deg',     180.0)

        self.declare_parameter('backup_dist_m',               0.6)
        self.declare_parameter('backup_speed',                0.12)
        self.declare_parameter('backup_escalation_extra_m',   0.3)
        self.declare_parameter('max_escalations',             1)
        self.declare_parameter('min_acceptable_clearance_m',  0.6)
        self.declare_parameter('backup_corridor_step_m',      0.25)
        self.declare_parameter('backup_min_dist_m',           0.2)

        self.declare_parameter('recovery_carrot_dist_m',      0.8)
        self.declare_parameter('publish_rate_hz',             8.0)
        self.declare_parameter('goal_align_weight',           1.0)
        self.declare_parameter('clearance_weight',            0.3)
        self.declare_parameter('road_run_weight',             0.4)
        self.declare_parameter('min_road_run_m',              0.5)
        self.declare_parameter('road_run_max_m',              1.5)

        self.declare_parameter('settle_min_sec',               0.5)
        self.declare_parameter('settle_timeout_sec',           3.0)
        self.declare_parameter('settle_heading_tol_deg',      20.0)

        self.declare_parameter('recovery_cooldown_sec',        8.0)

        # -- WAIT-UNTIL-CLEAR handoff --
        self.declare_parameter('wait_for_clear_path',        True)
        self.declare_parameter('wait_clear_confirm_count',   5)
        self.declare_parameter('wait_clear_timeout_sec',     0.0)    # 0 = wait indefinitely
        self.declare_parameter('wait_clear_face_goal',       True)
        self.declare_parameter('wait_clear_use_costmap',     False)

        # -- Make-room-to-spin / open-loop backup --
        # nav2 BackUp/Spin abort ("Collision Ahead") when the footprint already
        # overlaps the obstacle, so the robot can never reverse off it. We
        # reverse OPEN-LOOP via /cmd_vel instead, guarded by the local costmap
        # behind (unknown = passable; no rear lidar), backing up in steps until
        # there is enough room to spin, then spinning.
        self.declare_parameter('open_loop_backup',          True)
        self.declare_parameter('cmd_vel_topic',             '/cmd_vel')
        self.declare_parameter('use_stamped_cmd_vel',       False)  # True for Jazzy diff_drive_controller (TwistStamped)
        self.declare_parameter('max_total_backup_m',        1.2)    # total reverse budget while making room
        self.declare_parameter('spin_clearance_radius_m',   1.5)   # footprint-ish disk that must be clear to spin
        self.declare_parameter('max_spin_retries',          3)      # extra backup+spin attempts if spin keeps colliding
        self.declare_parameter('motion_rate_hz',            20.0)   # open-loop cmd_vel publish rate

        p = lambda n: self.get_parameter(n).value
        self._stuck_window      = float(p('stuck_window_sec'))
        self._stuck_move_m      = float(p('stuck_move_m'))
        self._stuck_progress    = float(p('stuck_progress_m'))
        self._goal_tol          = float(p('goal_tolerance'))
        self._goal_settle_sec   = float(p('goal_settle_sec'))

        self._safe_cost_max     = int(p('safe_cost_max'))
        self._min_clear_m       = float(p('min_clear_m'))
        self._safety_r          = float(p('safety_radius'))
        self._pothole_r         = float(p('pothole_radius_m'))
        self._pothole_cost_max  = int(p('pothole_cost_max'))

        self._rear_check_r      = float(p('rear_check_radius_m'))
        self._rear_min_clear    = float(p('rear_check_min_clear_m'))
        self._local_cost_max    = int(p('local_cost_max'))
        local_topic             = str(p('local_costmap_topic'))

        self._require_obstacle  = bool(p('require_obstacle'))
        self._obstacle_dist     = float(p('obstacle_check_dist_m'))
        self._obstacle_cone     = float(p('obstacle_check_cone_deg'))
        self._min_eff_move      = float(p('min_effective_move_m'))
        self._min_spin_rot      = math.radians(float(p('min_spin_rotation_deg')))

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
        self._run_w             = float(p('road_run_weight'))
        self._min_road_run      = float(p('min_road_run_m'))
        self._road_run_max      = float(p('road_run_max_m'))

        self._settle_min_sec    = float(p('settle_min_sec'))
        self._settle_timeout    = float(p('settle_timeout_sec'))
        self._settle_heading_tol = math.radians(float(p('settle_heading_tol_deg')))

        self._recovery_cooldown = float(p('recovery_cooldown_sec'))

        self._wait_for_clear    = bool(p('wait_for_clear_path'))
        self._wait_confirm_n    = int(p('wait_clear_confirm_count'))
        self._wait_timeout      = float(p('wait_clear_timeout_sec'))
        self._wait_face_goal    = bool(p('wait_clear_face_goal'))
        self._wait_use_costmap  = bool(p('wait_clear_use_costmap'))

        self._open_loop_backup  = bool(p('open_loop_backup'))
        cmd_topic               = str(p('cmd_vel_topic'))
        self._use_stamped_cmd   = bool(p('use_stamped_cmd_vel'))
        self._max_total_backup  = float(p('max_total_backup_m'))
        self._spin_clear_r      = float(p('spin_clearance_radius_m'))
        self._max_spin_retries  = int(p('max_spin_retries'))
        motion_rate             = float(p('motion_rate_hz'))

        self.get_logger().info(
            f'Params | stuck_window={self._stuck_window}s  '
            f'backup={self._backup_dist_base}m@{self._backup_speed}m/s  '
            f'min_accept_clear={self._min_accept_clear}m  '
            f'cooldown={self._recovery_cooldown}s')
        self.get_logger().info(
            f'MakeRoom | open_loop_backup={self._open_loop_backup} '
            f'cmd_vel={cmd_topic} stamped={self._use_stamped_cmd} '
            f'max_total_backup={self._max_total_backup}m '
            f'spin_clear_r={self._spin_clear_r}m max_spin_retries={self._max_spin_retries}')
        self.get_logger().info(
            f'WaitClear | enabled={self._wait_for_clear} '
            f'confirm={self._wait_confirm_n} ticks  '
            f'timeout={self._wait_timeout}s (0=forever)  '
            f'use_costmap={self._wait_use_costmap}')

        # -- Recovery state --
        self._state               = RS.IDLE
        self._escalation_count    = 0
        self._current_backup_dist = self._backup_dist_base
        self._scan_samples        = []
        self._best_overall        = None
        self._settle_target       = None
        self._settle_target_yaw   = None
        self._settle_start_time   = None

        # -- Tracked goal / pose state --
        self._final_goal            = None
        self._pose_history          = []
        self._stuck_reported        = False
        self._goal_active           = False
        self._recovery_end_time     = None
        self._goal_set_time         = None
        self._active_goal_handle    = None
        self._backup_ineffective    = False
        self._spin_ineffective      = False
        self._phase_start_pose      = None

        # -- WAIT_CLEAR (hold-until-path-clears) state --
        self._wait_hold             = None
        self._wait_start            = None
        self._wait_streak           = 0

        # -- Make-room / open-loop backup state --
        self._total_backed_up       = 0.0
        self._spin_retry            = 0
        self._ol_active             = False   # open-loop reverse in progress
        self._ol_start              = (0.0, 0.0)
        self._ol_target             = 0.0
        self._ol_start_time         = None
        self._ol_after              = None    # callback when an increment finishes

        # -- Sensor caches --
        self._road_grid    = None
        self._road_info    = None
        self._pothole_grid = None
        self._pothole_info = None
        self._scan_pts_map = None
        self._scan_msg_raw  = None
        self._local_grid   = None
        self._local_info   = None
        self._local_frame  = ''

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
        self.create_subscription(OccupancyGrid, local_topic,
                                 self._local_cb,   lq)
        self.create_subscription(LaserScan,     '/scan',
                                 self._scan_cb,    sq)

        # Same output topic as the lane node. The lane node pauses while
        # /recovery_active is True, so there is no publisher race.
        self._pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self._active_pub = self.create_publisher(Bool, '/recovery_active', 10)

        # Direct velocity command for the OPEN-LOOP backup (bypasses nav2's
        # behavior-server collision gate, which refuses to move off a touching
        # obstacle). Twist (Humble) or TwistStamped (Jazzy diff_drive_controller).
        if self._use_stamped_cmd:
            self._cmd_pub = self.create_publisher(TwistStamped, cmd_topic, 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        self._robot_x = self._robot_y = self._robot_yaw = 0.0

        self.create_timer(1.0 / 5.0, self._monitor_tick)
        self.create_timer(1.0 / self._pub_rate, self._recovery_publish_tick)
        self.create_timer(1.0 / max(1.0, motion_rate), self._motion_tick)

        self.get_logger().info(
            'Ready. Watching /final_goal + map TF for stuck detection. '
            'Owns /goal_pose (and /cmd_vel during open-loop backup) only while recovering.')

    # ====================================================================
    # Callbacks
    # ====================================================================

    def _goal_cb(self, msg: PoseStamped):
        gx, gy = msg.pose.position.x, msg.pose.position.y
        self.get_logger().info(f'[GOAL] tracking new goal -> ({gx:.3f},{gy:.3f})')
        if self._state != RS.IDLE:
            self._cancel_recovery('new goal received')
        self._final_goal = msg
        self._pose_history.clear()
        self._stuck_reported = False
        self._goal_active = False
        self._goal_set_time = self.get_clock().now().nanoseconds / 1e9

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

    def _local_cb(self, msg: OccupancyGrid):
        self._local_info  = msg.info
        self._local_grid  = msg.data
        self._local_frame = msg.header.frame_id

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
    # Safety helpers
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
        c = self._road_cost(wx, wy)
        if c == -1 or c >= self._safe_cost_max:
            return False
        if self._pothole_cost(wx, wy) >= self._pothole_cost_max:
            return False
        return True

    def _lookup_R_t(self, target_frame, source_frame='map'):
        try:
            tf = self._tf_buf.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except tf2_ros.TransformException:
            return None
        return (_qrot(tf.transform.rotation), tf.transform.translation)

    def _local_cost_at(self, wx, wy, Rt) -> int:
        if self._local_grid is None:
            return -1
        info = self._local_info
        if Rt is None:
            px, py = wx, wy
        else:
            R, t = Rt
            pp = R @ np.array([wx, wy, 0.0])
            px, py = pp[0] + t.x, pp[1] + t.y
        col = int((px - info.origin.position.x) / info.resolution)
        row = int((py - info.origin.position.y) / info.resolution)
        if not (0 <= col < info.width and 0 <= row < info.height):
            return -1
        return int(self._local_grid[row * info.width + col])

    def _rear_cell_blocked(self, px, py, Rt) -> bool:
        lc = self._local_cost_at(px, py, Rt)
        if lc != -1 and lc >= self._local_cost_max:
            return True
        rc = self._road_cost(px, py)
        if rc != -1 and rc >= self._safe_cost_max:
            return True
        pc = self._pothole_cost(px, py)
        if pc != -1 and pc >= self._pothole_cost_max:
            return True
        return False

    def _safe_backup_distance(self, rx, ry, map_yaw, requested) -> float:
        rear_yaw = map_yaw + math.pi
        step = max(0.1, self._backup_step)
        Rt = None
        if self._local_grid is not None and self._local_frame not in ('map', ''):
            Rt = self._lookup_R_t(self._local_frame)
        last_ok = 0.0
        d = step
        while d <= requested + 1e-6:
            px = rx + d * math.cos(rear_yaw)
            py = ry + d * math.sin(rear_yaw)
            if self._rear_cell_blocked(px, py, Rt):
                break
            last_ok = d
            d += step
        return last_ok

    def _obstacle_ahead(self, rx, ry, yaw) -> bool:
        if self._scan_blocked(rx, ry, yaw):
            return True
        d = 0.2
        while d <= self._obstacle_dist + 1e-6:
            c = self._road_cost(rx + d * math.cos(yaw), ry + d * math.sin(yaw))
            if c != -1 and c >= self._safe_cost_max:
                return True
            d += 0.2
        return False

    def _scan_blocked(self, rx, ry, yaw) -> bool:
        """Live-scan-only forward-cone check (ground truth, costmap-independent)."""
        half_cone = math.radians(self._obstacle_cone) / 2.0
        if self._scan_pts_map is not None and len(self._scan_pts_map) > 0:
            dx = self._scan_pts_map[:, 0] - rx
            dy = self._scan_pts_map[:, 1] - ry
            dist = np.hypot(dx, dy)
            ang  = np.arctan2(dy, dx)
            rel  = np.abs(np.arctan2(np.sin(ang - yaw), np.cos(ang - yaw)))
            if np.any((rel < half_cone) & (dist < self._obstacle_dist)):
                return True
        return False

    def _spin_clearance_ok(self) -> bool:
        """True when a footprint-sized disk around the robot is free of lethal
        cells (road + pothole + local costmap) and live scan returns. This is
        the gate for 'can we rotate in place without nav2 aborting the spin'."""
        pose = self._map_pose()
        if pose is None:
            return False
        rx, ry, _ = pose
        R = self._spin_clear_r
        Rt = None
        if self._local_grid is not None and self._local_frame not in ('map', ''):
            Rt = self._lookup_R_t(self._local_frame)
        for deg in range(0, 360, 20):
            a = math.radians(deg)
            px = rx + R * math.cos(a)
            py = ry + R * math.sin(a)
            rc = self._road_cost(px, py)
            if rc != -1 and rc >= self._safe_cost_max:
                return False
            pc = self._pothole_cost(px, py)
            if pc != -1 and pc >= self._pothole_cost_max:
                return False
            lc = self._local_cost_at(px, py, Rt)
            if lc != -1 and lc >= self._local_cost_max:
                return False
        if self._scan_pts_map is not None and len(self._scan_pts_map) > 0:
            d = np.hypot(self._scan_pts_map[:, 0] - rx, self._scan_pts_map[:, 1] - ry)
            if np.any(d < R):
                return False
        return True

    def _road_run_length(self, rx, ry, yaw) -> float:
        step = 0.2
        d = step
        run = 0.0
        while d <= self._road_run_max + 1e-6:
            if self._is_on_road(rx + d * math.cos(yaw), ry + d * math.sin(yaw)):
                run = d
                d += step
            else:
                break
        return run

    @staticmethod
    def _angular_coverage(yaws) -> float:
        total = 0.0
        for a, b in zip(yaws[:-1], yaws[1:]):
            total += abs(math.atan2(math.sin(b - a), math.cos(b - a)))
        return total

    # ====================================================================
    # Stuck detection
    # ====================================================================

    def _map_pose(self):
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
        rx_map, ry_map, yaw = pose

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

        if self._goal_set_time is not None and \
                (now - self._goal_set_time) < self._goal_settle_sec:
            self._pose_history.clear()
            return

        if self._recovery_end_time is not None and \
                (now - self._recovery_end_time) < self._recovery_cooldown:
            self._pose_history.clear()
            return

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

        if spread < self._stuck_move_m:
            obstacle = (not self._require_obstacle) or self._obstacle_ahead(rx_map, ry_map, yaw)
            if not obstacle:
                self.get_logger().debug(
                    '[MONITOR] low movement but no obstacle ahead - not stuck',
                    throttle_duration_sec=3.0)
                return
            if not self._stuck_reported and self._state == RS.IDLE:
                self.get_logger().warn(
                    f'[MONITOR] STUCK - moved only {spread:.3f}m in {age:.0f}s '
                    f'WITH obstacle ahead - starting recovery')
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
        elif self._state == RS.WAIT_CLEAR:
            self._tick_wait_clear()

    # ====================================================================
    # Open-loop motion tick (drives the reverse cmd_vel)
    # ====================================================================

    def _motion_tick(self):
        if not self._ol_active:
            return
        if self._state == RS.IDLE:           # cancelled mid-reverse
            self._ol_active = False
            self._stop_motion()
            return

        sx, sy = self._ol_start
        moved = math.hypot(self._robot_x - sx, self._robot_y - sy)

        # Rear guard from the LOCAL COSTMAP only (no rear lidar). Unknown is
        # passable; a positively-occupied cell just behind stops the reverse.
        rear_blocked = False
        pose = self._map_pose()
        if pose is not None:
            rx, ry, yaw = pose
            Rt = None
            if self._local_grid is not None and self._local_frame not in ('map', ''):
                Rt = self._lookup_R_t(self._local_frame)
            ryaw = yaw + math.pi
            px = rx + (self._safety_r + 0.10) * math.cos(ryaw)
            py = ry + (self._safety_r + 0.10) * math.sin(ryaw)
            rear_blocked = self._rear_cell_blocked(px, py, Rt)

        elapsed = (self.get_clock().now() - self._ol_start_time).nanoseconds / 1e9
        max_t = (self._ol_target / max(0.01, self._backup_speed)) * 3.0 + 1.5
        timed_out = elapsed > max_t

        if moved >= self._ol_target or rear_blocked or timed_out:
            self._stop_motion()
            self._ol_active = False
            if rear_blocked:
                self.get_logger().warn(
                    f'[BACKING_UP] open-loop stopped early - rear costmap blocked at {moved:.2f}m')
            elif timed_out:
                self.get_logger().warn(
                    f'[BACKING_UP] open-loop timeout at {moved:.2f}m (is cmd_vel reaching the controller?)')
            self._finish_backup_increment(moved)
            return

        self._pub_cmd(-abs(self._backup_speed), 0.0)

    def _pub_cmd(self, vx: float, wz: float):
        if self._use_stamped_cmd:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'base_link'
            m.twist.linear.x = float(vx)
            m.twist.angular.z = float(wz)
        else:
            m = Twist()
            m.linear.x = float(vx)
            m.angular.z = float(wz)
        self._cmd_pub.publish(m)

    def _stop_motion(self):
        # Send a couple of zeros to be sure the controller latches a stop.
        self._pub_cmd(0.0, 0.0)

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
        self._backup_ineffective  = False
        self._spin_ineffective    = False
        self._phase_start_pose    = None
        self._active_goal_handle  = None
        self._wait_hold           = None
        self._wait_start          = None
        self._wait_streak         = 0
        self._total_backed_up     = 0.0
        self._spin_retry          = 0
        self._ol_active           = False
        self._ol_after            = None
        self._state = RS.CHECK_REAR
        self._active_pub.publish(Bool(data=True))
        self.get_logger().warn('>>> [RECOVERY] entering MAKE_ROOM (back up until we can spin)')
        self._make_room_then_spin()

    def _cancel_recovery(self, reason: str):
        if self._state == RS.IDLE:
            return
        self.get_logger().warn(f'[RECOVERY] CANCELLED ({reason}) - returning control to lane node')
        self._ol_active = False
        self._stop_motion()
        h = self._active_goal_handle
        self._active_goal_handle = None
        if h is not None:
            try:
                h.cancel_goal_async()
            except Exception as e:
                self.get_logger().error(f'[RECOVERY] cancel_goal failed: {e}')
        self._end_recovery()

    # ====================================================================
    # Phase: MAKE_ROOM  (back up in steps until we can spin)
    # ====================================================================

    def _make_room_then_spin(self):
        """Back up in steps until a footprint-sized disk is clear, then spin.
        This is the heart of the fix: when the robot is jammed against the
        obstacle, nav2's BackUp/Spin both abort on 'Collision Ahead', so we
        reverse a little (open-loop, bypassing nav2's collision gate), re-check
        spin clearance, and repeat until we can rotate - then spin."""
        self._state = RS.CHECK_REAR

        if self._spin_clearance_ok():
            self.get_logger().info(
                f'[MAKE_ROOM] spin clearance OK (backed up {self._total_backed_up:.2f}m total) - SPINNING')
            self._start_spin_phase()
            return

        if self._backup_ineffective:
            self.get_logger().warn(
                '[MAKE_ROOM] backup not moving the robot (wedged front+rear) - '
                'spinning anyway (will likely fall through to face-goal + wait)')
            self._start_spin_phase()
            return

        if self._total_backed_up >= self._max_total_backup:
            self.get_logger().warn(
                f'[MAKE_ROOM] reverse budget spent ({self._total_backed_up:.2f}/'
                f'{self._max_total_backup:.2f}m) but still no spin room - spinning anyway')
            self._start_spin_phase()
            return

        pose = self._map_pose()
        if pose is None:
            self.get_logger().error('[MAKE_ROOM] no TF - spinning')
            self._start_spin_phase()
            return
        rx, ry, yaw = pose
        rear_room = self._safe_backup_distance(rx, ry, yaw, self._backup_step)
        if rear_room < self._backup_min_dist:
            self.get_logger().warn(
                '[MAKE_ROOM] obstacle ahead and rear blocked in costmap - boxed in. '
                'Facing the goal and waiting for the obstacle to clear.')
            gx = self._final_goal.pose.position.x if self._final_goal is not None else None
            gy = self._final_goal.pose.position.y if self._final_goal is not None else None
            self._face_goal_fallback(gx, gy)
            return

        step = min(rear_room, self._backup_step)
        self.get_logger().warn(
            f'[MAKE_ROOM] not enough room to spin - reversing {step:.2f}m to make room '
            f'(total {self._total_backed_up:.2f}/{self._max_total_backup:.2f}m)')
        self._start_backup_increment(step, after=self._make_room_then_spin)

    # ====================================================================
    # Phase: BACKING_UP  (open-loop by default, nav2 action optional)
    # ====================================================================

    def _start_backup_increment(self, dist, after):
        self._state    = RS.BACKING_UP
        self._ol_after = after
        if self._open_loop_backup:
            self._ol_start      = (self._robot_x, self._robot_y)
            self._ol_target     = float(dist)
            self._ol_start_time = self.get_clock().now()
            self._ol_active     = True
            self.get_logger().warn(
                f'[BACKING_UP] open-loop reverse {dist:.2f}m @ {self._backup_speed:.2f}m/s '
                f'(bypasses nav2 front-collision gate)')
        else:
            self._send_nav2_backup(dist)

    def _send_nav2_backup(self, dist):
        self._phase_start_pose = self._map_pose()
        if not self._backup_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('[BACKING_UP] /backup server unavailable')
            self._finish_backup_increment(0.0)
            return
        goal          = BackUp.Goal()
        goal.target.x = float(dist)
        goal.speed    = self._backup_speed
        self.get_logger().warn(f'[BACKING_UP] nav2 BackUp goal dist={dist:.2f}m')
        fut = self._backup_client.send_goal_async(goal)
        fut.add_done_callback(self._nav2_backup_response_cb)

    def _nav2_backup_response_cb(self, future):
        if self._state == RS.IDLE:
            return
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('[BACKING_UP] nav2 goal REJECTED')
            self._finish_backup_increment(0.0)
            return
        self._active_goal_handle = handle
        handle.get_result_async().add_done_callback(self._nav2_backup_done_cb)

    def _nav2_backup_done_cb(self, future):
        if self._state == RS.IDLE:
            return
        self._active_goal_handle = None
        moved = 0.0
        end = self._map_pose()
        if end is not None and self._phase_start_pose is not None:
            moved = math.hypot(end[0] - self._phase_start_pose[0],
                               end[1] - self._phase_start_pose[1])
        self._finish_backup_increment(moved)

    def _finish_backup_increment(self, moved):
        """Common completion for both open-loop and nav2 backups. Accumulates
        the reverse budget, flags an ineffective (wheels-blocked) backup, and
        runs whatever was queued to happen next (re-check room, or retry spin)."""
        self._total_backed_up += moved
        if moved < self._min_eff_move:
            self._backup_ineffective = True
            self.get_logger().warn(
                f'[BACKING_UP] INEFFECTIVE - robot moved only {moved:.2f}m (blocked behind too?)')
        else:
            self.get_logger().info(
                f'[BACKING_UP] moved {moved:.2f}m (total {self._total_backed_up:.2f}m)')
        after = self._ol_after
        self._ol_after = None
        if after is not None and self._state != RS.IDLE:
            after()

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
        if self._state != RS.SPINNING:
            return
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

    def _spin_response_cb(self, future):
        if self._state == RS.IDLE:
            return
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('[SPINNING] goal REJECTED')
            self._do_evaluate()
            return
        self._active_goal_handle = handle
        self.get_logger().info('[SPINNING] goal accepted - spinning 360 deg')
        handle.get_result_async().add_done_callback(self._spin_done_cb)

    def _spin_done_cb(self, future):
        if self._state == RS.IDLE:
            return
        self._active_goal_handle = None
        result = future.result()
        yaws = [s[0] for s in self._scan_samples]
        coverage = self._angular_coverage(yaws) if len(yaws) >= 2 else 0.0

        if coverage < self._min_spin_rot:
            self._spin_ineffective = True
            self._spin_retry += 1
            self.get_logger().warn(
                f'[SPINNING] INEFFECTIVE - only {math.degrees(coverage):.0f} deg observed '
                f'(collision/abort). retry {self._spin_retry}/{self._max_spin_retries}')

            # The spin aborted on collision -> we still don't have room. Back up
            # a bit more and retry the spin, if budget / rear / retries allow.
            if (self._spin_retry <= self._max_spin_retries
                    and not self._backup_ineffective
                    and self._total_backed_up < self._max_total_backup):
                pose = self._map_pose()
                if pose is not None:
                    rx, ry, yaw = pose
                    rear_room = self._safe_backup_distance(rx, ry, yaw, self._backup_step)
                    if rear_room >= self._backup_min_dist:
                        step = min(rear_room, self._backup_step)
                        self.get_logger().warn(
                            f'[SPINNING] still no room - reversing another {step:.2f}m, then retry spin')
                        self._start_backup_increment(step, after=self._start_spin_phase)
                        return
                self.get_logger().warn(
                    '[SPINNING] cannot reverse further (rear blocked) - evaluating with what we have')
            else:
                self.get_logger().warn(
                    '[SPINNING] out of spin retries / reverse budget - evaluating with what we have')
        else:
            self._spin_ineffective = False
            self.get_logger().info(
                f'[SPINNING] complete (status={result.status}) '
                f'coverage={math.degrees(coverage):.0f} deg samples={len(self._scan_samples)}')

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

        truly_wedged = self._backup_ineffective and self._spin_ineffective

        pool = []
        for (yaw, clearance, rx, ry, on_road, safe) in self._scan_samples:
            if not (safe and on_road):
                continue
            run = self._road_run_length(rx, ry, yaw)
            if gx is not None:
                bearing = math.atan2(gy - ry, gx - rx)
                align = math.cos(yaw - bearing)
            else:
                align = 0.0
            score = (self._goal_align_w * align
                     + self._clear_w * (clearance / 2.0)
                     + self._run_w * (min(run, self._road_run_max) / self._road_run_max))
            pool.append((score, yaw, rx, ry, clearance, align, run))

        strict = [c for c in pool if c[6] >= self._min_road_run]
        use = strict if strict else pool
        best = max(use, key=lambda c: c[0]) if use else None

        if best is not None:
            self.get_logger().info(
                f'[EVALUATING] round {self._escalation_count}: best heading '
                f'yaw={math.degrees(best[1]):.1f} deg align={best[5]:+.2f} '
                f'clearance={best[4]:.2f}m run={best[6]:.2f}m')
            if self._best_overall is None or best[0] > self._best_overall[0]:
                self._best_overall = best
        else:
            self.get_logger().warn(
                f'[EVALUATING] round {self._escalation_count}: no on-road heading '
                f'among {len(self._scan_samples)} samples')

        out_of_attempts = self._escalation_count >= self._max_escalations
        good_enough     = best is not None and best[4] >= self._min_accept_clear

        if good_enough:
            self.get_logger().info('[EVALUATING] acceptable on-road heading - committing')
            self._commit_escape(self._best_overall)
        elif truly_wedged:
            self.get_logger().warn(
                '[EVALUATING] wedged (no backup, no spin) - facing goal and waiting')
            self._face_goal_fallback(gx, gy)
        elif out_of_attempts:
            if self._best_overall is not None:
                self.get_logger().warn(
                    '[EVALUATING] out of attempts - committing best on-road heading found')
                self._commit_escape(self._best_overall)
            else:
                self.get_logger().warn(
                    '[EVALUATING] out of attempts and nothing on-road - facing goal')
                self._face_goal_fallback(gx, gy)
        else:
            self._escalation_count    += 1
            self._current_backup_dist += self._backup_extra
            self.get_logger().warn(
                f'[EVALUATING] no good on-road heading - escalating '
                f'(attempt {self._escalation_count}/{self._max_escalations})')
            self._make_room_then_spin()

    def _commit_escape(self, best):
        _, yaw, rx, ry, clearance = best[0], best[1], best[2], best[3], best[4]
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
            f'heading={math.degrees(best_yaw):.1f} deg clearance={best_cl:.2f}m')

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

        facing  = heading_err <= self._settle_heading_tol
        reached = dist_to_target <= 0.3

        if elapsed >= self._settle_timeout:
            self.get_logger().warn(
                f'[SETTLING] timeout ({self._settle_timeout}s) - finishing recovery')
            self._finish_recovery()
            return

        if elapsed >= self._settle_min_sec and (facing or reached):
            self.get_logger().info(
                f'[SETTLING] oriented (heading_err={math.degrees(heading_err):.1f} deg) '
                f'- finishing recovery')
            self._finish_recovery()
            return

        self._publish_carrot(cx, cy, rx, ry)

    # ====================================================================
    # Finish: hand back NOW vs WAIT for a clear path
    # ====================================================================

    def _finish_recovery(self):
        if not self._wait_for_clear:
            self._end_recovery()
            return
        if self._goal_path_clear():
            self.get_logger().info('[FINISH] path toward goal already clear - handing back to lane node')
            self._end_recovery()
            return
        self._enter_wait_clear()

    def _goal_path_clear(self) -> bool:
        pose = self._map_pose()
        if pose is None:
            return False
        rx, ry, yaw = pose
        if self._final_goal is not None:
            gx = self._final_goal.pose.position.x
            gy = self._final_goal.pose.position.y
            if math.hypot(gx - rx, gy - ry) > 1e-3:
                yaw = math.atan2(gy - ry, gx - rx)
        if self._wait_use_costmap:
            return not self._obstacle_ahead(rx, ry, yaw)
        return not self._scan_blocked(rx, ry, yaw)

    # ====================================================================
    # Phase: WAIT_CLEAR  (hold until the path toward the goal clears)
    # ====================================================================

    def _enter_wait_clear(self):
        pose = self._map_pose()
        if pose is None:
            self.get_logger().error('[WAIT_CLEAR] no TF - handing back to lane node')
            self._end_recovery()
            return
        self._state       = RS.WAIT_CLEAR
        self._wait_hold   = (pose[0], pose[1])
        self._wait_start  = self.get_clock().now()
        self._wait_streak = 0
        self._active_pub.publish(Bool(data=True))
        self.get_logger().warn(
            '>>> [WAIT_CLEAR] obstacle still ahead toward goal - HOLDING. '
            'Lane node stays silent until the path clears.')

    def _tick_wait_clear(self):
        if self._wait_hold is None:
            self.get_logger().warn('[WAIT_CLEAR] no hold pose - handing back')
            self._end_recovery()
            return

        hx, hy = self._wait_hold
        self._publish_hold(hx, hy)

        if self._goal_path_clear():
            self._wait_streak += 1
        else:
            self._wait_streak = 0

        if self._wait_streak >= self._wait_confirm_n:
            self.get_logger().info(
                f'[WAIT_CLEAR] path clear for {self._wait_streak} ticks - handing back to lane node')
            self._end_recovery()
            return

        if self._wait_timeout > 0.0:
            elapsed = (self.get_clock().now() - self._wait_start).nanoseconds / 1e9
            if elapsed >= self._wait_timeout:
                self.get_logger().warn(
                    f'[WAIT_CLEAR] timeout ({self._wait_timeout:.0f}s) - handing back anyway')
                self._end_recovery()
                return

    def _publish_hold(self, hx: float, hy: float):
        pose = self._map_pose()
        if self._wait_face_goal and pose is not None and self._final_goal is not None:
            gx = self._final_goal.pose.position.x
            gy = self._final_goal.pose.position.y
            yaw = math.atan2(gy - pose[1], gx - pose[0])
        elif pose is not None:
            yaw = pose[2]
        else:
            yaw = 0.0
        msg = PoseStamped()
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.header.frame_id    = 'map'
        msg.pose.position.x    = hx
        msg.pose.position.y    = hy
        msg.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.orientation.w = math.cos(yaw / 2)
        self._pub.publish(msg)

    # ====================================================================
    # End recovery
    # ====================================================================

    def _end_recovery(self):
        self._ol_active          = False
        self._stop_motion()
        self._state              = RS.IDLE
        self._settle_target      = None
        self._settle_target_yaw  = None
        self._scan_samples       = []
        self._best_overall       = None
        self._stuck_reported     = False
        self._active_goal_handle = None
        self._backup_ineffective = False
        self._spin_ineffective   = False
        self._phase_start_pose   = None
        self._wait_hold          = None
        self._wait_start         = None
        self._wait_streak        = 0
        self._total_backed_up    = 0.0
        self._spin_retry         = 0
        self._ol_after           = None
        self._pose_history.clear()
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