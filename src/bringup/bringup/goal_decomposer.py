"""
goal_decomposer.py  (v5 — path-aware)
--------------------------------------
HOW PROFESSIONALS DO IT:
  1. User sets a goal (RViz or terminal topic).
  2. Ask Nav2's global planner (SMAC Hybrid-A*) to compute a
     collision-free, curve-aware path that respects the costmap.
  3. Sample waypoints every `path_sample_dist` metres along that path.
  4. Navigate to each waypoint in sequence using /navigate_to_pose.
  5. Gate-plane crossing (not Nav2 success) is the primary trigger to
     advance — robot doesn't need to stop exactly at each waypoint.
  6. If the planner fails (e.g. goal out of map), fall back to a
     direct single-waypoint send so the robot still does something.

KEY FIXES over all previous versions:
  - Uses /compute_path_to_pose (SMAC planner, not NavFn) for track-aware routing.
  - dispatch_delay prevents instant status=6 cascade.
  - _advancing flag prevents double-advance race condition.
  - Accepts goals on /goal_pose AND /goal_decomposer/goal (for terminal use).
  - Planner request retried once if first attempt returns empty path
    (handles the case where the costmap hasn't finished building yet).
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import NavigateToPose, ComputePathToPose
from std_msgs.msg import String
import tf_transformations


class GoalDecomposerNode(Node):

    def __init__(self):
        super().__init__('goal_decomposer_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('path_sample_dist',    1.5)
        self.declare_parameter('gate_dist',           0.8)
        self.declare_parameter('wp_timeout_sec',      25.0)
        self.declare_parameter('min_goal_dist',       0.3)
        self.declare_parameter('max_retries',         1)
        self.declare_parameter('dispatch_delay_sec',  0.4)
        # If planner fails, wait this long then retry planning once
        self.declare_parameter('plan_retry_delay_sec', 3.0)

        self.sample_dist       = self.get_parameter('path_sample_dist').value
        self.gate_dist         = self.get_parameter('gate_dist').value
        self.wp_timeout        = self.get_parameter('wp_timeout_sec').value
        self.min_goal_dist     = self.get_parameter('min_goal_dist').value
        self.max_retries       = int(self.get_parameter('max_retries').value)
        self.dispatch_delay    = self.get_parameter('dispatch_delay_sec').value
        self.plan_retry_delay  = self.get_parameter('plan_retry_delay_sec').value

        # ── State ─────────────────────────────────────────────────────
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        self._waypoints:       list[PoseStamped] = []
        self._wp_idx           = 0
        self._navigating       = False
        self._advancing        = False
        self._retry_count      = 0
        self._last_goal_xy     = None
        self._wp_start_time    = None
        self._pending_goal_msg = None   # stored while waiting for planner
        self._plan_attempts    = 0
        self._pending_dispatch = None   # (wp_idx, retry, wall_time)

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_cb, 10)
        self.create_subscription(
            PoseStamped, '/goal_decomposer/goal', self._goal_cb, 10)
        self.create_subscription(
            Odometry, '/diff_drive_controller/odom', self._odom_cb, 10)

        # ── Publishers ────────────────────────────────────────────────
        self._status_pub  = self.create_publisher(String, '/goal_decomposer/status', 5)
        self._path_pub    = self.create_publisher(Path,   '/goal_decomposer/debug_path', 5)

        # ── Action clients ────────────────────────────────────────────
        self._plan_client = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')
        self._nav_client  = ActionClient(self, NavigateToPose,    '/navigate_to_pose')

        # ── Timers ────────────────────────────────────────────────────
        self.create_timer(0.1,  self._gate_check)
        self.create_timer(0.05, self._dispatch_check)

        self.get_logger().info(
            f'GoalDecomposerNode v5 (path-aware)  '
            f'sample={self.sample_dist}m  gate={self.gate_dist}m')
        self.get_logger().info(
            'Terminal goal: ros2 topic pub --once /goal_decomposer/goal '
            'geometry_msgs/msg/PoseStamped '
            '"{header: {frame_id: map}, pose: {position: {x: 26.0, y: -7.0, z: 0.0}, '
            'orientation: {w: 1.0}}}"')

    # ── Odometry ──────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w])

    # ── Goal callback ─────────────────────────────────────────────────
    def _goal_cb(self, msg: PoseStamped):
        gx = msg.pose.position.x
        gy = msg.pose.position.y

        if self._last_goal_xy is not None:
            if math.hypot(gx - self._last_goal_xy[0],
                          gy - self._last_goal_xy[1]) < self.min_goal_dist:
                return

        self._last_goal_xy = (gx, gy)
        dist = math.hypot(gx - self.robot_x, gy - self.robot_y)
        self.get_logger().info(
            f'New goal ({gx:.2f}, {gy:.2f})  dist={dist:.2f}m  — requesting path...')
        self._pub_status(f'planning to ({gx:.1f},{gy:.1f}) dist={dist:.1f}m')

        # Reset everything
        self._reset_state()
        self._pending_goal_msg = msg
        self._plan_attempts    = 0

        self._request_plan(msg)

    def _reset_state(self):
        self._waypoints.clear()
        self._wp_idx          = 0
        self._navigating      = False
        self._advancing       = False
        self._retry_count     = 0
        self._pending_dispatch = None
        self._wp_start_time   = None

    # ── Planning ──────────────────────────────────────────────────────
    def _request_plan(self, goal_pose: PoseStamped):
        if not self._plan_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn(
                '/compute_path_to_pose unavailable — falling back to direct nav')
            self._fallback_direct(goal_pose)
            return

        plan_goal              = ComputePathToPose.Goal()
        plan_goal.goal         = goal_pose
        plan_goal.planner_id   = 'GridBased'
        plan_goal.use_start    = False

        self._plan_attempts += 1
        self.get_logger().info(
            f'Requesting path (attempt {self._plan_attempts})...')

        fut = self._plan_client.send_goal_async(plan_goal)
        fut.add_done_callback(self._plan_response_cb)

    def _plan_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Plan request rejected')
            self._handle_plan_failure()
            return
        handle.get_result_async().add_done_callback(self._plan_result_cb)

    def _plan_result_cb(self, future):
        result = future.result()
        path   = result.result.path

        if not path.poses:
            self.get_logger().warn('Planner returned empty path')
            self._handle_plan_failure()
            return

        self.get_logger().info(
            f'Path received: {len(path.poses)} poses')

        # Publish debug path to visualise in RViz
        self._path_pub.publish(path)

        # Sample waypoints along the path
        self._waypoints = self._sample_path(path)
        self.get_logger().info(
            f'Sampled {len(self._waypoints)} waypoints  '
            f'(every {self.sample_dist}m along planned path)')

        self._pub_status(f'path OK → {len(self._waypoints)} waypoints')
        self._schedule_dispatch(0, 0)

    def _handle_plan_failure(self):
        if self._plan_attempts < 2 and self._pending_goal_msg is not None:
            # Retry once after delay (costmap may not be built yet)
            self.get_logger().info(
                f'Retrying plan in {self.plan_retry_delay}s...')
            send_at = (self.get_clock().now().nanoseconds / 1e9
                       + self.plan_retry_delay)
            # Store as a special pending dispatch with negative index = plan retry
            self._pending_dispatch = (-1, 0, send_at)
        else:
            self.get_logger().warn(
                'Planning failed after retries — falling back to direct nav')
            self._fallback_direct(self._pending_goal_msg)

    def _fallback_direct(self, goal_pose: PoseStamped):
        """Send the goal directly without waypoint decomposition."""
        if goal_pose is None:
            return
        self.get_logger().warn('Fallback: sending goal directly to Nav2')
        self._waypoints = [goal_pose]
        self._schedule_dispatch(0, 0)

    # ── Path sampling ─────────────────────────────────────────────────
    def _sample_path(self, path: Path) -> list[PoseStamped]:
        """
        Sample poses every `sample_dist` metres along the PLANNED path.
        This preserves curves — waypoints follow the track shape, not
        a straight line.
        """
        poses  = path.poses
        result = []
        accum  = 0.0

        for i in range(1, len(poses)):
            px = poses[i-1].pose.position.x
            py = poses[i-1].pose.position.y
            cx = poses[i].pose.position.x
            cy = poses[i].pose.position.y
            accum += math.hypot(cx - px, cy - py)

            if accum >= self.sample_dist:
                wp = PoseStamped()
                wp.header         = poses[i].header
                wp.pose           = poses[i].pose
                result.append(wp)
                accum = 0.0

        # Always include exact final goal pose
        if poses:
            final = PoseStamped()
            final.header = poses[-1].header
            final.pose   = poses[-1].pose
            result.append(final)

        return result

    # ── Dispatch system ───────────────────────────────────────────────
    def _schedule_dispatch(self, wp_idx: int, retry: int):
        send_at = self.get_clock().now().nanoseconds / 1e9 + self.dispatch_delay
        self._pending_dispatch = (wp_idx, retry, send_at)

    def _dispatch_check(self):
        if self._pending_dispatch is None:
            return
        wp_idx, retry, send_at = self._pending_dispatch
        now = self.get_clock().now().nanoseconds / 1e9
        if now < send_at:
            return

        self._pending_dispatch = None

        if wp_idx == -1:
            # Plan retry
            if self._pending_goal_msg is not None:
                self._request_plan(self._pending_goal_msg)
        else:
            self._do_send_waypoint(wp_idx, retry)

    # ── Navigation ────────────────────────────────────────────────────
    def _do_send_waypoint(self, wp_idx: int, retry: int):
        if not self._waypoints or wp_idx >= len(self._waypoints):
            return

        self._wp_idx      = wp_idx
        self._retry_count = retry
        self._advancing   = False

        if not self._nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('/navigate_to_pose not available')
            return

        wp = self._waypoints[wp_idx]
        self.get_logger().info(
            f'→ wp {wp_idx + 1}/{len(self._waypoints)}: '
            f'({wp.pose.position.x:.2f}, {wp.pose.position.y:.2f})  retry={retry}')
        self._pub_status(
            f'wp {wp_idx + 1}/{len(self._waypoints)} '
            f'({wp.pose.position.x:.1f},{wp.pose.position.y:.1f})')

        goal_msg      = NavigateToPose.Goal()
        goal_msg.pose = wp

        self._navigating    = True
        self._wp_start_time = self.get_clock().now()

        fut = self._nav_client.send_goal_async(goal_msg)
        fut.add_done_callback(self._nav_response_cb)

    def _nav_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn(f'wp {self._wp_idx + 1} rejected')
            self._on_wp_done(success=False)
            return
        handle.get_result_async().add_done_callback(self._nav_result_cb)

    def _nav_result_cb(self, future):
        result = future.result()
        status = result.status   # 4=SUCCEEDED 5=CANCELED 6=ABORTED

        if status == 5:
            return   # intentionally canceled, ignore

        self._navigating = False

        if status == 4:
            self.get_logger().info(f'Nav2 succeeded wp {self._wp_idx + 1}')
            self._on_wp_done(success=True)
        else:
            self.get_logger().warn(
                f'wp {self._wp_idx + 1} aborted  retry={self._retry_count}/{self.max_retries}')
            if self._retry_count < self.max_retries:
                self._schedule_dispatch(self._wp_idx, self._retry_count + 1)
            else:
                self.get_logger().warn(f'Skipping wp {self._wp_idx + 1}')
                self._on_wp_done(success=False)

    # ── Waypoint advancement ──────────────────────────────────────────
    def _on_wp_done(self, success: bool):
        if self._advancing:
            return
        self._advancing = True

        next_idx = self._wp_idx + 1
        if next_idx >= len(self._waypoints):
            self.get_logger().info('All waypoints complete — mission done!')
            self._pub_status('MISSION COMPLETE')
            self._reset_state()
        else:
            self._schedule_dispatch(next_idx, 0)

    # ── Gate check ────────────────────────────────────────────────────
    def _gate_check(self):
        if (not self._waypoints or
                self._wp_idx >= len(self._waypoints) or
                not self._navigating or
                self._advancing):
            return

        wp  = self._waypoints[self._wp_idx]
        wx  = wp.pose.position.x
        wy  = wp.pose.position.y
        q   = wp.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w])

        fwd_x  = math.cos(yaw)
        fwd_y  = math.sin(yaw)
        gate_x = wx - self.gate_dist * fwd_x
        gate_y = wy - self.gate_dist * fwd_y

        dot = ((self.robot_x - gate_x) * fwd_x +
               (self.robot_y - gate_y) * fwd_y)

        if dot >= 0:
            self.get_logger().info(
                f'Gate passed: wp {self._wp_idx + 1}/{len(self._waypoints)} '
                f'({wx:.2f}, {wy:.2f})')
            self._on_wp_done(success=True)
            return

        # Timeout
        if self._wp_start_time is not None:
            elapsed = (self.get_clock().now() -
                       self._wp_start_time).nanoseconds / 1e9
            if elapsed > self.wp_timeout:
                self.get_logger().warn(
                    f'wp {self._wp_idx + 1} timed out — skipping')
                self._on_wp_done(success=False)

    # ── Status ────────────────────────────────────────────────────────
    def _pub_status(self, msg: str):
        s = String()
        s.data = msg
        self._status_pub.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = GoalDecomposerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()