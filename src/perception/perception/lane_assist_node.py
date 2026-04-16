"""
lane_assist_node.py
-------------------
Sits between Nav2's controller_server output and the robot drive:

  Nav2  →  /cmd_vel  →  [lane_assist_node]  →  /cmd_vel_nav  →  [twist_to_stamped]

It subscribes to:
  /cmd_vel              – raw velocity from Nav2 (Twist)
  /lane_center_error    – pixel offset from lane_detection node (Float32)
  /lane_visible         – whether a lane was found this frame (Bool)

It publishes:
  /cmd_vel_nav          – corrected Twist (linear unchanged, angular nudged)

Control law:
  normalised  = error / image_half_width          → [-1, 1]
  correction  = Kp * normalised - Kd * d(normalised)/dt

  PD controller: P term drives the robot to centre, D term damps
  oscillation by penalising rapid changes in error.

  Dead-band: errors smaller than dead_band_px are ignored so the node
  doesn't fight Nav2 when the robot is already nearly centred.

Safety:
  • If /lane_visible is False the node passes cmd_vel through unchanged.
  • Correction is clamped to ±max_correction so it can never override
    Nav2's full steering authority.
  • If no lane message has arrived for > timeout_sec the node passes
    cmd_vel through unchanged (stale / camera dead).
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Duration
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool


class LaneAssistNode(Node):

    def __init__(self):
        super().__init__('lane_assist_node')

        # ── tunable parameters ────────────────────────────────────────
        # NOTE: use_sim_time is a built-in ROS 2 parameter — do NOT declare it

        # Proportional gain. Lowered from 0.3 to reduce oscillation.
        self.declare_parameter('Kp', 0.15)

        # Derivative gain. Damps oscillation by penalising rapid error change.
        self.declare_parameter('Kd', 0.05)

        # Max angular correction magnitude (rad/s).
        self.declare_parameter('max_correction', 0.3)

        # Image half-width used for normalisation — must match bev_width / 2
        # in lane_detection_node.
        self.declare_parameter('image_half_width', 320.0)

        # Errors smaller than this (pixels) are ignored — prevents fighting
        # Nav2 when the robot is already close to centre.
        self.declare_parameter('dead_band_px', 20.0)

        # If lane_visible is False for this many seconds, pass through unchanged.
        self.declare_parameter('timeout_sec', 0.3)

        self.Kp             = self.get_parameter('Kp').value
        self.Kd             = self.get_parameter('Kd').value
        self.max_correction = self.get_parameter('max_correction').value
        self.half_w         = self.get_parameter('image_half_width').value
        self.dead_band_px   = self.get_parameter('dead_band_px').value
        self.timeout_sec    = self.get_parameter('timeout_sec').value

        # ── state ─────────────────────────────────────────────────────
        self.lane_error      = 0.0
        self.prev_error      = 0.0        # for D term
        self.lane_visible    = False
        self.last_lane_stamp = None       # rclpy.time.Time

        # ── subscribers ───────────────────────────────────────────────
        self.create_subscription(Twist,   '/cmd_vel',           self.cmd_vel_cb, 10)
        self.create_subscription(Float32, '/lane_center_error', self.error_cb,   10)
        self.create_subscription(Bool,    '/lane_visible',      self.visible_cb, 10)

        # ── publisher ─────────────────────────────────────────────────
        self.pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        self.get_logger().info(
            f'LaneAssistNode started  Kp={self.Kp}  Kd={self.Kd}  '
            f'max_corr={self.max_correction}  dead_band={self.dead_band_px}px')

    # ── lane callbacks ────────────────────────────────────────────────
    def error_cb(self, msg: Float32):
        self.lane_error = msg.data
        self.last_lane_stamp = self.get_clock().now()

    def visible_cb(self, msg: Bool):
        self.lane_visible = msg.data

    # ── main control callback ─────────────────────────────────────────
    def cmd_vel_cb(self, msg: Twist):
        corrected = Twist()
        corrected.linear  = msg.linear
        corrected.angular = msg.angular

        # Check staleness
        if self.last_lane_stamp is not None:
            age = self.get_clock().now() - self.last_lane_stamp
            stale = age > Duration(seconds=self.timeout_sec)
        else:
            stale = True

        if self.lane_visible and not stale:
            # Dead-band: skip correction when already close to centre
            if abs(self.lane_error) < self.dead_band_px:
                self.prev_error = self.lane_error
                self.pub.publish(corrected)
                return

            half_w = max(self.half_w, 1.0)

            # P term
            normalised  = self.lane_error / half_w

            # D term — derivative of normalised error
            derivative  = (self.lane_error - self.prev_error) / half_w

            correction  = self.Kp * normalised - self.Kd * derivative

            # Clamp
            correction = max(-self.max_correction,
                             min(self.max_correction, correction))

            corrected.angular.z += correction

            self.get_logger().debug(
                f'err={self.lane_error:.1f}px  norm={normalised:.3f}  '
                f'deriv={derivative:.3f}  corr={correction:.3f}  '
                f'final_w={corrected.angular.z:.3f}')

            self.prev_error = self.lane_error

        else:
            # Reset D term when lane is lost so there's no derivative spike
            # on re-acquisition
            self.prev_error = 0.0

        self.pub.publish(corrected)


def main(args=None):
    rclpy.init(args=args)
    node = LaneAssistNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()