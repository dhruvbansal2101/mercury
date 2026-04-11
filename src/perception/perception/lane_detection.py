"""
lane_detection.py
-----------------
Camera: xyz="0.15 0 2" rpy="0 0.4 0"

Single-lane recovery logic:
  When only the LEFT lane is visible:
    lane_cx = left_cx + calibrated_lane_width / 2
    error   = img_cx - lane_cx  →  negative  →  robot steers RIGHT ✓

  When only the RIGHT lane is visible:
    lane_cx = right_cx - calibrated_lane_width / 2
    error   = img_cx - lane_cx  →  positive  →  robot steers LEFT ✓

  The calibrated_lane_width is a rolling median from recent frames where
  BOTH lanes were visible (genuine separation > min_lane_separation_px).
  This makes single-lane extrapolation accurate even after a re-tune.

Fragment check:
  If two blobs are < min_lane_separation_px apart horizontally they are
  top/bottom fragments of the SAME line → merged into one centroid before
  the left/right classification above is applied.
"""

import cv2
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge


class LaneDetectionNode(Node):

    def __init__(self):
        super().__init__('lane_detection_node')

        self.declare_parameter('image_topic',           '/camera/image_raw')
        self.declare_parameter('bev_width',              640)
        self.declare_parameter('bev_height',             480)
        self.declare_parameter('show_debug',             True)
        self.declare_parameter('ema_alpha',              0.35)
        self.declare_parameter('min_blob_area',          2000.0)
        self.declare_parameter('min_lane_separation_px', 150.0)

        # Fallback lane width used ONLY before any dual-lane frame is seen.
        # Once both lanes have been seen, the auto-calibrated median is used.
        self.declare_parameter('lane_width_px',          560.0)

        self.bev_w      = self.get_parameter('bev_width').value
        self.bev_h      = self.get_parameter('bev_height').value
        self.show_debug = self.get_parameter('show_debug').value
        self.ema_alpha  = self.get_parameter('ema_alpha').value
        self.min_area   = self.get_parameter('min_blob_area').value
        self.min_sep    = self.get_parameter('min_lane_separation_px').value
        self._lw_param  = self.get_parameter('lane_width_px').value
        image_topic     = self.get_parameter('image_topic').value

        self.bridge     = CvBridge()
        self.homography = None
        self._ema_error = 0.0

        # Rolling buffer of lane widths measured from genuine dual-lane frames
        self._lw_buf: deque = deque(maxlen=60)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.image_sub   = self.create_subscription(
            Image, image_topic, self.image_callback, sensor_qos)
        self.error_pub   = self.create_publisher(Float32, '/lane_center_error', 10)
        self.visible_pub = self.create_publisher(Bool,    '/lane_visible',      10)
        self.debug_pub   = self.create_publisher(Image,   '/lane_debug/image',  sensor_qos)

        self.get_logger().info(
            f'LaneDetectionNode started  '
            f'min_sep={self.min_sep}px  '
            f'min_blob_area={self.min_area}px²  '
            f'fallback_lane_width={self._lw_param}px'
        )

    # ── calibrated lane width ─────────────────────────────────────────
    @property
    def _lane_width(self):
        """
        Returns the best available lane width estimate:
          - Median of recent dual-lane frames if we have enough samples
          - Parameter fallback otherwise
        """
        if len(self._lw_buf) >= 5:
            return float(np.median(self._lw_buf))
        return self._lw_param

    # ── homography ────────────────────────────────────────────────────
    def _compute_homography(self, frame):
        h, w = frame.shape[:2]
        src = np.float32([
            [w * 0.02,  h * 0.525],
            [w * 0.98,  h * 0.525],
            [w * 0.225, h * 0.30],
            [w * 0.775, h * 0.30],
        ])
        dst = np.float32([
            [0,          self.bev_h],
            [self.bev_w, self.bev_h],
            [0,          0],
            [self.bev_w, 0],
        ])
        self.homography = cv2.getPerspectiveTransform(src, dst)
        self.get_logger().info('Homography computed')

    # ── white-line mask ───────────────────────────────────────────────
    def _detect_white(self, bev):
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv,
                               np.array([0,   0, 180]),
                               np.array([180, 80, 255]))
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        _, mask_bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask_hsv, mask_bright)

        k_open    = cv2.getStructuringElement(cv2.MORPH_RECT, (3,   3))
        k_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5,  120))
        k_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (80,   5))
        k_dil     = cv2.getStructuringElement(cv2.MORPH_RECT, (7,   7))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close_v)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close_h)
        mask = cv2.dilate(mask, k_dil, iterations=1)
        mask[-20:, :] = 0
        return mask

    # ── contour blob finder ───────────────────────────────────────────
    def _find_lane_blobs(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            blobs.append((cx, area, cnt))
        blobs.sort(key=lambda b: b[1], reverse=True)
        blobs = blobs[:2]
        blobs.sort(key=lambda b: b[0])
        return blobs

    # ── error computation ─────────────────────────────────────────────
    def _compute_error(self, mask):
        img_cx = self.bev_w / 2.0
        blobs  = self._find_lane_blobs(mask)
        lw     = self._lane_width   # best available width estimate

        if len(blobs) == 0:
            return self._ema_error, False, 'none', []

        if len(blobs) == 2:
            left_cx  = blobs[0][0]
            right_cx = blobs[1][0]
            h_sep    = right_cx - left_cx

            if h_sep < self.min_sep:
                # ── fragment merge ────────────────────────────────────
                # Two blobs too close → same physical line, merge them
                total_area = blobs[0][1] + blobs[1][1]
                merged_cx  = (blobs[0][0] * blobs[0][1]
                              + blobs[1][0] * blobs[1][1]) / total_area
                blobs = [(merged_cx, total_area, None)]   # treat as one blob
                self.get_logger().debug(
                    f'[lane] Fragment merge sep={h_sep:.0f}px → '
                    f'merged_cx={merged_cx:.0f}')
            else:
                # ── genuine two-lane detection ────────────────────────
                # Update the calibrated lane width from this measurement
                self._lw_buf.append(h_sep)
                lane_cx = (left_cx + right_cx) / 2.0
                raw_err = img_cx - lane_cx
                self._ema_error = (self.ema_alpha * raw_err
                                   + (1.0 - self.ema_alpha) * self._ema_error)
                return self._ema_error, True, 'both', blobs

        # ── single blob (or post-merge single blob) ───────────────────
        # This is the recovery path: one lane visible, steer back to centre.
        #
        #   LEFT lane visible, robot drifted LEFT (or right lane gone):
        #     lane_cx = left_cx + lw/2  →  to the right of the visible line
        #     error   = img_cx - lane_cx  →  negative  →  lane_assist steers RIGHT ✓
        #
        #   RIGHT lane visible, robot drifted RIGHT (or left lane gone):
        #     lane_cx = right_cx - lw/2  →  to the left of the visible line
        #     error   = img_cx - lane_cx  →  positive  →  lane_assist steers LEFT ✓
        cx = blobs[0][0]
        if cx < img_cx:
            lane_cx = cx + lw / 2.0
            mode    = 'left_only' if len(blobs) == 1 else 'left_frag'
        else:
            lane_cx = cx - lw / 2.0
            mode    = 'right_only' if len(blobs) == 1 else 'right_frag'

        raw_err = img_cx - lane_cx
        self._ema_error = (self.ema_alpha * raw_err
                           + (1.0 - self.ema_alpha) * self._ema_error)
        return self._ema_error, True, mode, blobs

    # ── main callback ─────────────────────────────────────────────────
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        if self.homography is None:
            self._compute_homography(frame)

        bev  = cv2.warpPerspective(
            frame, self.homography, (self.bev_w, self.bev_h))
        mask = self._detect_white(bev)
        smooth_err, visible, mode, blobs = self._compute_error(mask)

        self.error_pub.publish(Float32(data=float(smooth_err)))
        self.visible_pub.publish(Bool(data=visible))

        # ── debug overlay ─────────────────────────────────────────────
        overlay     = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blob_colors = [(255, 100, 0), (0, 100, 255)]

        for i, (cx, area, cnt) in enumerate(blobs):
            if cnt is not None:
                color = blob_colors[i % 2]
                cv2.drawContours(overlay, [cnt], -1, color, 2)
            cv2.circle(overlay, (int(cx), self.bev_h // 2), 8,
                       blob_colors[i % 2], -1)

        # Draw separation line between two real blobs
        if len(blobs) == 2 and blobs[0][2] is not None:
            sep       = blobs[1][0] - blobs[0][0]
            sep_color = (0, 255, 0) if sep >= self.min_sep else (0, 0, 255)
            cv2.line(overlay,
                     (int(blobs[0][0]), self.bev_h // 2),
                     (int(blobs[1][0]), self.bev_h // 2),
                     sep_color, 2)
            mid_x = int((blobs[0][0] + blobs[1][0]) / 2)
            cv2.putText(overlay, f'sep={sep:.0f}px',
                        (mid_x - 35, self.bev_h // 2 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, sep_color, 1)

        if visible:
            img_cx  = self.bev_w // 2
            lane_cx = int(img_cx - smooth_err)
            cv2.line(overlay, (img_cx,  0), (img_cx,  self.bev_h),
                     (0, 255, 0), 2)
            cv2.line(overlay, (lane_cx, 0), (lane_cx, self.bev_h),
                     (0, 0, 255), 2)

        mode_colors = {
            'both':       (0,   255, 0),
            'left_only':  (0,   165, 255),
            'right_only': (0,   165, 255),
            'left_frag':  (0,   0,   255),
            'right_frag': (0,   0,   255),
            'none':       (128, 128, 128),
        }
        cv2.putText(
            overlay,
            f'err={smooth_err:.1f}px  [{mode}]  '
            f'lw={self._lane_width:.0f}px  blobs={len(blobs)}',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            mode_colors.get(mode, (255, 255, 0)), 2)

        debug_msg        = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

        if self.show_debug:
            cv2.imshow('BEV mask', overlay)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()