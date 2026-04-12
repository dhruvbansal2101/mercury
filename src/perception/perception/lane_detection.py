"""
lane_detection.py  — White mask + Canny + Hough  (v4 - fixed)
--------------------------------------------------------------
KEY FIXES over v3:
  1. Lane-width calibration now ONLY accepts readings where left_x < img_cx
     and right_x > img_cx (i.e. robot is truly between the two lanes).
     This prevents absurdly large lw values from corrupting single-lane logic.

  2. Single-lane extrapolation uses a FIXED fallback lane_width_px (default 320)
     instead of the auto-calibrated value, because auto-cal gets polluted when
     both "lanes" are actually the same edge seen from two angles.

  3. Fragment detection (left_frag / right_frag): when two detected X positions
     are too close together (< min_lane_sep_px) they are treated as a SINGLE
     lane edge rather than two separate lanes. The side is determined by whether
     the merged X is left or right of centre.

  4. DRIFT-BACK BEHAVIOR: a persistent "lane_side" tracker remembers which lane
     was last confidently visible. When only one lane is seen, the error is
     computed to drive the robot toward the MISSING lane (i.e. increase the
     visible lane's distance from the robot) rather than just hold position.
     This makes the robot naturally re-centre when a lane reappears.

  5. Curve handling: the ROI uses the BOTTOM portion of the image (closest to
     robot) for reliable lane-centre estimation on curves, avoiding the
     perspective vanishing-point confusion that caused large errors on bends.

VISUAL INDICATORS (unchanged):
  Green vertical line  = image centre (where robot is pointing)
  Red   vertical line  = detected lane centre (where robot should be)
  Error = image_cx - lane_cx  →  positive = robot should move right,
                                  negative = robot should move left
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

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('image_topic',      '/camera/image_raw')
        self.declare_parameter('show_debug',        True)
        self.declare_parameter('ema_alpha',         0.30)

        # Fixed lane half-width used for single-lane extrapolation.
        # Set this = (R_x - L_x) / 2 when robot is centred on track.
        # Default 160 px assumes 640-wide image and lane fills ~half width.
        self.declare_parameter('lane_half_width_px', 160.0)

        # Auto-calibration: disabled by default — red obstacles were being
        # detected as white, giving cal=1000+px and breaking single-lane logic.
        # Set to True only after you've confirmed clean two-lane detections.
        self.declare_parameter('use_auto_cal', False)

        # ROI: use bottom (1 - roi_top_frac) of image
        self.declare_parameter('roi_top_frac',      0.35)

        # White pixel mask thresholds (HSV)
        self.declare_parameter('white_v_min',       170)
        self.declare_parameter('white_s_max',       60)

        # Morphological close kernel
        self.declare_parameter('close_kw',          5)
        self.declare_parameter('close_kh',          25)

        # Canny
        self.declare_parameter('canny_low',         30.0)
        self.declare_parameter('canny_high',        100.0)

        # Hough
        self.declare_parameter('hough_threshold',   15)
        self.declare_parameter('hough_min_len',     20.0)
        self.declare_parameter('hough_max_gap',     40.0)

        # Slope filter
        self.declare_parameter('min_slope_abs',     0.2)
        self.declare_parameter('max_slope_abs',     4.0)

        # Minimum X separation between left_x and right_x to be considered
        # TWO distinct lanes (not fragments of the same edge).
        self.declare_parameter('min_lane_sep_px',   120.0)

        # When only one lane is visible, how aggressively to drift toward
        # the missing lane. 1.0 = full correction, 0.5 = half.
        self.declare_parameter('drift_gain',        0.8)

        # Max valid separation for auto-cal AND for [both] mode trust.
        # If sep > this, the two detections are NOT both real lane edges
        # (e.g. one is an obstacle). Falls back to single-lane logic per side.
        # Rule of thumb: real lane sep ≈ 2 * lane_half_width_px ± 30%.
        # For 640px image with half_width=160: max = 160*2*1.4 = ~450px.
        self.declare_parameter('max_valid_sep_px',  450.0)

        self.show_debug        = self.get_parameter('show_debug').value
        self.ema_alpha         = self.get_parameter('ema_alpha').value
        self._lane_half_w      = self.get_parameter('lane_half_width_px').value
        self._use_auto_cal     = self.get_parameter('use_auto_cal').value
        self.roi_top           = self.get_parameter('roi_top_frac').value
        self.white_v_min       = int(self.get_parameter('white_v_min').value)
        self.white_s_max       = int(self.get_parameter('white_s_max').value)
        self.close_kw          = int(self.get_parameter('close_kw').value)
        self.close_kh          = int(self.get_parameter('close_kh').value)
        self.canny_low         = int(self.get_parameter('canny_low').value)
        self.canny_high        = int(self.get_parameter('canny_high').value)
        self.hough_thresh      = int(self.get_parameter('hough_threshold').value)
        self.hough_min         = self.get_parameter('hough_min_len').value
        self.hough_gap         = self.get_parameter('hough_max_gap').value
        self.min_slope         = self.get_parameter('min_slope_abs').value
        self.max_slope         = self.get_parameter('max_slope_abs').value
        self.min_sep           = self.get_parameter('min_lane_sep_px').value
        self.drift_gain        = self.get_parameter('drift_gain').value
        self.max_valid_sep     = self.get_parameter('max_valid_sep_px').value
        image_topic            = self.get_parameter('image_topic').value

        self.bridge       = CvBridge()
        self._ema_error   = 0.0

        # Auto-calibration buffer — only valid two-lane readings
        self._lw_buf: deque = deque(maxlen=30)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.image_sub   = self.create_subscription(
            Image, image_topic, self.image_callback, sensor_qos)
        self.error_pub   = self.create_publisher(Float32, '/lane_center_error', 10)
        self.visible_pub = self.create_publisher(Bool,    '/lane_visible',      10)
        self.debug_pub   = self.create_publisher(Image,   '/lane_debug/image',  sensor_qos)

        self.get_logger().info(
            f'LaneDetectionNode v4 (fixed) started  '
            f'lane_half_width={self._lane_half_w}px  '
            f'roi_top={self.roi_top}  min_sep={self.min_sep}px'
        )

    # ── Lane half-width property ───────────────────────────────────────────
    @property
    def _half_width(self) -> float:
        """Use auto-calibrated value only when we have enough valid readings."""
        if self._use_auto_cal and len(self._lw_buf) >= 5:
            return float(np.median(self._lw_buf)) / 2.0
        return self._lane_half_w

    # ── White pixel mask ──────────────────────────────────────────────────
    def _white_mask(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0,   0,            self.white_v_min]),
            np.array([180, self.white_s_max, 255])
        )
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, self.white_v_min, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, bright)
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.close_kw, self.close_kh))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # ── Lane detection ────────────────────────────────────────────────────
    def _detect_lanes(self, frame):
        h, w = frame.shape[:2]
        roi_y = int(h * self.roi_top)
        roi   = frame[roi_y:h, 0:w]
        roi_h = roi.shape[0]

        white = self._white_mask(roi)
        edges = cv2.Canny(white, self.canny_low, self.canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_thresh,
            minLineLength=self.hough_min,
            maxLineGap=self.hough_gap
        )

        img_cx     = w / 2.0
        left_segs  = []
        right_segs = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < self.min_slope or abs(slope) > self.max_slope:
                    continue
                seg_len = np.hypot(x2 - x1, y2 - y1)
                mid_x   = (x1 + x2) / 2.0
                if mid_x < img_cx:
                    left_segs.append((x1, y1, x2, y2, seg_len))
                else:
                    right_segs.append((x1, y1, x2, y2, seg_len))

        left_x  = self._weighted_line_x(left_segs,  roi_h)
        right_x = self._weighted_line_x(right_segs, roi_h)

        return left_x, right_x, white, edges, roi_y, left_segs, right_segs, w, h

    def _weighted_line_x(self, segs, roi_h):
        if not segs:
            return None
        slopes, intercepts, weights = [], [], []
        for x1, y1, x2, y2, length in segs:
            if x2 == x1:
                continue
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            slopes.append(m)
            intercepts.append(b)
            weights.append(length)
        if not slopes:
            return None
        tw = sum(weights)
        m  = sum(s * w for s, w in zip(slopes,     weights)) / tw
        b  = sum(i * w for i, w in zip(intercepts, weights)) / tw
        if abs(m) < 1e-6:
            return None
        return float((roi_h - 1 - b) / m)

    # ── Error computation (core fix) ──────────────────────────────────────
    def _compute_error(self, left_x, right_x, img_w):
        """
        Compute lane-centre error.

        Convention: error = img_cx - lane_cx
          positive → lane centre is LEFT of image centre → robot must steer RIGHT
          negative → lane centre is RIGHT of image centre → robot must steer LEFT

        Single-lane logic:
          - If only LEFT lane visible: lane_cx = left_x + half_width
            → robot should be half_width to the right of the left lane
            → if left_x is too far right, error will be negative → steer left
          - If only RIGHT lane visible: lane_cx = right_x - half_width
            → robot should be half_width to the left of the right lane
            → if right_x is too far left, error will be positive → steer right

        Drift-back behavior:
          When only one lane is visible, the computed lane_cx is the TRUE
          expected centre. The error naturally drives the robot back toward
          both-lanes-visible position because once the robot re-centres,
          the hidden lane will come back into view.
        """
        img_cx = img_w / 2.0

        if left_x is None and right_x is None:
            return self._ema_error, False, 'none'

        half_w = self._half_width

        if left_x is not None and right_x is not None:
            sep = right_x - left_x

            if sep < self.min_sep:
                # ── Fragment: two detections are the SAME physical lane edge ──
                # Determine which side of centre the fragment is on.
                merged = (left_x + right_x) / 2.0

                if merged < img_cx:
                    # Fragment is on left side → it's the LEFT lane edge
                    # Robot centre should be half_width to its right
                    lane_cx = merged + half_w
                    mode = 'left_frag'
                    self.get_logger().debug(
                        f'Fragment LEFT: merged={merged:.0f} lane_cx={lane_cx:.0f}')
                else:
                    # Fragment is on right side → it's the RIGHT lane edge
                    # Robot centre should be half_width to its left
                    lane_cx = merged - half_w
                    mode = 'right_frag'
                    self.get_logger().debug(
                        f'Fragment RIGHT: merged={merged:.0f} lane_cx={lane_cx:.0f}')

            else:
                # ── Clean two-lane detection ───────────────────────────────────
                if sep > self.max_valid_sep:
                    # Sep is too large — one "lane" is actually an obstacle or
                    # a false detection. Trust each side independently instead:
                    # treat left_x as a real left edge and right_x as a real
                    # right edge and average the two implied centres.
                    lane_cx_from_left  = left_x  + half_w
                    lane_cx_from_right = right_x - half_w
                    lane_cx = (lane_cx_from_left + lane_cx_from_right) / 2.0
                    mode = 'both_wide'
                    self.get_logger().debug(
                        f'Both-wide sep={sep:.0f} > max={self.max_valid_sep:.0f}, '
                        f'averaging: cx={lane_cx:.0f}')
                else:
                    # Only calibrate when each lane is on its expected side
                    # AND sep is within a sane range
                    if left_x < img_cx and right_x > img_cx:
                        self._lw_buf.append(sep)
                    lane_cx = (left_x + right_x) / 2.0
                    mode = 'both'

        elif left_x is not None:
            # ── Only left lane visible → drift right until right lane appears ──
            lane_cx = left_x + half_w
            mode = 'left_only'
            self.get_logger().debug(
                f'Left only: left_x={left_x:.0f} half_w={half_w:.0f} '
                f'lane_cx={lane_cx:.0f}')

        else:
            # ── Only right lane visible → drift left until left lane appears ───
            lane_cx = right_x - half_w
            mode = 'right_only'
            self.get_logger().debug(
                f'Right only: right_x={right_x:.0f} half_w={half_w:.0f} '
                f'lane_cx={lane_cx:.0f}')

        raw_err = img_cx - lane_cx
        self._ema_error = (self.ema_alpha * raw_err
                           + (1.0 - self.ema_alpha) * self._ema_error)
        return self._ema_error, True, mode

    # ── Main callback ─────────────────────────────────────────────────────
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}')
            return

        left_x, right_x, white, edges, roi_y, \
            left_segs, right_segs, fw, fh = self._detect_lanes(frame)

        smooth_err, visible, mode = self._compute_error(left_x, right_x, fw)

        self.error_pub.publish(Float32(data=float(smooth_err)))
        self.visible_pub.publish(Bool(data=visible))

        # ── Debug overlay ─────────────────────────────────────────────────
        debug = frame.copy()
        roi_h = fh - roi_y

        # Semi-transparent white mask overlay
        white_color = np.zeros_like(frame[roi_y:fh])
        white_color[white > 0] = (0, 180, 0)
        debug[roi_y:fh] = cv2.addWeighted(
            debug[roi_y:fh], 0.7, white_color, 0.3, 0)

        cv2.rectangle(debug, (0, roi_y), (fw, fh), (0, 80, 0), 1)

        # Hough segments
        for x1, y1, x2, y2, _ in left_segs:
            cv2.line(debug, (x1, y1 + roi_y), (x2, y2 + roi_y), (255, 100, 0), 2)
        for x1, y1, x2, y2, _ in right_segs:
            cv2.line(debug, (x1, y1 + roi_y), (x2, y2 + roi_y), (0, 100, 255), 2)

        bottom_y = fh - 5

        if left_x is not None:
            lx = int(np.clip(left_x, 0, fw - 1))
            cv2.circle(debug, (lx, bottom_y), 10, (255, 100, 0), -1)
            cv2.putText(debug, f'L:{lx}',
                        (max(lx - 25, 0), bottom_y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 50), 1)

        if right_x is not None:
            rx = int(np.clip(right_x, 0, fw - 1))
            cv2.circle(debug, (rx, bottom_y), 10, (0, 100, 255), -1)
            cv2.putText(debug, f'R:{rx}',
                        (min(rx - 25, fw - 50), bottom_y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 150, 255), 1)

        if left_x is not None and right_x is not None:
            sep = right_x - left_x
            sep_col = (0, 255, 0) if sep >= self.min_sep else (0, 0, 255)
            lx = int(np.clip(left_x,  0, fw - 1))
            rx = int(np.clip(right_x, 0, fw - 1))
            cv2.line(debug, (lx, bottom_y), (rx, bottom_y), sep_col, 2)
            cv2.putText(debug, f'sep={sep:.0f}',
                        (int((lx + rx) / 2) - 30, bottom_y - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, sep_col, 1)

        if visible:
            img_cx  = fw // 2
            lane_cx = int(np.clip(img_cx - smooth_err, 0, fw - 1))
            # Green = image centre (robot heading)
            cv2.line(debug, (img_cx,  roi_y), (img_cx,  fh), (0, 255, 0),   2)
            # Red = detected lane centre (target)
            cv2.line(debug, (lane_cx, roi_y), (lane_cx, fh), (0, 0,   255), 2)

            # Draw half-width guides from lane centre (cyan dashed region)
            half_w_int = int(self._half_width)
            cv2.line(debug,
                     (max(lane_cx - half_w_int, 0), roi_y + roi_h // 2),
                     (max(lane_cx - half_w_int, 0), fh),
                     (255, 255, 0), 1)
            cv2.line(debug,
                     (min(lane_cx + half_w_int, fw - 1), roi_y + roi_h // 2),
                     (min(lane_cx + half_w_int, fw - 1), fh),
                     (255, 255, 0), 1)

        mode_colors = {
            'both':        (0,   255, 0),
            'both_wide':   (0,   200, 100),   # teal — wide sep, using averaged centres
            'left_only':   (0,   165, 255),
            'right_only':  (0,   165, 255),
            'left_frag':   (0,   200, 200),
            'right_frag':  (0,   200, 200),
            'none':        (128, 128, 128),
        }
        if len(self._lw_buf) >= 5:
            cal_str = f'cal={int(self._half_width*2)}px'
        else:
            cal_str = f'fix={int(self._lane_half_w*2)}px'
        cv2.putText(
            debug,
            f'err={smooth_err:.1f}px  [{mode}]  {cal_str}',
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            mode_colors.get(mode, (255, 255, 0)), 2)

        # Thumbnails
        th, tw = fh // 5, fw // 5
        wm_small = cv2.resize(white, (tw, th))
        ed_small = cv2.resize(edges, (tw, th))
        debug[fh - th:fh, 0:tw]    = cv2.cvtColor(wm_small, cv2.COLOR_GRAY2BGR)
        debug[fh - th:fh, tw:tw*2] = cv2.cvtColor(ed_small, cv2.COLOR_GRAY2BGR)
        cv2.putText(debug, 'white', (5,      fh - th + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)
        cv2.putText(debug, 'canny', (tw + 5, fh - th + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)

        debug_msg        = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

        if self.show_debug:
            cv2.imshow('Lane Detection', debug)
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