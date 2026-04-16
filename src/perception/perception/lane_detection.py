"""
lane_detection.py  — v5 (clean math, robust calibration)
----------------------------------------------------------
═══════════════════════════════════════════════════════════
ERROR SIGN CONVENTION  (error = img_cx − lane_cx)
═══════════════════════════════════════════════════════════

  error > 0  →  lane centre is LEFT  of camera centre
                robot drifted RIGHT → must steer LEFT
                angular.z increases (positive = ROS left turn) ✓

  error < 0  →  lane centre is RIGHT of camera centre
                robot drifted LEFT  → must steer RIGHT
                angular.z decreases (negative = ROS right turn) ✓

  Proof: if the robot drifts RIGHT by D pixels, both lane edges
  shift LEFT in the camera image by D pixels.  lane_cx also shifts
  left, so  error = img_cx − lane_cx = +D  → steer left ✓

═══════════════════════════════════════════════════════════
LANE CENTRE CALCULATION PER MODE
═══════════════════════════════════════════════════════════

  mode         lane_cx                      notes
  ──────────── ──────────────────────────── ──────────────────────────
  both         (left_x + right_x) / 2       exact — no parameters!
  left_only    left_x  + half_w             centre = half_w right of L
  right_only   right_x − half_w             centre = half_w left of R
  left_frag    merged_x + half_w            same as left_only
  right_frag   merged_x − half_w            same as right_only
  none         hold previous EMA            no detection

  merged_x = (left_x + right_x) / 2  when sep < min_lane_sep_px
  (two groups detected but they are the SAME physical lane edge)

═══════════════════════════════════════════════════════════
HALF_W (half lane width in pixels)
═══════════════════════════════════════════════════════════

  Calibrated online from clean "both" detections where:
    • left_x  < img_cx − cal_min_offset_px   (clearly left of centre)
    • right_x > img_cx + cal_min_offset_px   (clearly right of centre)
    • cal_min_sep_px ≤ sep ≤ cal_max_sep_px  (plausible lane width)

  Uses median of a rolling window (maxlen=50) → robust to outliers.
  Falls back to lane_half_width_px parameter until min_cal_samples
  clean readings have been collected.

  HOW TO SET lane_half_width_px:
    1. Drive until the robot is visually centred on the track.
    2. Look at the debug overlay — the L: and R: dots show left/right
       edge pixel positions.
    3. Set lane_half_width_px = (R - L) / 2.
    4. Once "both" readings accumulate, calibration takes over.

═══════════════════════════════════════════════════════════
VISUAL OVERLAY
═══════════════════════════════════════════════════════════

  Green  vertical │ img_cx           image centre (robot heading)
  Red    vertical │ lane_cx          computed lane centre (target)
  Cyan   verticals│ lane_cx ± half_w expected lane edge positions
  Blue   segments │ left  Hough group
  Orange segments │ right Hough group
  dot  L:NNN      │ left_x  at bottom of ROI
  dot  R:NNN      │ right_x at bottom of ROI
  sep=NNN         │ right_x − left_x (green if ≥ min_lane_sep, red if fragment)
  hw=NNpx         │ current half_w estimate and calibration status

═══════════════════════════════════════════════════════════
BUG FIXES vs v4
═══════════════════════════════════════════════════════════

  1. x clipping: _weighted_line_x now clips the extrapolated
     x to [0, img_w−1].  Previously, steep lines could extrapolate
     to x < 0 or x >> img_w, corrupting lane_cx and the error.

  2. EMA reset on (re)acquisition: when the lane was invisible for
     one or more frames and then reappears, the EMA is hard-reset to
     the current raw_err instead of blending from a stale large value.
     Previously, err=247px would persist for many frames after the
     lane reappeared.

  3. Removed both_wide mode, use_auto_cal flag, drift_gain, and
     max_valid_sep_px — these added complexity without correctness.

  4. Removed auto-cal inhibit: calibration now always runs whenever
     both lanes meet the strict quality criteria. The old
     use_auto_cal=False default meant calibration never activated.

  5. ema_alpha raised from 0.30 → 0.40 for faster convergence
     (still smooth enough to reject single-frame noise).
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

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter('image_topic',         '/camera/image_raw')
        self.declare_parameter('show_debug',           True)

        # EMA smoothing factor for the published error.
        # 0.40 = faster response than v4's 0.30 while still damping noise.
        self.declare_parameter('ema_alpha',            0.40)

        # ── Half-width calibration ──────────────────────────────────────────
        # Startup fallback used until min_cal_samples clean readings arrive.
        # See "HOW TO SET" in the module docstring above.
        self.declare_parameter('lane_half_width_px',  160.0)

        # Require this many clean "both" readings before using auto-cal.
        self.declare_parameter('min_cal_samples',       8)

        # Each lane edge must be at least this far from img_cx (px) for a
        # "both" reading to count toward calibration.
        self.declare_parameter('cal_min_offset_px',   40.0)

        # Acceptable full-lane-width range for calibration (px).
        self.declare_parameter('cal_min_sep_px',       80.0)
        self.declare_parameter('cal_max_sep_px',      560.0)

        # ── ROI / image processing ─────────────────────────────────────────
        # Use only the bottom (1 − roi_top_frac) of the image.
        # Ignoring the top reduces perspective distortion on curves.
        self.declare_parameter('roi_top_frac',         0.35)

        # White-pixel HSV thresholds
        self.declare_parameter('white_v_min',          170)
        self.declare_parameter('white_s_max',           60)

        # Morphological closing kernel (stitches broken white blobs)
        self.declare_parameter('close_kw',               5)
        self.declare_parameter('close_kh',              25)

        # Canny edge thresholds
        self.declare_parameter('canny_low',             30.0)
        self.declare_parameter('canny_high',           100.0)

        # HoughLinesP parameters
        self.declare_parameter('hough_threshold',       15)
        self.declare_parameter('hough_min_len',         20.0)
        self.declare_parameter('hough_max_gap',         40.0)

        # Slope filter: reject nearly horizontal/vertical lines
        self.declare_parameter('min_slope_abs',          0.2)
        self.declare_parameter('max_slope_abs',          4.0)

        # If right_x − left_x < this → treat both as ONE physical edge
        self.declare_parameter('min_lane_sep_px',      120.0)

        # ── Read all parameters ─────────────────────────────────────────────
        def pv(name):
            return self.get_parameter(name).value

        self.show_debug       = pv('show_debug')
        self.ema_alpha        = float(pv('ema_alpha'))
        self._default_half_w  = float(pv('lane_half_width_px'))
        self._min_cal_samples = int(pv('min_cal_samples'))
        self._cal_min_off     = float(pv('cal_min_offset_px'))
        self._cal_min_sep     = float(pv('cal_min_sep_px'))
        self._cal_max_sep     = float(pv('cal_max_sep_px'))
        self.roi_top          = float(pv('roi_top_frac'))
        self.white_v_min      = int(pv('white_v_min'))
        self.white_s_max      = int(pv('white_s_max'))
        self.close_kw         = int(pv('close_kw'))
        self.close_kh         = int(pv('close_kh'))
        self.canny_low        = int(pv('canny_low'))
        self.canny_high       = int(pv('canny_high'))
        self.hough_thresh     = int(pv('hough_threshold'))
        self.hough_min        = float(pv('hough_min_len'))
        self.hough_gap        = float(pv('hough_max_gap'))
        self.min_slope        = float(pv('min_slope_abs'))
        self.max_slope        = float(pv('max_slope_abs'))
        self.min_sep          = float(pv('min_lane_sep_px'))
        image_topic           = str(pv('image_topic'))

        # ── Internal state ──────────────────────────────────────────────────
        self.bridge        = CvBridge()
        self._ema_error    = 0.0
        self._prev_visible = False   # used to detect re-acquisition
        # Rolling buffer of full lane widths (right_x − left_x) from quality
        # "both" detections.  half_w = median(buf) / 2.
        self._lw_buf: deque = deque(maxlen=50)

        # ── ROS I/O ─────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.image_sub   = self.create_subscription(
            Image, image_topic, self.image_callback, sensor_qos)
        self.error_pub   = self.create_publisher(Float32, '/lane_center_error', 10)
        self.visible_pub = self.create_publisher(Bool,    '/lane_visible',      10)
        self.debug_pub   = self.create_publisher(Image,   '/lane_debug/image',  sensor_qos)

        self.get_logger().info(
            f'LaneDetectionNode v5 | '
            f'default_half_w={self._default_half_w:.0f}px  '
            f'ema_alpha={self.ema_alpha}  '
            f'min_cal_samples={self._min_cal_samples}'
        )
        self.get_logger().info(
            'Calibration will activate once '
            f'{self._min_cal_samples} clean both-lane readings are collected. '
            'Drive on a straight section with both lane edges visible.'
        )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def _half_w(self) -> float:
        """Best current estimate of half lane width in pixels."""
        if len(self._lw_buf) >= self._min_cal_samples:
            return float(np.median(self._lw_buf)) / 2.0
        return self._default_half_w

    @property
    def _is_calibrated(self) -> bool:
        return len(self._lw_buf) >= self._min_cal_samples

    # ── Image processing ─────────────────────────────────────────────────────

    def _white_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Return binary mask of white/bright pixels in the ROI."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        # Low saturation + high value = white
        mask = cv2.inRange(
            hsv,
            np.array([0,   0,               self.white_v_min]),
            np.array([180, self.white_s_max, 255])
        )
        # AND with a simple brightness threshold for robustness
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, self.white_v_min, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, bright)
        # Morphological close: bridge small gaps in lane markings
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.close_kw, self.close_kh))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    def _weighted_line_x(self, segs: list, roi_h: int, img_w: int):
        """
        Fit a length-weighted line through the given segments and return its
        x-coordinate at the BOTTOM of the ROI (y = roi_h − 1), i.e., the
        x-position of that lane edge closest to the robot.

        FIX vs v4: result is CLIPPED to [0, img_w − 1].
        Previously, extrapolating a shallow-slope line to the bottom of the
        ROI could give x values far outside the image (e.g. x = 700 in a
        640-pixel image), corrupting all downstream calculations.

        Returns None if fewer than one usable segment.
        """
        if not segs:
            return None

        slopes, intercepts, weights = [], [], []
        for (x1, y1, x2, y2, length) in segs:
            dx = x2 - x1
            if dx == 0:
                continue                   # vertical line: skip
            m = (y2 - y1) / dx
            b = y1 - m * x1
            slopes.append(m)
            intercepts.append(b)
            weights.append(length)

        if not slopes:
            return None

        tw    = sum(weights)
        m_avg = sum(m * w for m, w in zip(slopes, weights)) / tw
        b_avg = sum(b * w for b, w in zip(intercepts, weights)) / tw

        if abs(m_avg) < 1e-6:
            return None                    # nearly horizontal: x undefined

        # x at bottom row of ROI (y = roi_h − 1)
        x_bottom = (roi_h - 1 - b_avg) / m_avg

        # ── KEY FIX: clip to valid image columns ────────────────────────────
        # If the line only exists in the upper ROI and its extrapolation to
        # the bottom row falls outside the image, we clamp rather than
        # producing an invalid position.  A clamped value (0 or img_w−1)
        # still conveys "lane edge is at the far left/right" which is the
        # correct semantics when the edge is near or beyond the image border.
        return float(np.clip(x_bottom, 0.0, img_w - 1.0))

    def _detect_lanes(self, frame: np.ndarray):
        """
        Run the full detection pipeline: mask → Canny → Hough → split L/R.

        Returns:
          left_x, right_x : extrapolated x at bottom of ROI (or None)
          white, edges    : debug images
          roi_y           : y-offset of ROI in the full frame
          left_segs, right_segs : raw Hough segments per side
          fw, fh          : full frame width/height
        """
        fh, fw = frame.shape[:2]
        roi_y  = int(fh * self.roi_top)
        roi    = frame[roi_y:fh, 0:fw]
        roi_h  = roi.shape[0]

        white = self._white_mask(roi)
        edges = cv2.Canny(white, self.canny_low, self.canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=1, theta=np.pi / 180,
            threshold=self.hough_thresh,
            minLineLength=self.hough_min,
            maxLineGap=self.hough_gap,
        )

        img_cx     = fw / 2.0
        left_segs  = []
        right_segs = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope   = (y2 - y1) / (x2 - x1)
                if not (self.min_slope <= abs(slope) <= self.max_slope):
                    continue
                seg_len = float(np.hypot(x2 - x1, y2 - y1))
                mid_x   = (x1 + x2) / 2.0
                if mid_x < img_cx:
                    left_segs.append((x1, y1, x2, y2, seg_len))
                else:
                    right_segs.append((x1, y1, x2, y2, seg_len))

        left_x  = self._weighted_line_x(left_segs,  roi_h, fw)
        right_x = self._weighted_line_x(right_segs, roi_h, fw)

        return left_x, right_x, white, edges, roi_y, left_segs, right_segs, fw, fh

    # ── Core error calculation ────────────────────────────────────────────────

    def _compute_error(self, left_x, right_x, img_w: int):
        """
        Compute lane-centre error and update the EMA.

        Returns (smoothed_error, lane_visible, mode_string).

        ── Mode decision tree ──────────────────────────────────────────────

        left_x != None AND right_x != None:
          sep = right_x − left_x

          sep < min_lane_sep_px  →  FRAGMENT
            Both Hough groups detected the SAME physical edge.
            merged = (left_x + right_x) / 2
            if merged < img_cx  →  left_frag  : lane_cx = merged + half_w
            else                →  right_frag : lane_cx = merged − half_w

          sep >= min_lane_sep_px  →  BOTH
            lane_cx = (left_x + right_x) / 2   ← exact, no parameters!
            Calibrate half_w if edges are on expected sides.

        left_x != None, right_x == None  →  LEFT ONLY
          lane_cx = left_x + half_w

        left_x == None, right_x != None  →  RIGHT ONLY
          lane_cx = right_x − half_w

        both None  →  NONE  (hold EMA)

        ── EMA reset ───────────────────────────────────────────────────────
        If the lane was invisible last frame and is now visible, the EMA is
        hard-reset to the current raw_err rather than blending from a
        potentially large stale value.  This fixes the symptom where
        err=247px persisted long after the lane reappeared.
        """
        img_cx  = img_w / 2.0
        visible = (left_x is not None) or (right_x is not None)

        if not visible:
            self._prev_visible = False
            return self._ema_error, False, 'none'

        half_w = self._half_w

        # ── Determine lane centre ──────────────────────────────────────────
        if left_x is not None and right_x is not None:
            sep = right_x - left_x      # >= 0 guaranteed by x clipping

            if sep < self.min_sep:
                # ── FRAGMENT ──────────────────────────────────────────────
                # Two separate Hough groups but they represent the same
                # physical lane line (e.g. a thick marking split by img_cx).
                # Merge into a single x and treat as single-lane.
                merged = (left_x + right_x) / 2.0

                if merged < img_cx:
                    # Fragment sits LEFT of centre → it is the LEFT edge
                    # Robot centre should be half_w to the right of this edge
                    lane_cx = merged + half_w
                    mode    = 'left_frag'
                else:
                    # Fragment sits RIGHT of centre → it is the RIGHT edge
                    # Robot centre should be half_w to the left of this edge
                    lane_cx = merged - half_w
                    mode    = 'right_frag'

                self.get_logger().debug(
                    f'[{mode}] sep={sep:.0f} < {self.min_sep:.0f}  '
                    f'merged={merged:.0f}  lane_cx={lane_cx:.0f}'
                )

            else:
                # ── BOTH LANES CLEANLY DETECTED ───────────────────────────
                # This is the only mode that needs ZERO assumptions about
                # lane width — the midpoint IS the lane centre.
                lane_cx = (left_x + right_x) / 2.0
                mode    = 'both'

                # Calibrate half_w only when each edge is clearly on its
                # expected side of the image and separation is plausible.
                if (left_x  < img_cx - self._cal_min_off
                        and right_x > img_cx + self._cal_min_off
                        and self._cal_min_sep <= sep <= self._cal_max_sep):
                    self._lw_buf.append(sep)
                    self.get_logger().debug(
                        f'Cal: sep={sep:.0f}px  '
                        f'n={len(self._lw_buf)}  '
                        f'→ half_w={sep / 2:.0f}px'
                    )

        elif left_x is not None:
            # ── LEFT LANE ONLY ────────────────────────────────────────────
            # Robot centre should be half_w to the RIGHT of the left edge.
            #
            # Why: if robot is centred, left_x = img_cx − half_w.
            # So lane_cx = left_x + half_w = img_cx when centred.
            # If robot drifts right, left_x decreases, lane_cx decreases,
            # error = img_cx − lane_cx increases → steer left ✓
            lane_cx = left_x + half_w
            mode    = 'left_only'

        else:
            # ── RIGHT LANE ONLY ───────────────────────────────────────────
            # Robot centre should be half_w to the LEFT of the right edge.
            #
            # If robot drifts left, right_x increases, lane_cx increases,
            # error = img_cx − lane_cx decreases (more negative) → steer right ✓
            lane_cx = right_x - half_w
            mode    = 'right_only'

        # ── Error and EMA ──────────────────────────────────────────────────
        raw_err = img_cx - lane_cx

        if not self._prev_visible:
            # ── KEY FIX: hard-reset EMA on re-acquisition ─────────────────
            # If we just went from no-lane → lane, skip blending so we
            # don't carry the stale large error from the invisible period.
            self._ema_error = raw_err
        else:
            self._ema_error = (self.ema_alpha * raw_err
                               + (1.0 - self.ema_alpha) * self._ema_error)

        self._prev_visible = True

        self.get_logger().debug(
            f'[{mode}]  L={left_x}  R={right_x}  '
            f'lane_cx={lane_cx:.1f}  raw={raw_err:.1f}  '
            f'ema={self._ema_error:.1f}  half_w={half_w:.0f}'
        )

        return self._ema_error, True, mode

    # ── Main callback ─────────────────────────────────────────────────────────

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}')
            return

        (left_x, right_x, white, edges,
         roi_y, left_segs, right_segs, fw, fh) = self._detect_lanes(frame)

        smooth_err, visible, mode = self._compute_error(left_x, right_x, fw)

        self.error_pub.publish(Float32(data=float(smooth_err)))
        self.visible_pub.publish(Bool(data=visible))

        # ── Debug overlay ─────────────────────────────────────────────────
        debug  = frame.copy()
        roi_h  = fh - roi_y
        img_cx = fw // 2

        # Green tint on detected white pixels in ROI
        tint = np.zeros_like(frame[roi_y:fh])
        tint[white > 0] = (0, 180, 0)
        debug[roi_y:fh] = cv2.addWeighted(debug[roi_y:fh], 0.7, tint, 0.3, 0)
        cv2.rectangle(debug, (0, roi_y), (fw, fh), (0, 80, 0), 1)

        # Hough segments — blue = left group, orange = right group
        for (x1, y1, x2, y2, _) in left_segs:
            cv2.line(debug,
                     (x1, y1 + roi_y), (x2, y2 + roi_y), (255, 100, 0), 2)
        for (x1, y1, x2, y2, _) in right_segs:
            cv2.line(debug,
                     (x1, y1 + roi_y), (x2, y2 + roi_y), (0, 100, 255), 2)

        # Detected edge dots at the bottom of the ROI
        bot_y = fh - 5
        if left_x is not None:
            lxi = int(left_x)
            cv2.circle(debug, (lxi, bot_y), 10, (255, 100, 0), -1)
            cv2.putText(debug, f'L:{lxi}',
                        (max(lxi - 25, 0), bot_y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 50), 1)

        if right_x is not None:
            rxi = int(right_x)
            cv2.circle(debug, (rxi, bot_y), 10, (0, 100, 255), -1)
            cv2.putText(debug, f'R:{rxi}',
                        (min(rxi - 25, fw - 55), bot_y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 150, 255), 1)

        # Separation bar between L and R dots
        if left_x is not None and right_x is not None:
            sep     = right_x - left_x
            seg_col = (0, 255, 0) if sep >= self.min_sep else (0, 0, 255)
            cv2.line(debug,
                     (int(left_x), bot_y), (int(right_x), bot_y),
                     seg_col, 2)
            mid_x_px = int((left_x + right_x) / 2)
            cv2.putText(debug, f'sep={sep:.0f}',
                        (mid_x_px - 30, bot_y - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, seg_col, 1)

        # Vertical guide lines
        if visible:
            lane_cx_px = int(np.clip(img_cx - smooth_err, 0, fw - 1))
            hw_int     = int(self._half_w)

            # Green = image centre (where robot is heading)
            cv2.line(debug, (img_cx, roi_y), (img_cx, fh), (0, 255, 0), 2)

            # Red = computed lane centre (where robot should be heading)
            cv2.line(debug,
                     (lane_cx_px, roi_y), (lane_cx_px, fh),
                     (0, 0, 255), 2)

            # Cyan = expected lane edge positions (lane_cx ± half_w)
            for edge_x in (lane_cx_px - hw_int, lane_cx_px + hw_int):
                edge_x = int(np.clip(edge_x, 0, fw - 1))
                cv2.line(debug,
                         (edge_x, roi_y + roi_h // 2), (edge_x, fh),
                         (255, 255, 0), 1)

        # ── HUD ───────────────────────────────────────────────────────────
        if self._is_calibrated:
            hw_str = (f'hw={self._half_w:.0f}px'
                      f'(cal,n={len(self._lw_buf)})')
        else:
            hw_str = (f'hw={self._half_w:.0f}px'
                      f'(default,n={len(self._lw_buf)}/{self._min_cal_samples})')

        mode_col = {
            'both':       (0, 255, 0),
            'left_only':  (0, 165, 255),
            'right_only': (0, 165, 255),
            'left_frag':  (0, 220, 220),
            'right_frag': (0, 220, 220),
            'none':       (120, 120, 120),
        }.get(mode, (255, 255, 0))

        cv2.putText(
            debug,
            f'err={smooth_err:.1f}px  [{mode}]  {hw_str}',
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_col, 2)

        # Small thumbnails of white mask and Canny edges (bottom-left)
        th = fh // 5
        tw = fw // 5
        debug[fh - th:fh, 0:tw] = cv2.cvtColor(
            cv2.resize(white, (tw, th)), cv2.COLOR_GRAY2BGR)
        debug[fh - th:fh, tw:tw * 2] = cv2.cvtColor(
            cv2.resize(edges, (tw, th)), cv2.COLOR_GRAY2BGR)
        cv2.putText(debug, 'white', (5,      fh - th + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)
        cv2.putText(debug, 'canny', (tw + 5, fh - th + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)

        debug_msg        = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

        if self.show_debug:
            cv2.imshow('Lane Detection v26', debug)
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