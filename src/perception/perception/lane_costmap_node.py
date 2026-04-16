"""
lane_costmap_node.py  (v1)
--------------------------
Projects detected lane boundaries from the camera image into the Nav2
global costmap frame, building a persistent OccupancyGrid that marks
areas OUTSIDE the lane as lethal obstacles (cost = 100).

Nav2's global_costmap subscribes to /perception/road_costmap via its
StaticLayer plugin.  As the robot drives forward, the lane costmap fills
in and constrains the planner to stay on the track — no sub-goals or
fighting with Nav2 required.

Pipeline
--------
  /camera/image_raw
        │
        ▼  (same white-mask → Hough pipeline as lane_detection.py)
  left_line, right_line  (pixels)
        │
        ▼  pinhole projection + camera_link → map TF
  world (x, y) boundary points on the ground plane (z = 0)
        │
        ▼  fill outside → 100, inside → 0, unseen → -1
  /perception/road_costmap  (OccupancyGrid, latched, map frame)
        │
        ▼
  Nav2 global_costmap  StaticLayer plugin

Costmap cell values
-------------------
  -1   unknown  (never seen — planner treats as traversable unknown)
   0   free     (confirmed inside lane)
  100  lethal   (confirmed outside lane / at lane boundary)

Camera model
------------
The camera_link URDF mount: xyz="0.15 0 2"  rpy="0 0.4 0"
  → camera is 2 m high, pitched ~23° downward.
Camera sensor: hfov = 1.047 rad, 640 × 480 px.

Image pixels are in the camera optical frame (x=right, y=down, z=forward).
TF supplies the camera_link→map transform; we apply an additional fixed
rotation from optical to camera_link to build the full ray in world space,
then intersect with the z = 0 ground plane.
"""

import math

import cv2
import numpy as np

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
)

import tf2_ros
from cv_bridge import CvBridge

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image


class LaneCostmapNode(Node):
    """
    Builds a persistent world-frame OccupancyGrid from camera lane detections.

    The grid covers the full track extent (default 70 × 70 m, matching the
    global_costmap configuration).  Cells start as -1 (unknown) and are
    updated to 0 (free) or 100 (lethal) as the robot drives and the camera
    sees each stretch of track.
    """

    # ------------------------------------------------------------------
    # Rotation from camera optical frame → camera_link frame.
    #
    # optical  : x = right,   y = down,   z = forward (into image)
    # link     : x = forward, y = left,   z = up       (ROS convention)
    #
    # Mapping:
    #   link_x =  opt_z   (forward  ← forward)
    #   link_y = -opt_x   (left     ← -right)
    #   link_z = -opt_y   (up       ← -down)
    # ------------------------------------------------------------------
    _R_OPT_TO_LINK = np.array(
        [[ 0,  0,  1],
         [-1,  0,  0],
         [ 0, -1,  0]],
        dtype=np.float64)

    def __init__(self):
        super().__init__('lane_costmap')

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter('map_width_m',            70.0)
        self.declare_parameter('map_height_m',           70.0)
        # Grid resolution in metres/cell.  0.10 m is 2× coarser than the
        # main costmap (0.05 m) which reduces CPU while still giving useful
        # granularity for lane boundary marking.
        self.declare_parameter('resolution',              0.10)
        # Bottom-left corner of the grid in the map frame.
        # Must be consistent with global_costmap origin.
        self.declare_parameter('map_origin_x',          -35.0)
        self.declare_parameter('map_origin_y',          -35.0)
        # How often to re-publish the costmap (Hz).
        self.declare_parameter('publish_rate',            5.0)
        # Camera sensor parameters — must match the URDF / Gazebo sensor.
        self.declare_parameter('camera_hfov',             1.047)   # rad
        self.declare_parameter('image_width',             640)
        self.declare_parameter('image_height',            480)
        # Only use the bottom portion of the image (same as lane_detection.py).
        self.declare_parameter('roi_top_frac',            0.35)
        # How many row samples to take across the ROI for projection.
        # More rows → denser costmap update per frame, higher CPU.
        self.declare_parameter('sample_rows',             8)
        # Pixels projected OUTSIDE each boundary → marked obstacle.
        self.declare_parameter('obstacle_pixels_outside', 48)
        # Pixels projected INSIDE each boundary → marked free.
        self.declare_parameter('free_pixels_inside',      32)
        # White-lane pixel thresholds (mirror lane_detection.py defaults).
        self.declare_parameter('white_v_min',             170)
        self.declare_parameter('white_s_max',              60)
        # Process only 1-in-N frames to limit CPU load.
        self.declare_parameter('process_every_n',          3)

        # ── Read parameters ─────────────────────────────────────────────────
        def _p(name):
            return self.get_parameter(name).value

        map_w_m          = float(_p('map_width_m'))
        map_h_m          = float(_p('map_height_m'))
        self._res        = float(_p('resolution'))
        self._origin_x   = float(_p('map_origin_x'))
        self._origin_y   = float(_p('map_origin_y'))
        rate             = float(_p('publish_rate'))
        self._hfov       = float(_p('camera_hfov'))
        self._img_w      = int(_p('image_width'))
        self._img_h      = int(_p('image_height'))
        self._roi_top    = float(_p('roi_top_frac'))
        self._sample_rows = int(_p('sample_rows'))
        self._obs_px     = int(_p('obstacle_pixels_outside'))
        self._free_px    = int(_p('free_pixels_inside'))
        self._white_vmin = int(_p('white_v_min'))
        self._white_smax = int(_p('white_s_max'))
        self._skip_n     = int(_p('process_every_n'))

        # ── Derived values ──────────────────────────────────────────────────
        self._grid_w  = int(round(map_w_m  / self._res))
        self._grid_h  = int(round(map_h_m  / self._res))
        total_cells   = self._grid_w * self._grid_h

        # Pinhole intrinsics derived from HFoV and image width.
        # Assuming square pixels: fy = fx.
        self._fx = (self._img_w / 2.0) / math.tan(self._hfov / 2.0)
        self._fy = self._fx
        self._cx = self._img_w  / 2.0
        self._cy = self._img_h  / 2.0

        # Persistent costmap: -1 = unknown, 0 = free, 100 = lethal
        self._grid = np.full(total_cells, -1, dtype=np.int8)

        self._frame_count = 0

        # ── TF ─────────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # ── ROS I/O ─────────────────────────────────────────────────────────
        # Latched so Nav2's StaticLayer receives the map on (re)connection
        # even if it starts after this node.
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self._bridge  = CvBridge()
        self._map_pub = self.create_publisher(
            OccupancyGrid, '/perception/road_costmap', latched_qos)

        self.create_subscription(
            Image, '/camera/image_raw', self._image_cb, sensor_qos)

        self.create_timer(1.0 / rate, self._publish_costmap)

        self.get_logger().info(
            f'LaneCostmapNode started: '
            f'grid={self._grid_w}×{self._grid_h} @ {self._res}m/cell  '
            f'fx={self._fx:.0f}px  publish={rate}Hz  '
            f'skip_n={self._skip_n}  cells={total_cells:,}')

    # ═══════════════════════════════════════════════════════════════════
    # Lane detection  (mirrors the core of lane_detection.py v5)
    # ═══════════════════════════════════════════════════════════════════

    def _detect_boundaries(self, frame):
        """
        Detect left / right lane boundary lines in the image.

        Returns a list of (left_x, right_x, y_pixel_in_full_frame) tuples
        sampled at evenly-spaced rows across the ROI.
        left_x / right_x are pixel x-coordinates (float) or None.
        """
        fh, fw = frame.shape[:2]
        roi_y  = int(fh * self._roi_top)
        roi    = frame[roi_y:fh, :]
        roi_h  = roi.shape[0]
        img_cx = fw / 2.0

        # White-pixel mask
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0,   0,               self._white_vmin]),
            np.array([180, self._white_smax, 255]))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, self._white_vmin, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, bright)
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # Hough lines
        edges = cv2.Canny(mask, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=15, minLineLength=20.0, maxLineGap=40.0)

        left_segs, right_segs = [], []
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                if x2 == x1:
                    continue
                slope = abs((y2 - y1) / float(x2 - x1))
                if not (0.2 <= slope <= 4.0):
                    continue
                length = float(np.hypot(x2 - x1, y2 - y1))
                if (x1 + x2) / 2.0 < img_cx:
                    left_segs.append((x1, y1, x2, y2, length))
                else:
                    right_segs.append((x1, y1, x2, y2, length))

        def _fit(segs):
            """Weighted least-squares line fit. Returns (m, b) or None."""
            ms, bs, ws = [], [], []
            for (x1, y1, x2, y2, w) in segs:
                dx = float(x2 - x1)
                if dx == 0:
                    continue
                m = (y2 - y1) / dx
                ms.append(m);  bs.append(y1 - m * x1);  ws.append(w)
            if not ms:
                return None
            tw   = sum(ws)
            m_av = sum(m * w for m, w in zip(ms, ws)) / tw
            b_av = sum(b * w for b, w in zip(bs, ws)) / tw
            return (m_av, b_av) if abs(m_av) > 1e-6 else None

        left_line  = _fit(left_segs)
        right_line = _fit(right_segs)
        if left_line is None and right_line is None:
            return []

        # Sample boundary x at evenly-spaced rows in the ROI.
        result = []
        step   = max(1, roi_h // max(1, self._sample_rows))

        for y_roi in range(roi_h - 1, 0, -step):
            y_full = y_roi + roi_y
            lx = rx = None

            if left_line:
                m, b = left_line
                lx = float(np.clip((y_roi - b) / m, 0.0, fw - 1.0))

            if right_line:
                m, b = right_line
                rx = float(np.clip((y_roi - b) / m, 0.0, fw - 1.0))

            result.append((lx, rx, y_full))

        return result

    # ═══════════════════════════════════════════════════════════════════
    # Ground-plane projection
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _quat_to_rot(q) -> np.ndarray:
        """Build a 3×3 rotation matrix from a ROS quaternion message."""
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)

    def _pixel_to_ground(self, u: float, v: float, tf_stamped) -> 'tuple|None':
        """
        Project image pixel (u, v) onto the ground plane (z = 0 in map frame).

        1. Form the normalised ray in the camera optical frame.
        2. Rotate to camera_link frame  (_R_OPT_TO_LINK).
        3. Rotate to map frame          (quaternion from TF).
        4. Intersect with z = 0.

        Returns (world_x, world_y) or None if the ray is parallel to the
        ground, points behind the camera, or the result would be unreasonably
        far (> 15 m from the camera — avoids extrapolation artefacts near
        the horizon).
        """
        # Step 1 — ray in optical frame
        ray_opt = np.array([
            (u - self._cx) / self._fx,
            (v - self._cy) / self._fy,
            1.0
        ], dtype=np.float64)

        # Step 2 — ray in camera_link frame
        ray_link = self._R_OPT_TO_LINK @ ray_opt

        # Step 3 — camera origin + rotation in map frame
        t       = tf_stamped.transform.translation
        cam_pos = np.array([t.x, t.y, t.z], dtype=np.float64)
        R_map   = self._quat_to_rot(tf_stamped.transform.rotation)
        ray_map = R_map @ ray_link

        # Step 4 — intersect with z = 0
        if abs(ray_map[2]) < 1e-6:
            return None                          # ray parallel to ground

        lam = -cam_pos[2] / ray_map[2]
        if lam <= 0:
            return None                          # intersection behind camera

        wx = cam_pos[0] + lam * ray_map[0]
        wy = cam_pos[1] + lam * ray_map[1]

        # Sanity check — reject horizon extrapolations
        dist_from_cam = math.hypot(wx - cam_pos[0], wy - cam_pos[1])
        if dist_from_cam > 15.0:
            return None

        return wx, wy

    # ═══════════════════════════════════════════════════════════════════
    # Costmap helpers
    # ═══════════════════════════════════════════════════════════════════

    def _world_to_cell(self, wx: float, wy: float) -> 'tuple|None':
        """Convert world (x, y) → grid (col, row).  Returns None if OOB."""
        col = int((wx - self._origin_x) / self._res)
        row = int((wy - self._origin_y) / self._res)
        if 0 <= col < self._grid_w and 0 <= row < self._grid_h:
            return col, row
        return None

    def _mark(self, col: int, row: int, value: int):
        """
        Write a value into the grid.

        Rule: obstacles (100) are never overwritten by free (0),
        but they can be overwritten by another obstacle.
        This prevents noisy free readings from clearing real boundaries.
        """
        idx = row * self._grid_w + col
        if value == 100 or self._grid[idx] != 100:
            self._grid[idx] = np.int8(value)

    def _update_from_boundaries(self, boundaries, tf_stamped):
        """
        Walk every sampled row, project boundary pixels to ground, and
        stamp the costmap cells on both sides of each boundary.
        """
        for (lx, rx, y_full) in boundaries:

            # ── Left boundary ──────────────────────────────────────────────
            if lx is not None:
                # Pixels outside (to the LEFT) → obstacle
                for du in range(0, self._obs_px, 8):
                    pt = self._pixel_to_ground(lx - du, y_full, tf_stamped)
                    if pt is None:
                        continue
                    cell = self._world_to_cell(pt[0], pt[1])
                    if cell:
                        self._mark(cell[0], cell[1], 100)

                # Pixels just inside (to the RIGHT) → free
                for du in range(6, self._free_px, 8):
                    pt = self._pixel_to_ground(lx + du, y_full, tf_stamped)
                    if pt is None:
                        continue
                    cell = self._world_to_cell(pt[0], pt[1])
                    if cell:
                        self._mark(cell[0], cell[1], 0)

            # ── Right boundary ─────────────────────────────────────────────
            if rx is not None:
                # Pixels outside (to the RIGHT) → obstacle
                for du in range(0, self._obs_px, 8):
                    pt = self._pixel_to_ground(rx + du, y_full, tf_stamped)
                    if pt is None:
                        continue
                    cell = self._world_to_cell(pt[0], pt[1])
                    if cell:
                        self._mark(cell[0], cell[1], 100)

                # Pixels just inside (to the LEFT) → free
                for du in range(6, self._free_px, 8):
                    pt = self._pixel_to_ground(rx - du, y_full, tf_stamped)
                    if pt is None:
                        continue
                    cell = self._world_to_cell(pt[0], pt[1])
                    if cell:
                        self._mark(cell[0], cell[1], 0)

    # ═══════════════════════════════════════════════════════════════════
    # ROS callbacks
    # ═══════════════════════════════════════════════════════════════════

    def _image_cb(self, msg: Image):
        # Skip every N frames to keep CPU low
        self._frame_count += 1
        if self._frame_count % self._skip_n != 0:
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}')
            return

        # Look up camera_link → map (use latest TF, not image stamp, for robustness)
        try:
            tf_stamped = self._tf_buf.lookup_transform(
                'map',
                'camera_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except tf2_ros.TransformException as ex:
            self.get_logger().debug(f'TF not ready: {ex}')
            return

        boundaries = self._detect_boundaries(frame)
        if not boundaries:
            return

        self._update_from_boundaries(boundaries, tf_stamped)

    def _publish_costmap(self):
        msg                       = OccupancyGrid()
        msg.header.stamp          = self.get_clock().now().to_msg()
        msg.header.frame_id       = 'map'
        msg.info.resolution       = self._res
        msg.info.width            = self._grid_w
        msg.info.height           = self._grid_h
        msg.info.origin.position.x = self._origin_x
        msg.info.origin.position.y = self._origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data                  = self._grid.tolist()
        self._map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()