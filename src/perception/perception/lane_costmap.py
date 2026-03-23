import cv2
import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from tf_transformations import euler_from_quaternion
import tf2_ros
from tf2_ros import TransformException

class RoadLineCostmapNode(Node):

    def __init__(self):
        super().__init__('road_line_costmap_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('costmap_topic', '/perception/road_costmap')
        self.declare_parameter('costmap_frame', 'odom')
        self.declare_parameter('bev_width', 800)
        self.declare_parameter('bev_height', 600)
        self.declare_parameter('decay_rate', 0.985)
        self.declare_parameter('resolution', 0.0051)
        self.declare_parameter('undistort', True)
        self.declare_parameter('show_debug', True)

        self._read_params()

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.homography = None

        self.map_grid = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_res = None
        self.map_width = None
        self.map_height = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(Image, self.image_topic, self.image_callback, sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)

        self.costmap_pub = self.create_publisher(OccupancyGrid, self.costmap_topic, map_qos)

    def _read_params(self):
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.costmap_topic = self.get_parameter('costmap_topic').value
        self.costmap_frame = self.get_parameter('costmap_frame').value
        self.decay_rate = self.get_parameter('decay_rate').value
        self.bev_w = self.get_parameter('bev_width').value
        self.bev_h = self.get_parameter('bev_height').value
        self.resolution = self.get_parameter('resolution').value
        self.undistort = self.get_parameter('undistort').value
        self.show_debug = self.get_parameter('show_debug').value

    def map_callback(self, msg):
        new_w = msg.info.width
        new_h = msg.info.height
        new_res = msg.info.resolution
        new_ox = msg.info.origin.position.x
        new_oy = msg.info.origin.position.y

        if self.map_grid is None or new_w != self.map_width or new_h != self.map_height:
            new_grid = np.zeros((new_h, new_w), dtype=np.uint8)

            if self.map_grid is not None:
                ox_off = int(round((self.map_origin_x - new_ox) / new_res))
                oy_off = int(round((self.map_origin_y - new_oy) / new_res))

                src_r0 = max(0, -oy_off)
                src_c0 = max(0, -ox_off)
                dst_r0 = max(0, oy_off)
                dst_c0 = max(0, ox_off)

                rows = min(self.map_height - src_r0, new_h - dst_r0)
                cols = min(self.map_width - src_c0, new_w - dst_c0)

                if rows > 0 and cols > 0:
                    new_grid[dst_r0:dst_r0+rows, dst_c0:dst_c0+cols] = \
                        self.map_grid[src_r0:src_r0+rows, src_c0:src_c0+cols]

            self.map_grid = new_grid
            self.map_width = new_w
            self.map_height = new_h

        self.map_res = new_res
        self.map_origin_x = new_ox
        self.map_origin_y = new_oy

    def camera_info_callback(self, msg):
        if self.camera_matrix is not None:
            return
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def compute_homography(self, frame):
        h, w = frame.shape[:2]

        src = np.float32([
            [w * 0.02, h * 0.65], #bl
            [w * 0.98, h * 0.65], #br
            [w * 0.25, h * 0.1], #tl
            [w * 0.75, h * 0.1] #tr
        ])

        dst = np.float32([
            [0, self.bev_h],
            [self.bev_w, self.bev_h],
            [0, 0],
            [self.bev_w, 0]
        ])

        self.homography = cv2.getPerspectiveTransform(src, dst)

        if self.show_debug:
            dbg = frame.copy()
            for p in src:
                cv2.circle(dbg, tuple(p.astype(int)), 6, (0, 0, 255), -1)
            cv2.imshow("src_points", dbg)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except:
            return

        if self.undistort and self.camera_matrix is not None:
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

        if self.homography is None:
            self.compute_homography(frame)
        
        if self.show_debug:
            cv2.imshow("1_raw", frame)

        bev = cv2.warpPerspective(frame, self.homography, (self.bev_w, self.bev_h))
        
        if self.show_debug:
            cv2.imshow("2_bev", bev)
        
        mask = self.detect_white_lines(bev)
        if self.show_debug:
            cv2.imshow("3_mask", mask)
        grid = self.mask_to_grid(mask, msg.header.stamp)

        if grid is not None:
            self.costmap_pub.publish(grid)

        if self.show_debug:
            cv2.waitKey(1)

    def detect_white_lines(self, bev):
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 80, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_and(white_mask, bright)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.dilate(mask, kernel2, iterations=1)

        return mask

    def mask_to_grid(self, mask, stamp):
        mask[-20:, :] = 0

        if self.map_grid is None:
            return None

        try:
            t = self.tf_buffer.lookup_transform(
                self.costmap_frame,
                "camera_link",
                rclpy.time.Time()
            )
        except TransformException:
            return None

        tx = t.transform.translation.x
        ty = t.transform.translation.y
        q = t.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        bev_res = self.resolution
        bev_h, bev_w = mask.shape

        current_grid = np.zeros((self.map_height, self.map_width), dtype=np.uint8)

        white_pixels = np.argwhere(mask > 0)
        if len(white_pixels) == 0:
            return self._build_grid_msg(stamp, current_grid)

        rows = white_pixels[:, 0].astype(np.float64)
        cols = white_pixels[:, 1].astype(np.float64)

        dx_robot = (bev_h - 1.0 - rows) * bev_res
        dy_robot = (bev_w / 2.0 - cols) * bev_res

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        map_x = tx + dx_robot * cos_y - dy_robot * sin_y
        map_y = ty + dx_robot * sin_y + dy_robot * cos_y

        gx = ((map_x - self.map_origin_x) / self.map_res).astype(int)
        gy = ((map_y - self.map_origin_y) / self.map_res).astype(int)

        valid = (gx >= 0) & (gx < self.map_width) & (gy >= 0) & (gy < self.map_height)

        temp = np.zeros_like(current_grid, dtype=np.uint8)
        temp[gy[valid], gx[valid]] = 255

        dist = cv2.distanceTransform(255 - temp, cv2.DIST_L2, 5)

        sigma = 5.0
        cost = np.exp(-(dist**2) / (2 * sigma**2)) * 100
        current_grid = cost.astype(np.uint8)

        return self._build_grid_msg(stamp, current_grid)

    def _build_grid_msg(self, stamp, grid_data):
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = stamp
        grid.header.frame_id = self.costmap_frame

        grid.info.resolution = self.map_res
        grid.info.width = self.map_width
        grid.info.height = self.map_height

        grid.info.origin.position.x = self.map_origin_x
        grid.info.origin.position.y = self.map_origin_y
        grid.info.origin.orientation.w = 1.0

        grid.data = grid_data.flatten().tolist()
        return grid

def main():
    rclpy.init()
    node = RoadLineCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()