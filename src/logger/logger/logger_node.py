import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Imu, NavSatFix, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from action_msgs.msg import GoalStatusArray
from rclpy.qos import qos_profile_sensor_data
import json
import os
from datetime import datetime
import psutil

class LoggerNode(Node):

    def __init__(self):
        super().__init__('logger_node')

        self.base_path = os.path.expanduser("~/robot_logs")
        os.makedirs(self.base_path, exist_ok=True)

        self.system_file = open(self._file("system_log.json"), "a")
        self.alert_file = open(self._file("alerts_log.json"), "a")
        self.state_file = open(self._file("state_log.json"), "a")
        self.control_file = open(self._file("control_log.json"), "a")
        self.nav_file = open(self._file("navigation_log.json"), "a")
        self.resource_file = open(self._file("system_resource_log.json"), "a")
        self.encoder_file = open(self._file("encoder_log.json"), "a")
        self.power_file = open(self._file("power_log.json"), "a")

        self.get_logger().info("Logger node started...")

        self.create_subscription(String, '/system_status', self.system_cb, 10)
        self.create_subscription(String, '/system_alerts', self.alert_cb, 10)

        self.create_subscription(Imu, '/imu', self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, '/gps', self.gps_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/diff_drive_controller/odom', self.odom_cb, qos_profile_sensor_data)

        self.create_subscription(JointState, '/joint_states', self.encoder_cb, qos_profile_sensor_data)

        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.create_subscription(Twist, '/cmd_vel_nav', self.cmd_nav_cb, 10)

        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.nav_status_cb, 10)

        self.create_timer(2.0, self.system_resource_cb)
        self.create_timer(5.0, self.power_cb)

    def _file(self, name):
        return os.path.join(self.base_path, name)

    def _time(self):
        return datetime.utcnow().isoformat()

    def _write(self, file, data, label):
        json_line = json.dumps(data)
        file.write(json_line + "\n")
        file.flush()
        self.get_logger().info(f"{label}: {json_line}")

    def system_cb(self, msg):
        self._write(self.system_file, {
            "time": self._time(),
            "system_status": msg.data
        }, "SYSTEM")

    def alert_cb(self, msg):
        self._write(self.alert_file, {
            "time": self._time(),
            "alert": msg.data
        }, "ALERT")

    def imu_cb(self, msg):
        self._write(self.state_file, {
            "time": self._time(),
            "imu": {
                "orientation": [
                    float(msg.orientation.x),
                    float(msg.orientation.y),
                    float(msg.orientation.z),
                    float(msg.orientation.w)
                ],
                "angular_velocity": [
                    float(msg.angular_velocity.x),
                    float(msg.angular_velocity.y),
                    float(msg.angular_velocity.z)
                ],
                "linear_acceleration": [
                    float(msg.linear_acceleration.x),
                    float(msg.linear_acceleration.y),
                    float(msg.linear_acceleration.z)
                ]
            }
        }, "IMU")

    def gps_cb(self, msg):
        self._write(self.state_file, {
            "time": self._time(),
            "gps": {
                "lat": float(msg.latitude),
                "lon": float(msg.longitude),
                "alt": float(msg.altitude)
            }
        }, "GPS")

    def odom_cb(self, msg):
        self._write(self.state_file, {
            "time": self._time(),
            "odom": {
                "pos": [
                    float(msg.pose.pose.position.x),
                    float(msg.pose.pose.position.y)
                ],
                "vel": [
                    float(msg.twist.twist.linear.x),
                    float(msg.twist.twist.angular.z)
                ]
            }
        }, "ODOM")

    def encoder_cb(self, msg):
        self._write(self.encoder_file, {
            "time": self._time(),
            "encoder": {
                "names": list(msg.name),
                "position": [float(x) for x in msg.position],
                "velocity": [float(x) for x in msg.velocity]
            }
        }, "ENCODER")

    def cmd_cb(self, msg):
        self._write(self.control_file, {
            "time": self._time(),
            "cmd_vel": {
                "linear": float(msg.linear.x),
                "angular": float(msg.angular.z)
            }
        }, "CMD")

    def cmd_nav_cb(self, msg):
        self._write(self.control_file, {
            "time": self._time(),
            "cmd_vel_nav": {
                "linear": float(msg.linear.x),
                "angular": float(msg.angular.z)
            }
        }, "CMD_NAV")

    def goal_cb(self, msg):
        self._write(self.nav_file, {
            "time": self._time(),
            "goal": {
                "x": float(msg.pose.position.x),
                "y": float(msg.pose.position.y)
            }
        }, "GOAL")

    def nav_status_cb(self, msg):
        statuses = []
        for s in msg.status_list:
            statuses.append({
                "status": int(s.status),
                "goal_id": [int(x) for x in s.goal_info.goal_id.uuid]
            })

        self._write(self.nav_file, {
            "time": self._time(),
            "nav_status": statuses
        }, "NAV")

    def system_resource_cb(self):
        self._write(self.resource_file, {
            "time": self._time(),
            "cpu": float(psutil.cpu_percent()),
            "memory": float(psutil.virtual_memory().percent)
        }, "SYS_RESOURCE")

    def power_cb(self):
        self._write(self.power_file, {
            "time": self._time(),
            "battery": "N/A",
            "voltage": "N/A"
        }, "POWER")


def main():
    rclpy.init()
    node = LoggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
