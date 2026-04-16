# mercury  

Official repository for **ICMTC UGVC-2026**

---

## Prerequisites

- Ubuntu 22.04 / 24.04  
- ROS 2 Jazzy  
- colcon  
- rosdep  
- Docker (optional)

---

## First-Time Setup (Fresh Clone)

> This repository is already a ROS 2 workspace (contains `src/`)

```bash
# Clone workspace
git clone <repo-url>
cd mercury

# Source ROS
source /opt/ros/jazzy/setup.bash

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build workspace
colcon build

# Source workspace
source install/setup.bash
```

---

## Environment Setup

Add this to your `~/.bashrc` or `~/.zshrc`:

```bash
# ROS
source /opt/ros/jazzy/setup.bash

# Workspace
source ~/mercury/install/setup.bash

# Gazebo resource path
export GZ_SIM_RESOURCE_PATH=$(ros2 pkg prefix simulation)/share/simulation/models:$GZ_SIM_RESOURCE_PATH

# Gazebo system plugins
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib
```

Apply:

```bash
source ~/.bashrc
```

---

## Running with Docker

```bash
sudo docker compose build
sudo docker compose run ros
```

---

## Running Simulation

```bash
cd mercury
source install/setup.bash

ros2 launch bringup bringup_sim.launch.py
```

---

# watchdog_monitor

A non-intrusive ROS 2 monitoring and observability package for the Mercury robot.
Runs alongside the existing stack without modifying control logic.

---

## Nodes

| Node                     | Publishes                               | Rate         | Description                                                |
| ------------------------ | --------------------------------------- | ------------ | ---------------------------------------------------------- |
| `system_monitor_node`    | `/system_status`                        | 2s           | Tracks running vs expected nodes and publishes JSON health |
| `watchdog_node`          | `/system_alerts`                        | 3s           | Detects node crashes, topic silence, TF failures           |
| `waypoint_detector_node` | `/waypoint_reached`, `/waypoint_status` | 10Hz / 1Hz   | Detects arrival at predefined waypoints                    |
| `control_listener_node`  | —                                       | Event-driven | Passive observer logging monitoring events                 |
| `monitoring_dashboard`   | —                                       | 1Hz          | Live terminal dashboard                                    |

---

## Quick Start

```bash
colcon build --packages-select watchdog_monitor
source install/setup.bash

ros2 launch watchdog_monitor monitoring_all.launch.py
ros2 launch watchdog_monitor dashboard.launch.py
```

---

## Launch Arguments

| Argument           | Default                       | Description                          |
| ------------------ | ----------------------------- | ------------------------------------ |
| `monitor_interval` | `2.0`                         | Node health check interval (seconds) |
| `topic_timeout`    | `5.0`                         | Topic silence threshold (seconds)    |
| `arrival_radius`   | `0.5`                         | Waypoint detection radius (meters)   |
| `odom_topic`       | `/diff_drive_controller/odom` | Odometry source topic                |

---

## Turret Control

To manually move the turret, publish commands to the controller:

```bash
ros2 topic pub /turret_controller/commands std_msgs/msg/Float64MultiArray "{data: [1.0, 0.0]}"
```

- The array represents joint commands (e.g., yaw, pitch).
- Adjust values based on your turret configuration.

---

## Waypoint Configuration

Edit `config/waypoints.yaml`:

```yaml
waypoint_detector_node:
  ros__parameters:
    waypoints: [2.0, 0.0, 2.0, 4.0, 0.0, 4.0]
    waypoint_names: ["WP-1", "WP-2", "WP-3"]
    arrival_radius: 0.5
```

---

## Topics

| Topic               | Type                     | Publisher                |
| ------------------- | ------------------------ | ------------------------ |
| `/system_status`    | `std_msgs/String` (JSON) | `system_monitor_node`    |
| `/system_alerts`    | `std_msgs/String` (JSON) | `watchdog_node`          |
| `/waypoint_reached` | `std_msgs/String` (JSON) | `waypoint_detector_node` |
| `/waypoint_status`  | `std_msgs/String` (JSON) | `waypoint_detector_node` |

---

## Sending Navigation Goal

```bash
ros2 topic pub --once /goal_decomposer/goal geometry_msgs/msg/PoseStamped \ 
  "{header: {frame_id: 'map'}, pose: {position: {x: p, y: q}}}"
```

- update coordinate assignment by replacing (p, q) with the target UGV destination coordinates.

---

## Troubleshooting

### Package not found

```bash
source install/setup.bash
```

### Dependencies missing

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### Gazebo models not loading

```bash
echo $GZ_SIM_RESOURCE_PATH
```

---

## Clean Build

```bash
rm -rf build/ install/ log/
colcon build
```
