# mercury
offical repo for ICMTC UGVC-2026

# How to run docker

```
sudo docker compose build 
sudo docker compose run ros
```

# How to run simulation

```
echo 'export GZ_SIM_RESOURCE_PATH=$(ros2 pkg prefix simulation)/share/simulation/models:$GZ_SIM_RESOURCE_PATH' >> ~/.bashrc
colcon build
ros2 launch bringup bringup_sim.launch.py
```

# Add this to your bashrc/zshrc
```
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/li
```
