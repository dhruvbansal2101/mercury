from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory
import os
import xacro
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable

def generate_launch_description():

    pkg_desc = get_package_share_directory('description')
    xacro_file = os.path.join(pkg_desc, 'urdf', 'robot.urdf.xacro')
    
    pkg_sim = get_package_share_directory('simulation')

    world_file = os.path.join(
        pkg_sim,
        'worlds',
        'mercury.sdf'
    )
    # Build photo model blocks dynamically with correct absolute paths
    images_dir = os.path.join(pkg_sim, 'models', 'images')

    # Read the world SDF and inject the correct image paths
    import re
    with open(world_file, 'r') as f:
        world_content = f.read()

    for i in range(1, 7):
        world_content = world_content.replace(
            f'photo{i}.jpg',
            f'file://{images_dir}/photo{i}.jpg'
        )

    # Write to a temp file
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False)
    tmp.write(world_content)
    tmp.flush()
    world_file = tmp.name

    return LaunchDescription([

        SetEnvironmentVariable('GZ_IP', '127.0.0.1'),

        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(pkg_sim, 'models') + ':' + os.path.join(pkg_sim, 'models', 'images'),

        ),
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world_file],
            output='screen'
        ),

        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-topic', 'robot_description'
            ],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['joint_state_broadcaster'],
                    parameters=[{'use_sim_time': True}],
                    output='screen'
                ),
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=[
                        'diff_drive_controller',
                        '--controller-ros-args',
                        '--ros-args --remap /diff_drive_controller/cmd_vel:=/cmd_vel_stamped'
                    ],
                    parameters=[{'use_sim_time': True}],
                    output='screen'
                ),
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['turret_controller'],
                    parameters=[{'use_sim_time': True}],
                    output='screen'
                ),
            ]
        ),
        
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
                '/gps@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat',
                '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                '/turret_camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
                '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            ],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        Node(
            package='bringup',
            executable='twist_to_stamped',
            name='twist_to_stamped',
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),
    ])