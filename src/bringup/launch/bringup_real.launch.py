from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_desc = get_package_share_directory('description')
    xacro_file = os.path.join(pkg_desc, 'urdf', 'robot_real.urdf.xacro')
    
    hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('hardware'),
                'launch',
                'hardware.launch.py'
            ])
        )
    )

    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(             
            PathJoinSubstitution([
                FindPackageShare('bringup'),
                'launch',
                'bringup_base.launch.py'
            ])
        ),
        launch_arguments={
            'xacro_file': xacro_file
        }.items()
    )

    return LaunchDescription([
        hardware,
        base
    ])