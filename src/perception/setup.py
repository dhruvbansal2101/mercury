from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
        glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # old costmap node (kept but not launched by default anymore)
            'calibrate_homography = perception.calibrate_homography:main',
            # NEW: lightweight detection node that feeds lane_assist_node
            'lane_detection = perception.lane_detection:main',
            'lane_costmap = perception.lane_costmap_node:main',
            'lane_assist_node = perception.lane_assist_node:main',
        ],
    },
)