from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'bringup'

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
        (os.path.join('share', package_name, 'config'),
        glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@todo.todo',
    description='Bringup package for mercury UGV',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'twist_to_stamped = bringup.twist_to_stamped:main',
            # NEW: lane-assist correction node
            'lane_assist_node = bringup.lane_assist_node:main',
            'goal_decomposer  = bringup.goal_decomposer:main',
        ],
    },
)