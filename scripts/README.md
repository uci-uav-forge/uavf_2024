# This directory is for running Python nodes.

## [List of PX4's ROS 2 topics](https://github.com/PX4/PX4-Autopilot/blob/main/src/modules/uxrce_dds_client/dds_topics.yaml)

## [List of message definitions](https://github.com/PX4/px4_msgs/tree/release/1.14/msg)

## Launching from launch files example

1. Go into `~/ws`
1. `source install/setup.bash`
1. `colcon build --merge-install`
1. `ros2 launch src/libuavf_2024/launches/imaging_demo.launch`