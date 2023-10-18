#!/bin/bash
docker run -v $(pwd)/../uavf_2024:/root/ros2_ws/src/uavf_2024 -v $(pwd)/../uavf_ros2_msgs:/root/ros2_ws/src/uavf_ros2_msgs -it uavf_2024:latest /bin/bash
