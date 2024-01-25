#!/usr/bin/env python3

from imaging_node import ImagingNode
import mavros_msgs.msg
import mavros_msgs.srv
import rclpy
import rclpy.node
import argparse
from threading import Thread
import sys

# Command to run: ros2 run uavf_2024 demo_imaging_node.py

if __name__ == '__main__':
    rclpy.init()

    node = ImagingNode()

    spinner = Thread(target = rclpy.spin, args = (node,))
    spinner.start()

    response = node.take_picture()

    print(response)
    node.destroy_node()
    rclpy.shutdown()