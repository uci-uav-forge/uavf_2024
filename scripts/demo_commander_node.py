#!/usr/bin/env python3

from uavf_2024.gnc.commander_node import CommanderNode
import mavros_msgs.msg
import mavros_msgs.srv
import rclpy
import rclpy.node
import argparse
from threading import Thread
import sys

# Command to run: ros2 run libuavf_2024 demo_commander_node.py /home/ws/uavf_2024/uavf_2024/gnc/data/primary.gpx /home/ws/uavf_2024/uavf_2024/gnc/data/PAYLOAD_LIST 12 9

if __name__ == '__main__':
    rclpy.init()
    

    parser = argparse.ArgumentParser()
    parser.add_argument('gpx_file')
    parser.add_argument('payload_list')
    parser.add_argument('geofence_points')
    parser.add_argument('image_width_m', type = float)
    parser.add_argument('image_height_m', type = float)
    parser.add_argument('--exit-early', action='store_true')
    parser.add_argument('--servo-test', action='store_true')
    parser.add_argument('--call-imaging', action='store_true')
    parser.add_argument('--call-imaging-period', type = float, default = 5)
    parser.add_argument('--demo-setpoint-loop', action='store_true')
    args = parser.parse_args()

    node = CommanderNode(args)

    spinner = Thread(target = rclpy.spin, args = (node,))
    spinner.start()

    node.execute_mission_loop()


    node.destroy_node()
    rclpy.shutdown()