#!/usr/bin/env python3

from uavf_2024.gnc.util import read_gps
from uavf_2024.gnc.dropzone_planner import DropzonePlanner
from uavf_2024.gnc.commander_node import CommanderNode
from threading import Thread
import rclpy
import argparse


# example: ros2 run uavf_2024 demo_dropzone_planner.py /home/ws/uavf_2024/uavf_2024/gnc/data/TEST_MISSION /home/ws/uavf_2024/uavf_2024/gnc/data/AIRDROP_BOUNDARY 0 0 0 0 12 9

if __name__ == '__main__':
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('mission_file')
    parser.add_argument('dropzone_file')
    parser.add_argument('payload_shape_color_id', type = int)
    parser.add_argument('payload_shape_id', type = int)
    parser.add_argument('payload_letter_color_id', type = int)
    parser.add_argument('payload_letter_id', type = int)
    parser.add_argument('image_width_m', type = float)
    parser.add_argument('image_height_m', type = float)
    args = parser.parse_args()

    node = CommanderNode(args)


    spinner = Thread(target = rclpy.spin, args = (node,))
    spinner.start()

    node.dropzone_planner.conduct_air_drop()

    node.destroy_node()
    rclpy.shutdown()