#!/usr/bin/env python3

from libuavf_2024.gnc.util import read_gps
from libuavf_2024.gnc.dropzone_planner import DropzonePlanner
from libuavf_2024.gnc.commander_node import CommanderNode
from threading import Thread
import rclpy
import argparse


# example: ros2 run uavf_2024 demo_dropzone_planner.py /home/ws/uavf_2024/uavf_2024/gnc/data/AIRDROP_BOUNDARY 0 0 0 40 30

if __name__ == '__main__':
    rclpy.init()
    node = CommanderNode()

    parser = argparse.ArgumentParser()
    parser.add_argument('dropzone_file')
    parser.add_argument('payload_color_id', type = int)
    parser.add_argument('payload_shape_id', type = int)
    parser.add_argument('payload_letter_id', type = int)
    parser.add_argument('image_width_m', type = float)
    parser.add_argument('image_height_m', type = float)

    args = parser.parse_args()


    spinner = Thread(target = rclpy.spin, args = (node,))
    spinner.start()

    planner = DropzonePlanner(node, read_gps(args.dropzone_file), args.image_width_m, args.image_height_m)

    planner.conduct_air_drop(args.payload_color_id, args.payload_shape_id, args.payload_letter_id)




    node.destroy_node()
    rclpy.shutdown()