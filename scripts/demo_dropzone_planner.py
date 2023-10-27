#!/usr/bin/env python3

from uavf_2024.gnc.util import read_gps
from uavf_2024.gnc.dropzone_planner import DropzonePlanner
from uavf_2024.gnc.commander_node import CommanderNode
from threading import Thread
import rclpy
import argparse


if __name__ == '__main__':
    rclpy.init()
    node = CommanderNode()

    parser = argparse.ArgumentParser()
    parser.add_argument('dropzone_file')
    parser.add_argument('payload_color_id', type = int)
    parser.add_argument('payload_shape_id', type = int)
    parser.add_argument('image_width_m', type = float)
    parser.add_argument('image_height_m', type = float)

    parser.parse_args()


    spinner = Thread(target = rclpy.spin, args = (node,))
    spinner.start()

    planner = DropzonePlanner(node, read_gps(parser.dropzone_file), parser.image_width_m, parser.image_height_m)

    planner.conduct_air_drop(parser.payload_color_id, parser.payload_shape_id)




    node.destroy_node()
    node.shutdown()