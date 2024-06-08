#!/usr/bin/env python3
from libuavf_2024.srv import PointCam,ZoomCam,ResetLogDir
import rclpy
from rclpy.node import Node

if __name__ == '__main__':
    print('Welcome to the imaging client CLI')
    print('Enter a number key to set zoom level (1-9), 0 for 10')
    print('Enter "r" to reset log directory')
    print('Enter "d" to point camera down')
    print('Enter "u" to point camera up')
    print('Enter "q" to quit')
    rclpy.init()
    node = Node('imaging_client_cli')
    point_cam_client = node.create_client(PointCam, 'point_cam')
    zoom_client = node.create_client(ZoomCam, 'zoom_cam')
    while 1:
        user_input = input()
        if '0' <= user_input <= '9':
            req = ZoomCam.Request()
            if user_input == '0':
                req.zoom_level = 10
            else:
                req.zoom_level = int(user_input)
            zoom_client.call_async(req)
        elif user_input == 'r':
            req = ResetLogDir.Request()
            node.get_logger().info('Resetting log directory')
            node.create_client(ResetLogDir, 'reset_log_dir').call_async(req)
        elif user_input == 'd':
            req = PointCam.Request()
            req.down = True
            point_cam_client.call_async(req)
        elif user_input == 'u':
            req = PointCam.Request()
            req.down = False
            point_cam_client.call_async(req)
    node.destroy_node()