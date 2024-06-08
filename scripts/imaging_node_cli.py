#!/usr/bin/env python3
from libuavf_2024.srv import PointCam,ZoomCam,ResetLogDir
import rclpy
from rclpy.node import Node

if __name__ == '__main__':
    rclpy.init()
    node = Node('imaging_client_cli')
    point_cam_client = node.create_client(PointCam, 'recenter_service')
    while not point_cam_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')
    node.get_logger().info("Finished intializing client")
    print('Welcome to the imaging client CLI')
    print('Enter a number key to set zoom level (1-9), 0 for 10')
    print('Enter "r" to reset log directory')
    print('Enter "d" to point camera down')
    print('Enter "u" to point camera up')
    print('Enter "q" to quit')
    zoom_client = node.create_client(ZoomCam, 'zoom_service')
    reset_client = node.create_client(ResetLogDir, 'reset_log_dir')
    while 1:
        user_input = input()
        if '0' <= user_input <= '9':
            req = ZoomCam.Request()
            if user_input == '0':
                req.zoom_level = 10
            else:
                req.zoom_level = int(user_input)

            node.get_logger().info('Setting zoom')
            future = zoom_client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
        elif user_input == 'r':
            req = ResetLogDir.Request()
            node.get_logger().info('Resetting log directory')
            future = reset_client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
        elif user_input == 'd':
            req = PointCam.Request()
            req.down = True
            node.get_logger().info('Pointing down')
            future = point_cam_client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
        elif user_input == 'u':
            req = PointCam.Request()
            req.down = False
            node.get_logger().info('Pointing up')
            future = point_cam_client.call_async(req)
            rclpy.spin_until_future_complete(node, future)
    node.destroy_node()