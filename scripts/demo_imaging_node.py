#!/usr/bin/env python3
from libuavf_2024.srv import TakePicture
import rclpy
from rclpy.node import Node
from time import sleep

class DemoImagingClient(Node):

    def __init__(self):
        super().__init__('demo_imaging_client')
        self.get_logger().info("Initializing Client")
        self.cli = self.create_client(TakePicture, 'imaging_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_logger().info("Finished intializing client")
        self.req = TakePicture.Request()
        sleep(5)
        res = self.send_request()
        self.get_logger().info(str(res.detections))

    def send_request(self):
        self.get_logger().info("Sending request")
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        res = self.future.result()
        return res
    
# Command to run: ros2 run uavf_2024 demo_imaging_node.py

if __name__ == '__main__':
    print('Starting client node...')
    rclpy.init()
    node = DemoImagingClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()