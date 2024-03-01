#!/usr/bin/env python3
from libuavf_2024.srv import TakePicture
from libuavf_2024.msg import TargetDetection
import rclpy
from rclpy.node import Node
from time import sleep
from uavf_2024.imaging import TargetTracker, Target3D

class ContinuousImagingClient(Node):
    def __init__(self):
        super().__init__('continuous_imaging_client')
        self.get_logger().info("Initializing Client")
        self.cli = self.create_client(TakePicture, 'imaging_service')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_logger().info("Finished intializing client")

        self.tracker = TargetTracker()

        self.req = TakePicture.Request()

        sleep(5)
        self.get_logger().info("Sending request")
        while True:
            res: list[TargetDetection] = self.send_request()
            self.tracker.update([
                Target3D.from_ros(detection) for detection in res
            ])

            sleep(1)

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