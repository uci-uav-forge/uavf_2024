#!/usr/bin/env python3
from libuavf_2024.srv import TakePicture
from libuavf_2024.msg import TargetDetection
import rclpy
from rclpy.node import Node
from time import sleep
from uavf_2024.imaging import TargetTracker, Target3D, CertainTargetDescriptor

search_candidates = [
    CertainTargetDescriptor("red", "rectangle", "green", "C")
]

class ContinuousImagingClient(Node):
    def __init__(self):
        super().__init__('continuous_imaging_client')
        self.get_logger().info("Initializing Client")
        self.cli = self.create_client(TakePicture, 'imaging_service')

        while not self.cli.wait_for_service():
            self.get_logger().info('service not available, waiting again...')

        self.tracker = TargetTracker()

        self.req = TakePicture.Request()

        sleep(10)
        num_requests = 10
        for i in range(num_requests):
            self.get_logger().info(f"Sending request {i+1}/{num_requests}")
            res: list[TargetDetection] = self.send_request().detections
            self.tracker.update([
                Target3D.from_ros(detection) for detection in res
            ])

            sleep(1)

        self.get_logger().info("Done with requests")
        self.get_logger().info(str(self.tracker.estimate_positions(search_candidates)))

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
    node = ContinuousImagingClient()
    rclpy.spin(node)
    node.destroy_node()