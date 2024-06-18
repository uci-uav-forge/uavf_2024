#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from uavf_2024.imaging import Perception, PoseProvider
from concurrent.futures import as_completed
from time import strftime
from pathlib import Path
import traceback

class PerceptionMinimalNode(Node):
    ''' Receives input from every ongoing process.
        Output to PX4 Interface Node.
    '''

    def __init__(self) -> None:
        super().__init__('perception_test_node')
        logs_dir = Path(f"/home/forge/ws/logs/{strftime('%m-%d %Hh%Mm')}")
        self.perception = Perception(PoseProvider(self, logs_dir / 'pose'), logs_path = logs_dir, logger = self.get_logger())
        self.perception.camera.start_recording()
        self.timer_period = 1.0  # seconds
        self.create_timer(self.timer_period, self.timer_cb)
        self.perception_futures = []
        self.time_alive = 0
    
    def log(self, msg):
        self.get_logger().info(msg)

    def timer_cb(self):
        self.log("Calling timer cb")
        self.perception_futures.append(self.perception.get_image_down_async())
        self.time_alive += 1
        if self.time_alive > 10:
            self.log('Shutting down...')
            # self.log(f"results: {self.collect_results()}")
            self.destroy_node()
            rclpy.shutdown()
    
    def collect_results(self):
        detections = []
        timeout = 2.0
        try:
            for future in as_completed(self.perception_futures, timeout=timeout):
                detections.extend(future.result())

            self.log(f"Successfully retrieved imaging detections: {detections}")
        except TimeoutError:
            self.log("Timed out waiting for imaging detections.")

        self.perception_futures = []

        return detections



def main(args=None) -> None:
    print('Starting commander node...')
    rclpy.init(args=args)
    node = PerceptionMinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()