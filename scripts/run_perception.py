#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from uavf_2024.imaging import Perception, PoseProvider, ImageProcessor
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
        sub_cb_group = ReentrantCallbackGroup()
        self.perception = Perception(PoseProvider(self, logs_dir / 'pose', callback_group=sub_cb_group), logs_path = logs_dir, logger = self.get_logger())
        timer_cb_group = MutuallyExclusiveCallbackGroup()
        second_timer_cb_group = MutuallyExclusiveCallbackGroup()

        self.perception.camera.start_recording()
        self.timer_period = 1.0  # seconds
        self.create_timer(self.timer_period, self.timer_cb, timer_cb_group)
        self.create_timer(0.2, self.other_timer, second_timer_cb_group)
        self.perception_futures = []
        self.time_alive = 0
    
    def log(self, msg):
        self.get_logger().info(msg)

    def timer_cb(self):
        self.log("Running perception image down")
        self.perception_futures.append(self.perception.get_image_down_async())
        self.time_alive += 1
        if self.time_alive > 60:
            self.log('Collecting results')
            self.log(f"results: {self.collect_results()}")
            self.log('Shutting down...')
            self.perception.camera.disconnect()
            self.destroy_node()
            self.executor.shutdown()
            quit()

    def other_timer(self):
        self.get_logger().info("Should run at regular interval")
        self.perception.run_work()
    
    def collect_results(self):
        detections = []
        timeout = 2.0
        try:
            for future in as_completed(self.perception_futures, timeout=1):
                self.log("Got result!")
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
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()