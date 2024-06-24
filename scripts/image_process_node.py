#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from uavf_2024.imaging import Perception, PoseProvider, ImageProcessor
from uavf_2024.imaging.imaging_types import Image, HWC
from concurrent.futures import as_completed
from time import strftime
from pathlib import Path
import traceback

from libuavf_2024.msg import Detection2D
from libuavf_2024.srv import ProcessImage

def log_exceptions(func):
    '''
    Decorator that can be applied to methods on any class that extends
    a ros `Node` to make them correctly log exceptions when run through
    a roslaunch file
    '''
    def wrapped_fn(self,*args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception:
            self.get_logger().error(traceback.format_exc())
    return wrapped_fn

class ImageProcessNode(Node):
    def __init__(self) -> None:
        super().__init__('image_process_node')
        logs_dir = Path(f"/home/forge/ws/logs/{strftime('%m-%d %Hh%Mm')}")
        self.image_processsor = ImageProcessor()
        self.process_service = self.create_service(ProcessImage, 'process_image_service', self.process_image)
        self.log("Finished processing node constructor")
    
    def log(self, msg):
        self.get_logger().info(msg)
    
    @log_exceptions
    def process_image(self, request, response):
        self.log("Got request!")
        cv_image = Image(request.image.reshape((1080, 1920, 3)), HWC)
        results = self.image_processsor.process_image(cv_image)
        ros_detections = [
            Detection2D(
                x=int(d.x),
                y=int(d.y),
                w=int(d.width),
                h=int(d.height),
                shape_conf=d.descriptor.shape_probs,
                letter_conf=d.descriptor.letter_probs,
                shape_color_conf=d.descriptor.shape_col_probs,
                letter_color_conf=d.descriptor.letter_col_probs,
                id=f'{d.img_id}/{d.det_id}'
            )
            for d in results
        ]
        response.detections = ros_detections

        return response


    
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
    print("Starting ros bullshit!")
    rclpy.init(args=args)
    node = ImageProcessNode()
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