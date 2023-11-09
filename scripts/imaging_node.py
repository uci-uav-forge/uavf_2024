#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from uavf_2024.msg import TargetDetection
from uavf_2024.srv import TakePicture
import numpy as np

class ImagingNode(Node):
    def __init__(self) -> None:
        super().__init__('imaging_node')
        self.imaging_service = self.create_service(TakePicture, 'imaging_service', self.imaging_callback)
    
    def imaging_callback(self, request, response: list[TargetDetection]):
        response.detections = [
            TargetDetection(
                timestamp = 69420,
                x = 1.0,
                y = 2.0,
                z = 3.0,
                shape_conf = np.zeros(8).tolist(),
                letter_conf = np.zeros(36).tolist(),
                shape_color_conf = np.zeros(8).tolist(),
                letter_color_conf = np.zeros(8).tolist(),
            )
        ]
        return response

def main(args=None) -> None:
    print('Starting imaging node...')
    rclpy.init(args=args)
    node = ImagingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)