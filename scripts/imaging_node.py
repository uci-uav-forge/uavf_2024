#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from uavf_2024.msg import TargetDetection
from uavf_2024.srv import TakePicture
from uavf_2024.imaging import Camera, ImageProcessor, Localizer
import numpy as np
from time import strftime, time

class ImagingNode(Node):
    def __init__(self) -> None:
        super().__init__('imaging_node')
        self.imaging_service = self.create_service(TakePicture, 'imaging_service', self.imaging_callback)
        self.camera = Camera()
        self.image_processor = ImageProcessor(f'logs/{strftime("%m-%d %H:%M")}')
        self.localizer = Localizer(30, (1920, 1080))

    def imaging_callback(self, request, response: list[TargetDetection]):
        timestamp = time()
        img = Camera.take_picture()

        detections = self.image_processor.process_image(img)

        cam_pose = np.array([0,0,0,0,0,0])
        preds_3d = [self.localizer.prediction_to_coords(d, cam_pose) for d in detections]

        response.detections = [
            TargetDetection(
                timestamp = timestamp,
                x = p.position[0],
                y = p.position[1],
                z = p.position[2],
                shape_conf = p.description.shape_probs,
                letter_conf = p.description.letter_probs,
                shape_color_conf = p.description.shape_col_probs,
                letter_color_conf = p.description.letter_col_probs
            ) for p in preds_3d
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