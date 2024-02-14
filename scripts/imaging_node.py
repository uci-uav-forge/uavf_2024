#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from libuavf_2024.msg import TargetDetection
from libuavf_2024.srv import TakePicture
from uavf_2024.imaging import Camera, ImageProcessor, Localizer
import numpy as np
from time import strftime, time

class ImagingNode(Node):
    def __init__(self) -> None:
        super().__init__('imaging_node')
        self.imaging_service = self.create_service(TakePicture, 'imaging_service', self.imaging_callback)
        self.camera = Camera()
        self.camera.setAbsoluteZoom(1)
        self.image_processor = ImageProcessor(f'logs/{strftime("%m-%d %H:%M")}')
        self.localizer = Localizer(30, (1920, 1080))
        self.get_logger().info("Finished initializing imaging node")

    def imaging_callback(self, request, response: list[TargetDetection]):
        self.get_logger().info("Received Request")
        self.camera.request_center()
        self.camera.request_autofocus()
        img = self.camera.take_picture()
        timestamp = time()

        self.get_logger().info("Picture taken")

        detections = self.image_processor.process_image(img)

        self.get_logger().info("Images processed")

        cam_pose = np.array([0,0,0,0,0,0])
        preds_3d = [self.localizer.prediction_to_coords(d, cam_pose) for d in detections]

        self.get_logger().info("Localization finished")

        response.detections = []

        for i, p in enumerate(preds_3d):
            t = TargetDetection(
                timestamp = int(timestamp*1000),
                x = p.position[0],
                y = p.position[1],
                z = p.position[2],
                shape_conf = p.descriptor.shape_probs.tolist(),
                letter_conf = p.descriptor.letter_probs.tolist(),
                shape_color_conf = p.descriptor.shape_col_probs.tolist(),
                letter_color_conf = p.descriptor.letter_col_probs.tolist()
            )

            response.detections.append(t)

        self.get_logger().info("Returning Response")
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