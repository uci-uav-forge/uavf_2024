#!/usr/bin/env python3

from uavf_2024.imaging import Camera, ImageProcessor, Localizer
import numpy as np
from time import strftime, time

# class ImagingNode:
#     def __init__(self) -> None:
#         super().__init__('imaging_node')
#         self.camera = Camera()
#         self.camera.setAbsoluteZoom(1)
#         self.image_processor = ImageProcessor(f'logs/{strftime("%m-%d %H:%M")}')
#         self.localizer = Localizer(30, (1920, 1080))
    
#     def get_attitudes(self, request, response: list[float]):
        

#         attitudes = self.camera.getAttitude()
#         self.get_logger().info(attitudes)

#         self.get_logger().info("prepping Request for attitudes")
#         response.attitudes = attitudes
#         return response
        

def main(args=None) -> None:
    print('Starting imaging node...')
    cam = Camera()
    print(cam.getAttitude())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)