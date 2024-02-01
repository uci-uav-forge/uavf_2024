#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from uavf_2024.imaging import Camera
from time import sleep

class CameraTestNode(Node):
    def __init__(self) -> None:
        super().__init__('imaging_node')
        self.camera = Camera()
        self.camera.setAbsoluteZoom(1)
        self.camera.cam.requestAbsolutePosition(0, 0)
        sleep(2)

    def loop(self):
        while True:
            print(self.camera.cam.getAttitude())
            sleep(1/10)
            # if self.camera.cam.requestGimbalAttitude():
            #     attitude = self.camera.cam.getAttitude()
            #     self.get_logger().info(str(attitude))
            # else:
            #     self.get_logger().info(":(")
            # sleep(1)
                

def main(args=None) -> None:
    print('Starting imaging node...')
    rclpy.init(args=args)
    node = CameraTestNode()
    node.loop()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)