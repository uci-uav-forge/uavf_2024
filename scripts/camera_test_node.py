#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from uavf_2024.imaging import Camera
from time import sleep, strftime

class CameraTestNode(Node):
    def __init__(self) -> None:
        super().__init__('imaging_node')
        self.camera = Camera(log_dir = f"/home/forge/ws/logs/{strftime('%m-%d %Hh%Mm')}/camera")
        self.camera.start_recording()
        self.camera.setAbsoluteZoom(1)
        self.camera.request_center()

    def loop(self):
        for _ in range(1000):
            sleep(0.1)
                

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