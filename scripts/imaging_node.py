#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from libuavf_2024.msg import TargetDetection
from libuavf_2024.srv import TakePicture,GetAttitude
from uavf_2024.imaging import Camera, ImageProcessor, Localizer
from scipy.spatial.transform import Rotation as R
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from time import strftime, time, sleep

class ImagingNode(Node):
    def __init__(self) -> None:
        super().__init__('imaging_node')
        self.imaging_service = self.create_service(TakePicture, 'imaging_service', self.get_image_down)
        self.attitude_service = self.create_service(GetAttitude, 'attitude_service', self.get_attitudes)
        self.camera = Camera()
        self.camera.setAbsoluteZoom(1)
        self.image_processor = ImageProcessor(f'logs/{strftime("%m-%d %H:%M")}')
        focal_len = 1952.0 # TODO: un-hard-code this
        self.localizer = Localizer.from_focal_length(focal_len, (1920, 1080))

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )

        self.world_position_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.got_pose_cb,
            qos_profile)

        self.got_pose = False
        self.get_logger().info("Finished initializing imaging node")
        
    def got_pose_cb(self, pose: PoseStamped):
        self.cur_pose = pose
        self.cur_position = pose.pose.position
        self.cur_rot = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w,])
        self.got_pose = True

    def log(self, *args, **kwargs):
        self.get_logger().info(*args, **kwargs)
    
    def get_image_down(self, request, response: list[TargetDetection]):
        '''
            autofocus, then wait till cam points down, take pic,
        
            We want to take photo when the attitude is down only. 
        '''
        self.get_logger().info("Received Down Image Request")

        self.camera.request_autofocus()
        self.camera.request_down()
        while abs(self.camera.getAttitude()[1] - 90) > 2:

            self.get_logger().info(f"Waiting to point down {self.camera.getAttitude()[1] } . " )
            sleep(0.1)
        sleep(1) # To let the autofocus finish
        
        start_angles = self.camera.getAttitude()
        img = self.camera.take_picture()
        timestamp = time()
        end_angles = self.camera.getAttitude()
        self.get_logger().info("Picture taken")

        detections = self.image_processor.process_image(img)
        self.get_logger().info("Images processed")

        true_angles = np.mean([start_angles, end_angles],axis=0) # yaw, pitch, roll

        cam_rotation = R.from_euler('yxz', true_angles)

        cam_to_world_mat = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, 1, 0]
        ])

        cam_to_world = R.from_matrix(cam_to_world_mat) 

        cam_pose = (self.cur_position, self.cur_rot @ cam_to_world @ cam_rotation)
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

    
    def get_attitudes(self, request, response: list[float]):
        self.get_logger().info("Received Request for attitudes")
        self.camera.request_down()
        sleep(0.5)
        response.attitudes = self.camera.getAttitude()
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