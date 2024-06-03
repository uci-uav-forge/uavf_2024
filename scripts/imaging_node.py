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
import cv2 as cv
import json
import os

class ImagingNode(Node):
    def __init__(self) -> None:
        super().__init__('imaging_node')
        self.camera = Camera()
        self.camera.setAbsoluteZoom(3)
        logs_path = f'logs/{strftime("%m-%d %H:%M")}/image_processor'
        os.makedirs(logs_path, exist_ok=True)
        self.log(f"Logging to {logs_path}")
        self.image_processor = ImageProcessor(logs_path)

        focal_len = self.camera.getFocalLength()
        self.localizer = Localizer.from_focal_length(
            focal_len, 
            (1920, 1080),
            (np.array([1,0,0]), np.array([0,-1, 0])),
            2    
        )

        self.log(f"Setting up imaging node ROS connections")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )

        self.got_pose = False

        self.world_position_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.got_pose_cb,
            qos_profile)

        self.imaging_service = self.create_service(TakePicture, 'imaging_service', self.get_image_down)
        self.get_logger().info("Finished initializing imaging node")
        self.counter = 0
        self.zoom_level = 1
        self.zoom_levels=[1,2,3,4,5]
        self.zoom_index=0
        
        
    def got_pose_cb(self, pose: PoseStamped):
        self.cur_pose = pose
        self.cur_position = pose.pose.position
        self.cur_rot = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w,])
        self.last_pose_timestamp_secs = pose.header.stamp.sec + pose.header.stamp.nanosec/1e9
        self.got_pose = True

    def log(self, *args, **kwargs):
        self.get_logger().info(*args, **kwargs)
    
    def get_image_down(self, request, response: list[TargetDetection]) -> list[TargetDetection]:
        '''
            autofocus, then wait till cam points down, take pic,
        
            We want to take photo when the attitude is down only. 
        '''
        self.counter+=1
        if self.counter%30==0:
            self.zoom_index+=1
            self.zoom_level = self.zoom_levels[self.zoom_index%len(self.zoom_levels)]

        self.log("Received Down Image Request")

        self.camera.request_autofocus()
        self.camera.request_down()
        while abs(self.camera.getAttitude()[1] - -90) > 2:

            self.log(f"Waiting to point down. Current angle: {self.camera.getAttitude()[1] } . " )
            sleep(0.1)
        
        start_angles = self.camera.getAttitude()
        img = self.camera.take_picture()
        timestamp = time()
        end_angles = self.camera.getAttitude()
        self.log("Picture taken")

        detections = self.image_processor.process_image(img)

        self.log("Images processed")

        avg_angles = np.mean([start_angles, end_angles],axis=0) # yaw, pitch, roll
        if not self.got_pose:

            for _ in range(5):
                if self.got_pose:
                    break
                self.log("Waiting for pose")
            if not self.got_pose:
                return
            else:
                self.log("Got pose finally!")

        cur_position_np = np.array([self.cur_position.x, self.cur_position.y, self.cur_position.z])
        self.log(f"Position: {self.cur_position.x:.02f},{self.cur_position.y:.02f},{self.cur_position.z:.02f}")
        cur_rot_quat = self.cur_rot.as_quat()


        world_orientation = self.camera.orientation_in_world_frame(self.cur_rot, avg_angles)
        cam_pose = (cur_position_np, world_orientation)

        self.log(f"{len(detections)} detections \t({'*'*len(detections)})")
        preds_3d = [self.localizer.prediction_to_coords(d, cam_pose) for d in detections]

        logs_folder = self.image_processor.get_last_logs_path()
        self.log(f"This frame going to {logs_folder}")
        self.log(f"Zoom level: {self.zoom_level}")
        os.makedirs(logs_folder, exist_ok=True)
        cv.imwrite(f"{logs_folder}/image.png", img.get_array())
        log_data = {
            'pose_time': self.last_pose_timestamp_secs,
            'image_time': timestamp,
            'drone_position': cur_position_np.tolist(),
            'drone_q': cur_rot_quat.tolist(),
            'gimbal_yaw': avg_angles[0],
            'gimbal_pitch': avg_angles[1],
            'gimbal_roll': avg_angles[2],
            'zoom level': self.zoom_level,
            'preds_3d': [
                {
                    'position': p.position.tolist(),
                    'id': p.id,
                } for p in preds_3d
            ]
        }
        json.dump(log_data, open(f"{logs_folder}/data.json", 'w+'), indent=4)
        self.log("Localization finished")

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
                letter_color_conf = p.descriptor.letter_col_probs.tolist(),
                id = p.id
            )

            response.detections.append(t)

        self.log("Returning Response")

        return response

        

def main(args=None) -> None:
    print('Starting imaging node...')
    rclpy.init(args=args)
    node = ImagingNode()
    rclpy.spin(node)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
