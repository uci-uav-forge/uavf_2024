#!/usr/bin/env python3

from pathlib import Path
import rclpy
from rclpy.node import Node
from libuavf_2024.msg import TargetDetection
from libuavf_2024.srv import TakePicture,PointCam,ZoomCam,GetAttitude
from uavf_2024.imaging import Camera, ImageProcessor, Localizer
from scipy.spatial.transform import Rotation as R
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from time import strftime, time, sleep
import cv2 as cv
import json
import os
import traceback

def log_exceptions(func):
    def wrapped_fn(self,*args, **kwargs):
        try:
            func(self, *args, **kwargs)
        except Exception:
            self.get_logger().error(traceback.format_exc())
    return wrapped_fn


class ImagingNode(Node):
    @log_exceptions
    def __init__(self) -> None:
        # Initialize the node
        super().__init__('imaging_node')
        logs_path = Path(f'logs/{strftime("%m-%d %H:%M")}')
        
        self.camera = Camera(logs_path / "camera")
        self.zoom_level = 3
        self.camera.setAbsoluteZoom(self.zoom_level)
        
        self.log(f"Logging to {logs_path}")
        self.image_processor = ImageProcessor(logs_path / "image_processor")

        1/0

        # Set up ROS connections
        self.log(f"Setting up imaging node ROS connections")
        # Init QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )

        # Subscribers ----
        # Set up mavros pose subscriber
        self.world_position_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_cb,
            qos_profile)

        # Services ----
        # Set up take picture service
        self.imaging_service = self.create_service(TakePicture, 'imaging_service', self.get_image_down)
        # Set up recenter camera service
        self.recenter_service = self.create_service(PointCam, 'recenter_service', self.request_point_cb)
        # Set up zoom camera service
        self.zoom_service = self.create_service(ZoomCam, 'zoom_service', self.setAbsoluteZoom_cb)
        # Set up reset log directory service
        self.reset_log_dir_service = self.create_service(ResetLogDir, 'reset_log_dir', self.reset_log_dir_cb)

        # Cleanup
        self.get_logger().info("Finished initializing imaging node")
        
    
    @log_exceptions
    def log(self, *args, **kwargs):
        self.get_logger().info(*args, **kwargs)

    @log_exceptions
    def pose_cb(self, pose: PoseStamped):
        # Update current position and rotation
        self.cur_position = pose.pose.position
        self.cur_rot = R.from_quat([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])
        self.last_pose_timestamp_secs = pose.header.stamp.sec + pose.header.stamp.nanosec / 1e9
        self.got_pose = True

    @log_exceptions
    def request_point_cb(self, request, response):
        self.log(f"Received Point Camera Down Request: {request}")
        if request.down:
            response.success = self.camera.request_down()
        else:
            response.success = self.camera.request_center()
        self.camera.request_autofocus()
        return response
    
    @log_exceptions
    def setAbsoluteZoom_cb(self, request, response):
        self.log(f"Received Set Zoom Request: {request}")
        response.success = self.camera.setAbsoluteZoom(request.zoom_level)
        self.camera.request_autofocus()
        self.zoom_level = request.zoom_level
        return response

    @log_exceptions
    def reset_log_dir_cb(self, request, response):
        new_logs_dir = Path('logs/{strftime("%m-%d %H:%M")}')
        self.log(f"Starting new log directory at {new_logs_dir}")
        os.makedirs(new_logs_dir, exist_ok = True)
        self.image_processor.reset_log_directory(new_logs_dir / 'image_processor')
        self.camera.set_log_dir(new_logs_dir / 'camera')
        response.success = True
        return response
    
    @log_exceptions
    def make_localizer(self):
        focal_len = self.camera.getFocalLength()
        localizer = Localizer.from_focal_length(
            focal_len, 
            (1920, 1080),
            (np.array([1,0,0]), np.array([0,-1, 0])),
            2    
        )
        return localizer

    @log_exceptions
    def point_camera_down(self):
        self.camera.request_down()
        while abs(self.camera.getAttitude()[1] - -90) > 2:
            self.log(f"Waiting to point down. Current angle: {self.camera.getAttitude()[1] } . " )
            sleep(0.1)
        self.log("Camera pointed down")
        self.camera.request_autofocus()

    @log_exceptions
    def get_image_down(self, request, response: list[TargetDetection]) -> list[TargetDetection]:
        '''
            autofocus, then wait till cam points down, take pic,
        
            We want to take photo when the attitude is down only. 
        '''
        self.log("Received Down Image Request")

        if abs(self.camera.getAttitude()[1] - -90) > 5: # Allow 5 degrees of error (Arbitrary)
            self.point_camera_down()

        #TODO: Figure out a way to detect when the gimbal is having an aneurism and figure out how to fix it or send msg to groundstation.
        
        # Take picture and grab relevant data
        localizer = self.make_localizer()
        start_angles = self.camera.getAttitude()
        img = self.camera.get_latest_image()
        timestamp = time()
        end_angles = self.camera.getAttitude()
        self.log("Picture taken")

        if img is None:
            self.log("Could not get image from Camera.")
            return []
    
        detections = self.image_processor.process_image(img)

        self.log("Images processed")

        # Get avg camera pose for the image
        avg_angles = np.mean([start_angles, end_angles],axis=0) # yaw, pitch, roll
        if not self.got_pose:
            for _ in range(5):
                if self.got_pose:
                    break
                self.log("Waiting for pose")
            if not self.got_pose:
                return
            else:
                self.log("Got pose!")

        cur_position_np = np.array([self.cur_position.x, self.cur_position.y, self.cur_position.z])
        self.log(f"Position: {self.cur_position.x:.02f},{self.cur_position.y:.02f},{self.cur_position.z:.02f}")
        cur_rot_quat = self.cur_rot.as_quat()


        world_orientation = self.camera.orientation_in_world_frame(self.cur_rot, avg_angles)
        cam_pose = (cur_position_np, world_orientation)

        self.log(f"{len(detections)} detections \t({'*'*len(detections)})")

        # Get 3D predictions
        preds_3d = [localizer.prediction_to_coords(d, cam_pose) for d in detections]

        # Log data
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
