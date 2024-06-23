#!/usr/bin/env python3

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, NamedTuple
from bisect import bisect_left

import cv2 as cv
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation, Slerp

from libuavf_2024.msg import TargetDetection
from libuavf_2024.srv import PointCam, ResetLogDir, TakePicture, ZoomCam, GetPose, GetFirstPose
from uavf_2024.logging_utils.async_utils import AsyncBuffer, OnceCallable, RosLoggingProvider, Subscriptions, PoseDatum
from uavf_2024.imaging import Camera, ImageProcessor, Localizer


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

class ImagingNode(Node):
    LOGS_BASE_DIR = Path('/home/forge/ws/logs/') # '/media/forge/SANDISK/logs/'
    
    @log_exceptions
    def __init__(self) -> None:
        # Initialize the node
        super().__init__('imaging_node') # type: ignore
        cb_group = ReentrantCallbackGroup()
        self.logs_path = Path(__class__.LOGS_BASE_DIR / f'{time.strftime("%m-%d %Hh%Mm")}')

        self.camera = Camera(self.logs_path / "camera")
        self.zoom_level = 3
        self.camera.setAbsoluteZoom(self.zoom_level)
        
        self.log(f"Logging to {self.logs_path}")
        self.image_processor = ImageProcessor(self.logs_path / "image_processor")

        # Set up ROS connections
        self.log(f"Setting up imaging node ROS connections")
        
        # Services ----
        # Set up take picture service
        self.imaging_service = self.create_service(TakePicture, 'imaging_service', self.get_image_down, callback_group=cb_group)

        # Set up recenter camera service
        self.recenter_service = self.create_service(PointCam, 'recenter_service', self.request_point_cb, callback_group=cb_group)
        # Set up zoom camera service
        self.zoom_service = self.create_service(ZoomCam, 'zoom_service', self.setAbsoluteZoom_cb, callback_group=cb_group)
        # Set up reset log directory service
        self.reset_log_dir_service = self.create_service(ResetLogDir, 'reset_log_dir', self.reset_log_dir_cb, callback_group=cb_group)

        self.get_pose_client = self.create_client(GetPose, 'get_pose_service', callback_group=cb_group)
        self.get_first_pose_client = self.create_client(GetFirstPose, 'get_first_pose_service', callback_group=cb_group)

        # while not self.get_first_pose_client.wait_for_service(5):
        #     self.log('First pose client not available. waiting 5 sec and retrying')

        # Cleanup
        self.camera.start_recording()
        self.first_pose = None

        self.log("Finished initializing imaging node")
        
    @log_exceptions
    def log(self, *args, **kwargs):
        self.get_logger().info(*args, **kwargs)

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
        
        if False and self.first_pose is None:
            req = GetFirstPose.Request()
            self.log("Making first pose request")
            future = self.get_first_pose_client.call_async(req)
            self.log("Got future")
            # while not future.done():
            #     self.log("Waiting on future")
            #     pass
            self.executor.spin_until_future_complete(future, timeout_sec = 1.0)
            self.log("finished spinning")
            res = future.result()
            # res = self.get_first_pose_client.call(req)
            self.log("Got res")
            self.first_pose = PoseDatum.from_ros(res.pose) if res is not None else None
            self.log(f"Finished getting first pose: {self.first_pose}") # should run immediately after printing the previous thing
        
        localizer = Localizer.from_focal_length(
            focal_len, 
            (1920, 1080),
            (np.array([1,0,0]), np.array([0,-1, 0])),
            2,
            self.first_pose.position.z -0.15 if self.first_pose is not None else 0 # cube is about 15cm off ground
        )
        return localizer

    @log_exceptions
    def point_camera_down(self):
        self.camera.request_down()
        # while abs(self.camera.getAttitude()[1] - -90) > 2:
        #     self.log(f"Waiting to point down. Current angle: {self.camera.getAttitude()[1] } . " )
        #     time.sleep(0.1)
        self.log("Camera pointed down")

    @log_exceptions
    def get_image_down(self, request, response: list[TargetDetection]) -> list[TargetDetection]:
        '''
            autofocus, then wait till cam points down, take pic,
        
            We want to take photo when the attitude is down only. 
        '''

        self.log("Received Down Image Request")
        # time.sleep(0.5)
        # req = GetPose.Request()
        # timestamp = time.time()
        # req.timestamp_seconds = timestamp
        # pose = PoseDatum.from_ros(self.get_pose_client.call(req).pose)
        # self.log(f"Got pose: {pose}")
        # response.detections = []
        # return response
        self.camera.request_autofocus()
        self.log("Autofocused")

        self.log("Finished waiting for data")

        if abs(self.camera.getAttitude()[1] - -90) > 5: # Allow 5 degrees of error (Arbitrary)
            self.point_camera_down()

        #TODO: Figure out a way to detect when the gimbal is having an aneurism and figure out how to fix it or send msg to groundstation.
        
        # Take picture and grab relevant data
        img = self.camera.get_latest_image()
        if img is None:
            self.log("Could not get image from Camera.")
            response.detections = []
            return response

        timestamp = time.time()
        self.log(f"Got image from Camera at time {timestamp}")
        
        localizer = self.make_localizer()
        if localizer is None:
            self.log("Could not get Localizer")
            response.detections = []
            return response
    
        detections = self.image_processor.process_image(img)
        self.log(f"Finished image processing. Got {len(detections)} detections")

        # Get avg camera pose for the image
        angles = self.camera.getAttitudeInterpolated(timestamp)
        self.log(f"Got camera attitude: {angles}")
        
        # Get the pose measured 0.75 seconds before we received the image
        # This is to account for the delay in the camera system, and was determined empirically
        # req = GetPose.Request()
        # req.timestamp_seconds = timestamp
        # pose = PoseDatum.from_ros(self.get_pose_client.call(req).pose)
        pose = PoseDatum(
            Point(x=0.0,y=0.0,z=0.0),
            Rotation.identity(),
            0
        )
        self.log(f"Got pose: {pose}")

        cur_position_np = np.array([pose.position.x, pose.position.y, pose.position.z])
        cur_rot_quat = pose.rotation.as_quat()

        world_orientation = self.camera.orientation_in_world_frame(pose.rotation, angles)
        cam_pose = (cur_position_np, world_orientation)

        # Get 3D predictions
        preds_3d = [localizer.prediction_to_coords(d, cam_pose) for d in detections]

        # Log data
        logs_folder = self.image_processor.get_last_logs_path()
        self.log(f"This frame going to {logs_folder}")
        self.log(f"Zoom level: {self.zoom_level}")
        # os.makedirs(logs_folder, exist_ok=True)
        # cv.imwrite(f"{logs_folder}/image.png", img.get_array())
        log_data = {
            'pose_time': pose.time_seconds,
            'image_time': timestamp,
            'drone_position': cur_position_np.tolist(),
            'drone_q': cur_rot_quat.tolist(),
            'gimbal_yaw': angles[0],
            'gimbal_pitch': angles[1],
            'gimbal_roll': angles[2],
            'zoom level': self.zoom_level,
            'preds_3d': [
                {
                    'position': p.position.tolist(),
                    'id': p.id,
                } for p in preds_3d
            ]
        }
        json.dump(log_data, open(f"{logs_folder}/data.json", 'w+'), indent=4)

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

        return response


def main(args=None) -> None:
    print('Starting imaging node...')
    rclpy.init(args=args)
    node = ImagingNode()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    executor.spin()



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
