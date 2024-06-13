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
from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation, Slerp

from libuavf_2024.msg import TargetDetection
from libuavf_2024.srv import PointCam, ResetLogDir, TakePicture, ZoomCam, GetPose, GetFirstPose
from uavf_2024.logging_utils.async_utils import AsyncBuffer, OnceCallable, RosLoggingProvider, Subscriptions, PoseDatum


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




QOS_PROFILE = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    depth = 1
)


class PoseProvider(RosLoggingProvider[PoseStamped, PoseDatum]):        
    def _subscribe_to_topic(self, action: Callable[[PoseStamped], Any]) -> None:
        self.node.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            action,
            # lambda _: self.log("Got pose from mavros"),
            QOS_PROFILE
        )
        
    def log_to_file(self, item: PoseDatum):
        logs_dir = self.get_logs_dir()
        if logs_dir is None:
            raise ValueError("Logging directory not given in initialization.")
        
        path = logs_dir / f"{item.time_seconds:.02f}.json"
        with open(path, "w") as f:
            json.dump(item.to_json(), f)
    
    @staticmethod
    def format_data(message: PoseStamped) -> PoseDatum:
        new_time = time.time()
        quaternion = message.pose.orientation
        
        return PoseDatum(
            position = message.pose.position,
            rotation = Rotation.from_quat(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w]),
            time_seconds = new_time
        )
        
    def _interpolate_from_buffer(self, time_seconds: float, timeout: float = float("inf")) -> PoseDatum | None:
        """
        Returns the pose datum interpolated between the two data points before and after the time given.
        
        If this interpolation is not possible because the pose is too old, returns the oldest pose.
        If this interpolation is not possible because the poses are not old enough, wait until enough data is available if wait is enabled.
            Otherwise, return the newest pose on timeout.
        """        
        data = self._buffer.get_all_reversed()
        
        closest_idx = bisect_left([d.time_seconds for d in data], time_seconds)
        
        if closest_idx == 0:
            return data[0]
        
        # Poll every 100ms until timeout
        start = time.time()
        while closest_idx == len(data):
            waited = time.time() - start
            self.log(f"Waiting for new pose. Waited {waited} secs")
            if waited > timeout:
                return data[closest_idx - 1]

            time.sleep(0.1)
            data = self._buffer.get_all_reversed()
            closest_idx = bisect_left([d.time_seconds for d in data], time_seconds)
            
        pt_before = data[closest_idx - 1]
        pt_after = data[closest_idx]

        # Interpolate between the two points
        proportion = (time_seconds - pt_before.time_seconds) / (pt_after.time_seconds - pt_before.time_seconds)
        position = Point(
            x = pt_before.position.x + proportion * (pt_after.position.x - pt_before.position.x),
            y = pt_before.position.y + proportion * (pt_after.position.y - pt_before.position.y),
            z = pt_before.position.z + proportion * (pt_after.position.z - pt_before.position.z)
        )
        key_times = [pt_before.time_seconds, pt_after.time_seconds]
        rotations = Rotation.concatenate([pt_before.rotation, pt_after.rotation])
        slerp = Slerp(key_times, rotations)
        rotation = Rotation.from_quat(slerp([time_seconds]).as_quat()[0])
        # go to and from quaternion to force returned
        # object to be single rotation

        return PoseDatum(
            position = position,
            rotation = rotation,
            time_seconds = time_seconds
        )

    def get_interpolated(self, time_seconds: float) -> tuple[PoseDatum, bool]:
        '''
        Gets the pose interpolated at `time_seconds`. Blocks if we don't have enough data in the queue
        to interpolate yet, for a maximum of 5 seconds before giving up and returning the latest datum
        regardless of its timestamp

        '''
        interp_pose = self._interpolate_from_buffer(time_seconds)
        if interp_pose is not None:
            return (interp_pose, True)
            
        return (self._buffer.get_fresh(), False)


class PoseProviderNode(Node):
    LOGS_BASE_DIR = Path('/home/forge/ws/logs/') # '/media/forge/SANDISK/logs/'
    
    @log_exceptions
    def __init__(self) -> None:
        # Initialize the node
        super().__init__('pose_provider_node') # type: ignore
        self.logs_path = Path(__class__.LOGS_BASE_DIR / f'{time.strftime("%m-%d %Hh%Mm")}')

        # Subscriptions ----
        self.pose_provider = PoseProvider(self, self.logs_path / "poses", 512)
        self.pose_provider.subscribe(self.cam_auto_point)
        self.log("Created pose provider and subscribed")
        self.camera_state = True # True if camera is pointing down for auto-cam-point. Only for auto-point FSM
        
        self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            lambda x: self.cam_auto_point(PoseProvider.format_data(x)), # type: ignore
            QOS_PROFILE
        )

        self.get_pose_service = self.create_service(GetPose, 'get_pose_service', self.get_pose_cb)
        self.get_first_pose_service = self.create_service(GetFirstPose, 'get_first_pose_service', self.get_first_pose_cb)
        self.point_cam_client = self.create_client(PointCam, 'recenter_service')
        
        self.log("Finished initializing pose provider")
        
    @log_exceptions
    def log(self, *args, **kwargs):
        self.get_logger().info(*args, **kwargs)
    
    @log_exceptions
    def get_pose_cb(self, request, response):
        self.log("Got into pose callback")
        timestamp = request.timestamp
        response.pose = self.pose_provider.get_interpolated(timestamp).to_ros()
        return response

    @log_exceptions
    def get_first_pose_cb(self, request, response):
        self.log("Got into first pose callback")
        response.pose = self.pose_provider.get_first_datum().to_ros()
        self.log("aksdjakljdlakjdsklaj")
        return response # wtf

    @log_exceptions
    def cam_auto_point(self, current_pose: PoseDatum):
        current_z = current_pose.position.z

        first_pose = self.pose_provider.get_first_datum()
        if first_pose is None:
            self.log("First datum not found trying to auto-point camera.")
            return
        
        alt_from_gnd = current_z - first_pose.position.z
        
        threshold = 3
        # If pointed down and close to the ground, point forward
        if self.point_cam_client.wait_for_service(0.1):
            if self.camera_state and alt_from_gnd < threshold: #3 meters ~ 30 feet
                self.camera_state = False
                req = PointCam.Request()
                req.down = False
                self.log("Making point cam request")
                self.point_cam_client.call_async(req)
                self.log(f"Crossing 3m down, pointing forward. Current altitude: {alt_from_gnd}")
            # If pointed forward and altitude is higher, point down
            elif not self.camera_state and alt_from_gnd > threshold:
                self.camera_state = True
                req = PointCam.Request()
                req.down = True
                self.log("Making point cam request")
                self.point_cam_client.call_async(req)
                self.log(f"Crossing 3m up, pointing down. Current altitude: {alt_from_gnd}")

    @log_exceptions
    def get_image_down(self, request, response: list[TargetDetection]) -> list[TargetDetection]:
        '''
            autofocus, then wait till cam points down, take pic,
        
            We want to take photo when the attitude is down only. 
        '''

        self.log("Received Down Image Request")
        self.camera.request_autofocus()
        self.log("Autofocused")
        try:
            self.pose_provider.wait_for_data(1)
        except TimeoutError:
            self.log("Pose timeout, returning empty list")
            response.detections = []
            return response

        self.log("Finished waiting for data")

        if abs(self.camera.getAttitude()[1] - -90) > 5: # Allow 5 degrees of error (Arbitrary)
            self.point_camera_down()

        #TODO: Figure out a way to detect when the gimbal is having an aneurism and figure out how to fix it or send msg to groundstation.
        
        # Take picture and grab relevant data
        img = self.camera.get_latest_image()
        # img = None
        # time.sleep(1)
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
        pose, is_timestamp_interpolated = self.pose_provider.get_interpolated(timestamp - 0.75) 
        if not is_timestamp_interpolated:
            self.log("Couldn't interpolate pose.")
        self.log(f"Got pose: {angles}")

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
        os.makedirs(logs_folder, exist_ok=True)
        cv.imwrite(f"{logs_folder}/image.png", img.get_array())
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
    print('Starting pose provider node...')
    rclpy.init(args=args)
    node = PoseProviderNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()