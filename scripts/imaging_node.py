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
from scipy.spatial.transform import Rotation, Slerp

from libuavf_2024.msg import TargetDetection
from libuavf_2024.srv import PointCam, ResetLogDir, TakePicture, ZoomCam
from uavf_2024.async_utils import AsyncBuffer, OnceCallable, RosLoggingProvider, Subscriptions
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


class PoseDatum(NamedTuple):
    """
    Our representation of the pose data from the Cube.
    """
    position: Point
    rotation: Rotation
    time_seconds: float
    
    def to_json(self):
        return {
            "position": [self.position.x, self.position.y, self.position.z],
            "rotation": list(self.rotation.as_quat()),
            "time_seconds": self.time_seconds
        }


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
            QOS_PROFILE
        )
        
    def log_to_file(self, item: PoseDatum):
        logs_dir = self.get_logs_dir()
        if logs_dir is None:
            raise ValueError("Logging directory not given in initialization.")
        
        path = logs_dir / f"{item.time_seconds:.02f}.json"
        with open(path, "w") as f:
            json.dump(item.to_json(), f)
            
    def format_data(self, message: PoseStamped) -> PoseDatum:
        quaternion = message.pose.orientation
        
        return PoseDatum(
            position = message.pose.position,
            rotation = Rotation.from_quat(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w]),
            time_seconds = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        )
        
    def _interpolate_from_buffer(self, time_seconds: float, wait: bool = False) -> PoseDatum | None:
        """
        Returns the pose datum interpolated between the two data points before and after the time given.
        
        If this interpolation is not possible because the pose is too old, returns the oldest pose.
        If this interpolation is not possible because the poses are not old enough, wait until enough data is available if wait is enabled.
            Otherwise, return the newest pose
        If there is no pose available, wait if enabled. Otherwise, return None.
        """
        if self._buffer.count == 0:
            return None
        
        data = self._buffer.get_all_reversed()
        
        closest_idx = bisect_left([d.time_seconds for d in data], time_seconds)
        
        if closest_idx == 0:
            return data[0]
        
        # Poll every 100ms
        while closest_idx == len(data):
            if not wait:
                return data[closest_idx - 1]

            time.sleep(0.1)
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
        for _ in range(50):
            interp_pose = self._interpolate_from_buffer(time_seconds)
            if interp_pose is not None:
                return (interp_pose, True)
            else:
                time.sleep(0.1)
        return (self._buffer.get_fresh(), False)


class ImagingNode(Node):
    @log_exceptions
    def __init__(self) -> None:
        # Initialize the node
        super().__init__('imaging_node') # type: ignore
        self.logs_path = Path(f'logs/{time.strftime("%m-%d %H:%M")}')
        
        self.camera = Camera(self.logs_path / "camera")
        self.zoom_level = 3
        self.camera_state = True # True if camera is pointing down for auto-cam-point. Only for auto-point FSM
        self.camera.setAbsoluteZoom(self.zoom_level)
        
        self.log(f"Logging to {self.logs_path}")
        self.image_processor = ImageProcessor(self.logs_path / "image_processor")

        # Set up ROS connections
        self.log(f"Setting up imaging node ROS connections")
        
        # Subscriptions ----
        self.pose_provider = PoseProvider(self, self.logs_path / "poses", 128)
        self.pose_provider.subscribe(self.cam_auto_point)
        
        # Only start the recording once (when the first pose comes in)
        start_recording_once = OnceCallable(lambda _: self.camera.start_recording())
        self.pose_provider.subscribe(start_recording_once)

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
    def cam_auto_point(self, current_pose: PoseDatum):
        current_z = current_pose.position.z

        first_pose = self.pose_provider.get_first_datum()
        if first_pose is None:
            self.log("First datum not found trying to auto-point camera.")
            return
        
        alt_from_gnd = current_z - first_pose.position.z
        
        # If pointed down and close to the ground, point forward
        if(self.camera_state and alt_from_gnd < 3): #3 meters ~ 30 feet
            self.camera.request_center()
            self.camera_state = False
            self.camera.stop_recording()
            self.log(f"Crossing 3m down, pointing forward. Current altitude: {alt_from_gnd}")
        # If pointed forward and altitude is higher, point down
        elif(not self.camera_state and alt_from_gnd > 3):
            self.camera.request_down()
            self.camera_state = True
            self.camera.start_recording()
            self.log(f"Crossing 3m up, pointing down. Current altitude: {alt_from_gnd}")
        else:
            return
        self.camera.request_autofocus()


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
        
        first_pose = self.pose_provider.get_first_datum()
        if first_pose is None:
            self.log("First datum does not exist trying to make Localizer.")
            return
        
        localizer = Localizer.from_focal_length(
            focal_len, 
            (1920, 1080),
            (np.array([1,0,0]), np.array([0,-1, 0])),
            2,
            first_pose.position.z - 0.15 # cube is about 15cm off ground
        )
        return localizer

    @log_exceptions
    def point_camera_down(self):
        self.camera.request_down()
        while abs(self.camera.getAttitude()[1] - -90) > 2:
            self.log(f"Waiting to point down. Current angle: {self.camera.getAttitude()[1] } . " )
            time.sleep(0.1)
        self.log("Camera pointed down")

    @log_exceptions
    def get_image_down(self, request, response: list[TargetDetection]) -> list[TargetDetection]:
        '''
            autofocus, then wait till cam points down, take pic,
        
            We want to take photo when the attitude is down only. 
        '''
        self.log("Received Down Image Request")
        self.camera.request_autofocus()
        self.pose_provider.wait_for_data()

        if abs(self.camera.getAttitude()[1] - -90) > 5: # Allow 5 degrees of error (Arbitrary)
            self.point_camera_down()

        #TODO: Figure out a way to detect when the gimbal is having an aneurism and figure out how to fix it or send msg to groundstation.
        
        # Take picture and grab relevant data
        img = self.camera.get_latest_image()
        if img is None:
            self.log("Could not get image from Camera.")
            return []
        timestamp = time.time()
        self.log(f"Got image from Camera at time {timestamp}")
        
        localizer = self.make_localizer()
        if localizer is None:
            self.log("Could not get Localizer")
            return []
    
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
    print('Starting imaging node...')
    rclpy.init(args=args)
    node = ImagingNode()
    rclpy.spin(node)



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
