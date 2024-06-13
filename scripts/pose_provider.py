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
from rclpy.callback_groups import ReentrantCallbackGroup,MutuallyExclusiveCallbackGroup
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
    def _subscribe_to_topic(self, action: Callable[[PoseStamped], Any], callback_group) -> None:
        self.node.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            action,
            # lambda _: self.log("Got pose from mavros"),
            QOS_PROFILE,
            callback_group = callback_group
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
        self.log("Entered inner inteprolate")
        data = self._buffer.get_all_reversed()
        
        self.log("Before bisect")
        closest_idx = bisect_left([d.time_seconds for d in data], time_seconds)
        self.log("After bisect")
        
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
        self.log("Kill me")
        interp_pose = self._interpolate_from_buffer(time_seconds)
        self.log(f"now: {interp_pose}")
        if interp_pose is not None:
            return (interp_pose, True)
            
        return (self._buffer.get_fresh(), False)


class PoseProviderNode(Node):
    LOGS_BASE_DIR = Path('/home/forge/ws/logs/') # '/media/forge/SANDISK/logs/'
    
    @log_exceptions
    def __init__(self) -> None:
        group1 = MutuallyExclusiveCallbackGroup()
        group2 = MutuallyExclusiveCallbackGroup()
        group3 = MutuallyExclusiveCallbackGroup()
        # Initialize the node
        super().__init__('pose_provider_node') # type: ignore
        self.logs_path = Path(__class__.LOGS_BASE_DIR / f'{time.strftime("%m-%d %Hh%Mm")}')

        # Subscriptions ----
        self.pose_provider = PoseProvider(self, self.logs_path / "poses", 512, callback_group=group1, logger_name="pose provider")
        self.pose_provider.subscribe(self.cam_auto_point)
        self.log("Created pose provider and subscribed")
        self.camera_state = True # True if camera is pointing down for auto-cam-point. Only for auto-point FSM
        
        # self.create_subscription(
        #     PoseStamped,
        #     '/mavros/local_position/pose',
        #     lambda x: self.cam_auto_point(PoseProvider.format_data(x)), # type: ignore
        #     QOS_PROFILE,
        #     callback_group=group2
        # )

        self.get_pose_service = self.create_service(GetPose, 'get_pose_service', self.get_pose_cb, callback_group=group2)
        self.get_first_pose_service = self.create_service(GetFirstPose, 'get_first_pose_service', self.get_first_pose_cb, callback_group=group3)
        # self.point_cam_client = self.create_client(PointCam, 'recenter_service', callback_group=cb_group)
        
        self.log("Finished initializing pose provider")
        
    @log_exceptions
    def log(self, *args, **kwargs):
        self.get_logger().info(*args, **kwargs)
    
    @log_exceptions
    def get_pose_cb(self, request, response):
        self.log("Got into pose callback")
        timestamp = request.timestamp_seconds
        response.pose = self.pose_provider.get_interpolated(timestamp)[0].to_ros()
        self.log(f"About to return {response.pose}")
        return response

    @log_exceptions
    def get_first_pose_cb(self, request, response):
        self.log("Got into first pose callback")
        response.pose = self.pose_provider.get_first_datum().to_ros()
        self.log(f"About to return {response.pose}")
        return response

    @log_exceptions
    def cam_auto_point(self, current_pose: PoseDatum):
        current_z = current_pose.position.z

        # first_pose = self.pose_provider.get_first_datum()
        # if first_pose is None:
        #     self.log("First datum not found trying to auto-point camera.")
        #     return
        
        # alt_from_gnd = current_z - first_pose.position.z
        
        # threshold = 3
        # If pointed down and close to the ground, point forward
        # if self.point_cam_client.wait_for_service(0.1):
        #     if self.camera_state and alt_from_gnd < threshold: #3 meters ~ 30 feet
        #         self.camera_state = False
        #         req = PointCam.Request()
        #         req.down = False
        #         self.log("Making point cam request")
        #         self.point_cam_client.call_async(req)
        #         self.log(f"Crossing 3m down, pointing forward. Current altitude: {alt_from_gnd}")
        #     # If pointed forward and altitude is higher, point down
        #     elif not self.camera_state and alt_from_gnd > threshold:
        #         self.camera_state = True
        #         req = PointCam.Request()
        #         req.down = True
        #         self.log("Making point cam request")
        #         self.point_cam_client.call_async(req)
        #         self.log(f"Crossing 3m up, pointing down. Current altitude: {alt_from_gnd}")

def main(args=None) -> None:
    print('Starting pose provider node...')
    rclpy.init(args=args)
    node = PoseProviderNode()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    executor.spin()

if __name__ == '__main__':
    main()