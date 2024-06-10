#!/usr/bin/env python3

import json
import os
import random
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Callable, Generic, NamedTuple, TypeVar
import logging
from bisect import bisect_left

import cv2 as cv
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from mavros_msgs.msg import Altitude
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation, Slerp

from libuavf_2024.msg import TargetDetection
from libuavf_2024.srv import PointCam, ResetLogDir, TakePicture, ZoomCam
from uavf_2024.async_utils import OnceCallable
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


class _PoseBuffer:
    def __init__(self, capacity: int = 4):
        if capacity < 1:
            raise ValueError(f"Buffer capacity cannot be less than 1. Got {capacity}")
        self._capacity = capacity
        self._queue: deque[PoseDatum] = deque(maxlen=self.capacity)
        
        # Lock for the whole queue. 
        # Not necessary if only the front is popped because that is thread-safe. 
        self.lock = threading.Lock()
        
    @property
    def capacity(self):
        return self._capacity
    
    @property
    def count(self):
        return len(self._queue)
        
    def __bool__(self):
        return bool(self.count)
    
    def put(self, datum: PoseDatum):
        with self.lock:
            # If the queue is too long, it'll automatically discard 
            # the item at the other end.
            self._queue.append(datum)
        
    def get_fresh(self, offset: int = 0):
        """
        Gets the item at the freshness offset specified (if specified).
        Otherwise, get the freshest datum
        """
        if offset < 0:
            raise ValueError(f"Offset cannot be less than 0. Got {offset}")
        
        with self.lock:
            return self._queue[-(offset + 1)]
        
    def get_interpolated(self, time_seconds: float) -> PoseDatum | None:
        '''
        Interpolates between the two closest points in the buffer for the given time.

        Returns None if we don't have enough data to interpolate yet
        '''

        if self.count == 0:
            raise ValueError("No data in the buffer to interpolate from.")
        
        with self.lock:
            if self._queue[-1].time_seconds < time_seconds:
                return None
            closest_idx = bisect_left([d.time_seconds for d in self._queue], time_seconds)
            times_only = [p.time_seconds for p in self._queue]
            
            if closest_idx == 0:
                return self._queue[0]
            pt_before = self._queue[closest_idx - 1]
            pt_after = self._queue[closest_idx]

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
    
    def get_all(self) -> list[PoseDatum]:
        """
        Returns all items in the buffer in the order of freshest first.
        
        Can be useful if we want a more refined search.
        """
        with self.lock:
            return list(reversed(self._queue))
        

InputT = TypeVar("InputT")
class Subscriptions(Generic[InputT]):
    """
    Manages subscriptions in a thread-safe way.
    
    This class can be used in the future to subsume ROS' subscription
    functionality when we stay within Python.
    """
    def __init__(self):
        self._callbacks: dict[float, Callable[[InputT], Any]] = {}
        self.lock = threading.Lock()
    
    def add(self, callback: Callable[[InputT], Any]) -> Callable[[], None]:
        """
        Adds the callback to the collection of subscriptions to be called
        when there is a notification.
        
        Returns a function to unsubscribe.
        """
        subscription_id = random.random()
        
        with self.lock:
            def unsubscribe():
                del self._callbacks[subscription_id]
            
            self._callbacks[subscription_id] = callback
        
        return unsubscribe

    def notify(self, new_value: InputT):
        """
        Calls all of the callbacks with the new value.
        
        Locks so that subscriptions will have to wait after a round of notifications.
        """
        with self.lock:
            for callback in self._callbacks.values():
                callback(new_value)


class PoseProvider:
    """
    Logs and buffers the world position for reading.
    
    Provides a method to subscribe to changes as well.
    """
    def __init__(
        self, 
        node_context: Node,
        logs_path: str | os.PathLike | Path | None = None, 
        buffer_size = 64
    ):
        """
        Parameters:
            logs_path: The parent directory to which to log.
            buffer_size: The number of world positions to keep in the buffer
                for offsetted access.
        """
        self._node = node_context
        self.logger = logging.getLogger("PoseProviderLogger")
        
        self._subscribers: Subscriptions[PoseDatum] = Subscriptions()
        
        self._log_index = 0
        self.logs_path = Path(logs_path) / "poses" if logs_path else None
        if self.logs_path:
            if not self.logs_path.exists():
                self.logs_path.mkdir(parents=True)
            elif not self.logs_path.is_dir():
                raise FileExistsError(f"{self.logs_path} exists but is not a directory")
        
        self._buffer = _PoseBuffer(buffer_size)
        
        self._subscribe_pose_and_altitude(self._handle_pose_update, self._handle_altitude_update)

        self.first_pose = None
        
        self.log(f"Finished intializing PoseProvider. Logging to {self.logs_path}")
        
    def log(self, message, level = logging.INFO):
        self.logger.log(level, message)
        
    def _subscribe_pose_and_altitude(self, pose_callback: Callable[[PoseStamped], Any], altitude_callback: Callable[[Altitude], Any]):        
        # Initialize Quality-of-Service profile for subscription
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )
    
        self._node.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            pose_callback,
            qos_profile
        )
        
        self._node.create_subscription(
            Altitude,
            '/mavros/altitude', 
            altitude_callback, 
            qos_profile
        )
        
    def _handle_pose_update(self, pose: PoseStamped) -> None:
        quaternion = pose.pose.orientation
        formatted = PoseDatum(
            position = pose.pose.position,
            rotation = Rotation.from_quat(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w]),
            time_seconds = pose.header.stamp.sec + pose.header.stamp.nanosec / 1e9
        )

        if self.first_pose is None:
            self.first_pose = formatted
        
        self._buffer.put(formatted)
        self._log_pose(formatted)
        self._subscribers.notify(formatted)
        
    def _handle_altitude_update(self, altitude: Altitude):
        """
        TODO: Rewrite this so that you can subscribe to the altitude as well.
        """
        if not self.logs_path:
            return

        logs_dir = self.logs_path.parent / "altitudes"
        if not logs_dir.is_dir():
            logs_dir.mkdir(parents=True)
        
        data = {
            "amsl": float(altitude.amsl),
            "local": float(altitude.local),
            "relative": float(altitude.relative),
            "terrain": float(altitude.terrain)
        }
        
        with open(logs_dir / (str(time.time()) + ".json"), "w") as f:
            json.dump(data, f)
        
    def get_first_pose(self) -> PoseDatum:
        return self.first_pose
        
    def get(self, offset: int = 0):
        """
        Gets the item at the freshness offset specified (if specified).
        Otherwise, get the freshest datum
        """
        return self._buffer.get_fresh(offset)
    
    def get_interpolated(self, time_seconds: float) -> tuple[PoseDatum, bool]:
        '''
        Gets the pose interpolated at `time_seconds`. Blocks if we don't have enough data in the queue
        to interpolate yet, for a maximum of 5 seconds before giving up and returning the latest datum
        regardless of its timestamp

        '''
        for _ in range(50):
            interp_pose = self._buffer.get_interpolated(time_seconds)
            if interp_pose is not None:
                return (interp_pose, True)
            else:
                time.sleep(0.1)
        return (self._buffer.get_fresh(), False)
    
    def _log_pose(self, pose: PoseDatum):
        if not self.logs_path:
            return
        
        with open(self.logs_path / f"{pose.time_seconds:.02f}.json", "w") as f:
            json.dump(pose.to_json(), f)
    
    def subscribe(self, callback: Callable[[PoseDatum], Any]):
        self._subscribers.add(callback)
        
    def wait_for_pose(self, timeout_seconds: float = float('inf')):
        """
        Waits until the first pose is added to the buffer.
        """
        start = time.time()
        
        while self._buffer.count == 0:
            if time.time() - start >= timeout_seconds:
                raise TimeoutError("Timed out waiting for pose")
            
            time.sleep(0.1)

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
        self.pose_provider = PoseProvider(self, self.logs_path)
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
        z = current_pose.position.z

        alt_from_gnd = z - self.pose_provider.get_first_pose().position.z
        
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
        localizer = Localizer.from_focal_length(
            focal_len, 
            (1920, 1080),
            (np.array([1,0,0]), np.array([0,-1, 0])),
            2,
            self.pose_provider.get_first_pose().position.z - 0.15 # cube is about 15cm off ground
        )
        return localizer

    @log_exceptions
    def point_camera_down(self):
        self.camera.request_down()
        while abs(self.camera.getAttitude()[1] - -90) > 2:
            self.log(f"Waiting to point down. Current angle: {self.camera.getAttitude()[1] } . " )
            time.sleep(0.1)
        self.log("Camera pointed down")
        self.camera.request_autofocus()

    @log_exceptions
    def get_image_down(self, request, response: list[TargetDetection]) -> list[TargetDetection]:
        '''
            autofocus, then wait till cam points down, take pic,
        
            We want to take photo when the attitude is down only. 
        '''
        self.log("Received Down Image Request")
        self.pose_provider.wait_for_pose()

        if abs(self.camera.getAttitude()[1] - -90) > 5: # Allow 5 degrees of error (Arbitrary)
            self.point_camera_down()

        #TODO: Figure out a way to detect when the gimbal is having an aneurism and figure out how to fix it or send msg to groundstation.
        
        # Take picture and grab relevant data
        localizer = self.make_localizer()
        img = self.camera.get_latest_image()
        timestamp = time.time()
        self.log(f"Got image from Camera at time {timestamp}")

        if img is None:
            self.log("Could not get image from Camera.")
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
