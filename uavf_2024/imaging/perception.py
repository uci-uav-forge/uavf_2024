"""
Perception class to supercede ImagingNode by using Python-native async functionality
"""


from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import time
from typing import Literal, Sequence

import numpy as np
import cv2

from . import ImageProcessor, Camera, Localizer, PoseProvider, PoseDatum, Target3D, Image
from .mock_camera import Camera as MockCamera


LOGS_PATH = Path(f'logs/{time.strftime("%m-%d %Hh%Mm")}')


class Perception:
    """
    External interface for the perception system.
    """
    
    # We need to use the singleton pattern because there can only be one Camera,
    # and this class manages a process pool
    _INSTANCE: "Perception | None" = None
    
    @staticmethod
    def _get_existing_instance() -> "Perception | None":
        """
        Get the existing instance of Perception, or None if it doesn't exist.
        """
        return Perception._INSTANCE
    
    def __init__(self, pose_provider: PoseProvider, zoom_level: int = 3, logs_path: Path = LOGS_PATH):
        """
        A PoseProvder must be injected because it depends on a MAVROS subscription.
        This might not be neccessary in the future if we can get that data from MAVSDK.
        """
        if Perception._INSTANCE is not None:
            raise ValueError("Only one instance of Perception can be created.")
        else:
            Perception._INSTANCE = self
        
        print("Initializing Perception. Logging to", LOGS_PATH)
        
        self.logger = logging.getLogger('perception')
        
        self.zoom_level = zoom_level
        self.logs_path = logs_path
        
        # Set up camera
        self.camera = MockCamera(LOGS_PATH / 'camera')
        self.camera.setAbsoluteZoom(zoom_level)
        self.camera_state = False
        
        self.image_processor = ImageProcessor(LOGS_PATH / 'image_processor')
        
        # There can only be one process because it uses the GPU
        self.processor_pool = ProcessPoolExecutor(1)
        
        self.logging_pool = ThreadPoolExecutor(2)
        
        # This has to be injected because it needs to subscribe to the mavros topic
        self.pose_provider = pose_provider
        self.pose_provider.subscribe(self.cam_auto_point)
    
    def log(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

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


    def request_point(self, direction: Literal["down", "center"]) -> bool:
        self.log(f"Received Point Camera Down Request: {direction}")
        
        match direction:
            case "down":
                success: bool = self.camera.request_down()
            case "center":
                success: bool = self.camera.request_center()
            case _:
                raise ValueError(f"Invalid direction: {direction}")
            
        self.camera.request_autofocus()

        return success
    
    def set_absolute_zoom(self, zoom_level: int) -> bool:
        self.log(f"Received Set Zoom Request to {zoom_level}")
        
        success: bool = self.camera.setAbsoluteZoom(zoom_level)
        self.camera.request_autofocus()
        
        self.zoom_level = zoom_level
        return success

    def reset_log_dir(self):
        new_logs_dir = Path('logs/{strftime("%m-%d %H:%M")}')
        self.log(f"Starting new log directory at {new_logs_dir}")
        os.makedirs(new_logs_dir, exist_ok = True)
        
        self.image_processor.reset_log_directory(new_logs_dir / 'image_processor')
        self.camera.set_log_dir(new_logs_dir / 'camera')
        self.pose_provider.set_log_dir(new_logs_dir / 'pose')

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

    def point_camera_down(self, wait: bool = True):
        self.camera.request_down()
        
        if not wait:
            self.log("Camera pointing down, but not waiting.")
            return
        
        while abs(self.camera.getAttitude()[1] - -90) > 2:
            self.log(f"Waiting to point down. Current angle: {self.camera.getAttitude()[1] } . " )
            time.sleep(0.1)
        self.log("Camera pointed down")

    def get_image_down_async(self) -> Future[list[Target3D]]:
        """
        Non-blocking implementation of the infrence pipeline,
        calling get_image_down in a separate process.
        """
        self.log("Got into async guy")
        return self.processor_pool.submit(self.get_image_down)
    
    def _log_image_down(
        self, 
        preds_3d: list[Target3D], 
        img: Image[np.ndarray], 
        timestamp: float,
        pose: PoseDatum,
        angles: Sequence[float], 
        cur_position_np: np.ndarray,
        cur_rot_quat: np.ndarray
    ):
        logs_folder = self.image_processor.get_last_logs_path()
        os.makedirs(logs_folder, exist_ok=True)
        cv2.imwrite(f"{logs_folder}/image.png", img.get_array())
        
        self.log(f"Logging inference frame at zoom level {self.zoom_level} to {logs_folder}")
        
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

    def get_image_down(self) -> list[Target3D]:
        """
        Blocking implementation of the infrence pipeline.
        
        Autofocus, then wait till cam points down, take pic,
    
        We want to take photo when the attitude is down only. 
        """
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
        cur_rot_quat = pose.rotation.as_quat() # type: ignore

        world_orientation = self.camera.orientation_in_world_frame(pose.rotation, angles)
        cam_pose = (cur_position_np, world_orientation)

        # Get 3D predictions
        preds_3d = [localizer.prediction_to_coords(d, cam_pose) for d in detections]

        # Log data asynchronously
        self.logging_pool.submit(
            self._log_image_down, 
            preds_3d, img, timestamp, pose, angles, cur_position_np, cur_rot_quat
        )

        return preds_3d
