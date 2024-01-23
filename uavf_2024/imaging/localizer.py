from __future__ import annotations
from .imaging_types import FullPrediction, Target3D, TargetDescription
import numpy as np
from typing import Callable, Any


class Localizer:
    def __init__(self, 
                 camera_hfov: float,
                 camera_resolution: tuple[int,int]
                 ):
        '''
        `camera_fov` is the horizontal FOV, in degrees
        `camera_resolution` (w,h) in pixels
        '''
        self.camera_hfov = camera_hfov
        self.camera_resolution = camera_resolution

    def prediction_to_coords(self, pred: FullPrediction, camera_pose: np.ndarray) -> Target3D:
        '''
            the pose is [x,y,z, altitude, azimuth, roll] in degrees, where the camera at (0,0,0) is
            pointed at the negative z axis, positive x axis is the right side of the camera and positive 
            y axis goes up from the camera. The rotations are applied relative to local frame in this order: 
            azimuth then altitude then roll.
        '''


        x,y = pred.x, pred.y
        w,h = self.camera_resolution
        focal_len = w/(2*np.tan(np.deg2rad(self.camera_hfov/2)))

        rot_alt, rot_az, rot_roll = np.deg2rad(camera_pose[3:])
        camera_position = camera_pose[:3]

        rot_alt_mat = np.array([[1,0,0],
                                [0,np.cos(rot_alt),-np.sin(rot_alt)],
                                [0,np.sin(rot_alt),np.cos(rot_alt)]])
    
        rot_az_mat = np.array([[np.cos(rot_az),0,np.sin(rot_az)],
                                [0,1,0],
                                [-np.sin(rot_az),0,np.cos(rot_az)]])
        
        rot_roll_mat = np.array([[np.cos(rot_roll),-np.sin(rot_roll),0],
                                [np.sin(rot_roll),np.cos(rot_roll),0],
                                [0,0,1]])

        # the vector pointing out the camera at the target, if the camera was facing positive Z
        initial_direction_vector = np.array([x-w//2,h//2-y,-focal_len])

        # rotate the vector to match the camera's rotation
        rotated_vector = rot_az_mat @ rot_alt_mat @ rot_roll_mat @ initial_direction_vector

        # solve camera_pose + t*rotated_vector = [x,0,z] = target_position
        t = -camera_position[1]/rotated_vector[1]
        target_position = camera_position + t*rotated_vector
        assert abs(target_position[1])<1e-3

        return Target3D(target_position, pred.description, id=f"img_{pred.img_id}/det_{pred.det_id}")
