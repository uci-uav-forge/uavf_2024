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
            camera_pose needs to be like [x,y,z,rot_x,rot_y,rot_z]
            where x,y, and z are euclidian right-hand coordinates, and rot_x,rot_y,and rot_z
            are right-handed rotation angles in degrees around their respective axes, applied in order x->y->z
        '''

        x,y = pred.x, pred.y
        w,h = self.camera_resolution
        focal_len = w/(2*np.tan(np.deg2rad(self.camera_hfov/2)))


        rot_x_rad, rot_y_rad, rot_z_rad = np.deg2rad(camera_pose[3:])
        camera_position = camera_pose[:3]

        rot_x_mat = np.array([[1,0,0],
                                [0,np.cos(rot_x_rad),-np.sin(rot_x_rad)],
                                [0,np.sin(rot_x_rad),np.cos(rot_x_rad)]])
    
        rot_y_mat = np.array([[np.cos(rot_y_rad),0,np.sin(rot_y_rad)],
                                [0,1,0],
                                [-np.sin(rot_y_rad),0,np.cos(rot_y_rad)]])
        
        rot_z_mat = np.array([[np.cos(rot_z_rad),-np.sin(rot_z_rad),0],
                                [np.sin(rot_z_rad),np.cos(rot_z_rad),0],
                                [0,0,1]])

        # the vector pointing out the camera at the target, if the camera was facing positive Z
        initial_direction_vector = np.array([w//2-x,h//2-y,focal_len])

        # rotate the vector to match the camera's rotation
        rotated_vector = rot_z_mat @ rot_y_mat @ rot_x_mat @ initial_direction_vector

        # solve camera_pose + t*rotated_vector = [x,y,0] = target_position
        t = -camera_position[2]/rotated_vector[2]
        target_position = camera_position + t*rotated_vector
        assert abs(target_position[2])<1e-3

        return Target3D(target_position, pred.description)