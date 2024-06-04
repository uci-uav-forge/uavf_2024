from __future__ import annotations
from .imaging_types import FullBBoxPrediction, Target3D
import numpy as np
from scipy.spatial.transform import Rotation

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

    @staticmethod
    def from_focal_length(cam_focal_len: float, cam_res: tuple[int,int]):
        '''
        Create a Localizer from a focal length and resolution
        '''
        w,h = cam_res
        hfov = 2*np.arctan(w/(2*cam_focal_len))
        return Localizer(np.rad2deg(hfov), cam_res)

    def prediction_to_coords(self, pred: FullBBoxPrediction, camera_pose: tuple[np.ndarray, Rotation]) -> Target3D:
        '''
            `camera_pose` is [x,y,z, qx, qy, qz, qw]

            currently, the code assumes that for quaternion [0,0,0,1],
            the camera is pointed at the negative z axis,
            with positive x to the right of the image, and positive y up from the image
        '''


        x,y = pred.x, pred.y
        w,h = self.camera_resolution
        focal_len = w/(2*np.tan(np.deg2rad(self.camera_hfov/2)))

        camera_position, rot_transform = camera_pose
        # the vector pointing out the camera at the target, if the camera was facing positive Z
        initial_direction_vector = np.array([x-w/2,h/2-y,-focal_len])

        # rotate the vector to match the camera's rotation
        rotated_vector = rot_transform.as_matrix() @ initial_direction_vector

        # solve camera_pose + t*rotated_vector = [x,0,z] = target_position
        t = -camera_position[1]/rotated_vector[1]
        target_position = camera_position + t*rotated_vector
        assert abs(target_position[1])<1e-3

        return Target3D(target_position, pred.descriptor, id=f"img_{pred.img_id}/det_{pred.det_id}")
        
    def coords_to_2d(self, coords: tuple[float,float,float], camera_pose: tuple[np.ndarray, Rotation]) -> tuple[int, int]:
        cam_position, rot_transform = camera_pose[:3]

        relative_coords = coords - cam_position 

        # apply inverse rotation
        rotated_vector = rot_transform.inv().as_matrix() @ relative_coords

        w,h = self.camera_resolution
        focal_len = w/(2*np.tan(np.deg2rad(self.camera_hfov/2)))

        # divide by the z component to get the 2d position
        x = -rotated_vector[0]*focal_len/rotated_vector[2] + w/2
        y = h/2 + rotated_vector[1]*focal_len/rotated_vector[2]
        return (x,y)