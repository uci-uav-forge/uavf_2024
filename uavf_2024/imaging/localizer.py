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

    def prediction_to_coords(self, pred: FullBBoxPrediction, camera_pose: tuple[np.ndarray, Rotation], cam_initial_directions: tuple[np.ndarray, np.ndarray], ground_axis: int) -> Target3D:
        '''
            `camera_pose` is [x,y,z, R]
            `cam_initial_direction` is a tuple of two vectors. the first vector is
            the direction the camera is facing when R is the identity, in world frame,
            and the second is the direction toward the right of the frame when R is the identity, in the world frame. Both
            need to be unit vectors.
        '''


        x,y = pred.x, pred.y
        w,h = self.camera_resolution
        focal_len = w/(2*np.tan(np.deg2rad(self.camera_hfov/2)))

        camera_position, rot_transform = camera_pose
        # the vector pointing out the camera at the target, if the camera was facing positive Z

        positive_y_direction = np.cross(cam_initial_directions[0], cam_initial_directions[1])

        initial_direction_vector = focal_len * cam_initial_directions[0] + (x-w/2)*cam_initial_directions[1] + (y - h/2)*positive_y_direction

        # rotate the vector to match the camera's rotation
        rotated_vector = rot_transform.apply(initial_direction_vector)

        # solve camera_pose + t*rotated_vector = [x,0,z] = target_position
        t = -camera_position[ground_axis]/rotated_vector[ground_axis]
        target_position = camera_position + t*rotated_vector
        assert abs(target_position[ground_axis])<1e-3

        return Target3D(target_position, pred.descriptor, id=f"img_{pred.img_id}/det_{pred.det_id}")
        
    def coords_to_2d(self, coords: tuple[float,float,float], camera_pose: tuple[np.ndarray, Rotation], cam_initial_directions: tuple[np.ndarray, np.ndarray]) -> tuple[int, int]:
        cam_position, rot_transform = camera_pose

        relative_coords = coords - cam_position 

        # apply inverse rotation
        rotated_vector = rot_transform.inv().apply(relative_coords)

        # find transformation that maps camera frame to initial directions
        cam_y_direction = np.cross(cam_initial_directions[0], cam_initial_directions[1])
        cam_initial_rot = Rotation.align_vectors([*cam_initial_directions, cam_y_direction], [
            np.array([0,0,-1], [1,0,0], [0,-1,0])
        ])


        w,h = self.camera_resolution
        focal_len = w/(2*np.tan(np.deg2rad(self.camera_hfov/2)))

        # divide by the z component to get the 2d position
        x = -rotated_vector[0]*focal_len/rotated_vector[2] + w/2
        y = h/2 + rotated_vector[1]*focal_len/rotated_vector[2]
        return (x,y)