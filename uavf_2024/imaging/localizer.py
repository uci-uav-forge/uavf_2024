from __future__ import annotations
from .imaging_types import FullBBoxPrediction, Target3D
import numpy as np
from scipy.spatial.transform import Rotation

class Localizer:
    def __init__(self, 
                 camera_hfov: float,
                 camera_resolution: tuple[int,int],
                 cam_initial_directions: tuple[np.ndarray, np.ndarray],
                 ground_axis: int,
                 ground_coordinate: float = 0 ):
        '''
        `camera_fov` is the horizontal FOV, in degrees
        `camera_resolution` (w,h) in pixels

        `cam_initial_directions` is a tuple of two vectors. the first vector is the direction the camera is facing when R is the identity, in world frame,
        and the second is the direction toward the right of the frame when R is the identity, in the world frame. Both need to be unit vectors.

        `ground_coordinate` is the level of the ground plane. For example, a value of one and ground_axis being 2 would mean the ground is at z=1
        '''
        self.camera_hfov = camera_hfov
        self.camera_resolution = camera_resolution
        self.cam_initial_directions = cam_initial_directions
        self.ground_axis = ground_axis
        self.ground_coordinate = ground_coordinate 
    
    @staticmethod
    def drone_initial_directions() -> tuple[np.ndarray, np.ndarray]:
        '''
        Returns the initial directions of the drone camera, in the world frame, for this year's comp quad.
        '''
        return (np.array([1,0,0]), np.array([0,-1, 0]))

    @staticmethod
    def from_focal_length(cam_focal_len: float, cam_res: tuple[int,int], cam_initial_directions: tuple[np.ndarray, np.ndarray], ground_axis: int):
        '''
        Create a Localizer from a focal length and resolution
        '''
        w,h = cam_res
        hfov = 2*np.arctan(w/(2*cam_focal_len))
        return Localizer(np.rad2deg(hfov), cam_res, cam_initial_directions, ground_axis)

    def prediction_to_coords(self, pred: FullBBoxPrediction, camera_pose: tuple[np.ndarray, Rotation]) -> Target3D:
        '''
            `camera_pose` is [x,y,z, R]
        '''


        x,y = pred.x, pred.y
        w,h = self.camera_resolution
        focal_len = w/(2*np.tan(np.deg2rad(self.camera_hfov/2)))

        camera_position, rot_transform = camera_pose
        # the vector pointing out the camera at the target, if the camera was facing positive Z

        positive_y_direction = np.cross(self.cam_initial_directions[0], self.cam_initial_directions[1])

        initial_direction_vector = focal_len * self.cam_initial_directions[0] + (x-w/2)*self.cam_initial_directions[1] + (y - h/2)*positive_y_direction

        # rotate the vector to match the camera's rotation
        rotated_vector = rot_transform.apply(initial_direction_vector)

        # solve camera_pos + t*rotated_vector = [x,ground_coord,z] = target_position
        t = (self.ground_coordinate-camera_position[self.ground_axis])/rotated_vector[self.ground_axis]
        target_position = camera_position + t*rotated_vector
        assert abs(target_position[self.ground_axis])<1e-3

        return Target3D(target_position, pred.descriptor, id=f"img_{pred.img_id}/det_{pred.det_id}")
        
    def coords_to_2d(self, coords: tuple[float,float,float], camera_pose: tuple[np.ndarray, Rotation]) -> tuple[int, int]:
        cam_position, rot_transform = camera_pose

        relative_coords = coords - cam_position 

        # apply inverse rotation
        rotated_vector = rot_transform.inv().apply(relative_coords)

        # find transformation that maps camera frame to initial directions
        cam_y_direction = np.cross(self.cam_initial_directions[0], self.cam_initial_directions[1])

        w,h = self.camera_resolution
        focal_len = w/(2*np.tan(np.deg2rad(self.camera_hfov/2)))

        scale_factor = focal_len/np.dot(rotated_vector, self.cam_initial_directions[0])
        x_component = np.dot(rotated_vector, self.cam_initial_directions[1])
        y_component = np.dot(rotated_vector, cam_y_direction)

        x = w/2 + x_component*scale_factor
        y = h/2 + y_component*scale_factor
        return (int(x),int(y))