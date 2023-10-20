from .imaging_types import FullPrediction, Target3D, TargetDescription
from .utils import calc_match_score
import numpy as np
from typing import Callable, Any


class TargetTracker:
    def __init__(self, 
                 latlng_to_local: Callable[[tuple[float,float]], np.ndarray], 
                 local_to_latlng: Callable[[np.ndarray], tuple[float,float]],
                 camera_fov: tuple[float, float],
                 camera_resolution: tuple[int,int]
                 ):
        '''
        `camera_fov` is (h,v) where h is the horizontal FOV and v is the vertical FOV, both in degrees
        `camera_resolution` (w,h) in pixels
        '''
        self._targets: list[Target3D] = []
        self.local_to_latlng = local_to_latlng
        self.latlng_to_global = latlng_to_local
        self.camera_fov = camera_fov
        self.camera_resolution = camera_resolution

    def update_with_new_data(self, pred: FullPrediction, camera_pose: np.ndarray):
        '''
            camera_pose needs to be like [x,y,z,rot_x,rot_y,rot_z]
            where x,y, and z are euclidian right-hand coordinates, and rot_x,rot_y,and rot_z
            are right-handed rotation angles in degrees around their respective axes, applied in order x->y->z
        '''

        x,y = pred.x, pred.y
        w,h = self.camera_resolution
        h_fov, v_fov = self.camera_fov
        focal_len = w/(2*np.tan(np.deg2rad(h_fov/2)))

        # the vector pointing out the camera at the target, if the camera was facing positive Z
        initial_direction_vector = np.array([w//2-x,h//2-y,focal_len])

        


        raise NotImplementedError()

    def closest_match(self, target_desc: TargetDescription) -> Target3D:
        '''
        Finds closest tracked target to the description using the class probabilities.
        Multiplies together P(class==target_class) for all 4 labels, which should be the mathematically correct way to do this, since desc==target_desc means shape_class==target_shape_class AND letter_class==target_letter_class AND ... etc.
        '''
        best_match, best_score = self._targets[0], 0

        for target in self._targets:
            new_score = calc_match_score(target_desc, target.description)
            if new_score > best_score:
                best_match = target
                best_score = new_score

        return best_match

    def get_all_targets(self):
        return self._targets