from .imaging_types import FullPrediction, Target3D, TargetDescription
from .utils import calc_match_score
import numpy as np
from typing import Callable, Any


class TargetTracker:
    def __init__(self, 
                 latlng_to_local: Callable[[tuple[float,float]], np.ndarray], 
                 local_to_latlng: Callable[[np.ndarray], tuple[float,float]],
                 camera_fov: float,
                 camera_resolution: int
                 ):
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
        
        raise NotImplementedError()

    def closest_match(self, target_desc: TargetDescription) -> Target3D:
        '''
        Finds closest tracked target to the description using the class probabilities.
        Multiplies together P(class==target_class) for all 4 labels, which should be the mathematically correct way to do this, since desc==target_desc means shape_class==target_shape_class AND letter_class==target_letter_class AND ... etc.
        '''
        best_match, best_score = self._targets[0], 0

        
        '''Converts the target description value from an int into a numpy prob distribution array to be dot producted with the list of targets 
            '''
        target_desc_list = list(vars(target_desc).values())

        target_desc = TargetDescription(
                    np.eye(13)[target_desc_list[0]],
                    np.eye(35)[target_desc_list[1]],
                    np.eye(8)[target_desc_list[2]],
                    np.eye(8)[target_desc_list[3]],
                )
        
        for target_detail, target_probability in vars(target_desc).items():
                target_probability[ target_probability == 0 ] = 10 ** -4 
        
        for target in self._targets:
            # converts all 0's in the prob numpy array for target into a small number and then back after the comparison
            for target_detail, target_probability in vars(target.description).items():
                target_probability[ target_probability == 0 ] = 10 ** -4 

            new_score = calc_match_score(target_desc, target.description)

            for target_detail, target_probability in vars(target.description).items():
                target_probability[ target_probability == 10 ** -4 ] = 0

            if new_score > best_score:
                best_match = target
                best_score = new_score

        return best_match

    def get_all_targets(self):
        return self._targets