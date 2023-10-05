from .imaging_types import FullPrediction, Target3D, TargetDescription
from dataclasses import dataclass
import numpy as np


class TargetTracker:
    def __init__(self):
        self.targets: list[Target3D] = []

    def update_with_new_data(self, new_preds: list[tuple[FullPrediction,np.ndarray]]):
        '''
            new_preds should look like this:
            [
                (prediction1, camera_pose1),
                (prediction2, camera_pose2)
                ... etc ...
            ]
            TODO: figure out format for camera pose, and update the Target3D class with our final decision for how to define a global position
        
        '''
        raise NotImplementedError()

    def closest_match(self, target_desc: TargetDescription) -> Target3D:
        '''
        Finds closest tracked target to the description using the class probabilities.
        Multiplies together P(class==target_class) for all 4 labels, which should be the mathematically correct way to do this, since desc==target_desc means shape_class==target_shape_class AND letter_class==target_letter_class AND ... etc.
        '''
        best_match, best_score = self.targets[0], 0

        for target in self.targets:
            shape_score = target.shape_probs[target_desc.shape]
            letter_score = target.letter_probs[target_desc.letter]
            shape_color_score = target.shape_col_probs[target_desc.shape_color]
            letter_color_score = target.letter_col_probs[target_desc.letter_color]

            new_score = shape_score * letter_score * shape_color_score * letter_color_score
            if new_score > best_score:
                best_match = target
                best_score = new_score

        return best_match