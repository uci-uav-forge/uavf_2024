import unittest
from uavf_2024.imaging.imaging_types import TargetDescription, Target3D
from uavf_2024.imaging.utils import calc_match_score
import numpy as np

class TestTargetTracker(unittest.TestCase):
    def test_with_certain_match(self):
        target_1 = Target3D(
                12.34,
                69.420,
                TargetDescription(
                    np.eye(13)[1],
                    np.eye(35)[3],
                    np.eye(8)[3],
                    np.eye(8)[7],
                )
            )
        
        target_2 = Target3D(
                12.34,
                69.420,
                TargetDescription(
                    np.eye(13)[2],
                    np.eye(35)[2],
                    np.eye(8)[2],
                    np.eye(8)[2],
                )
            )

        desc = TargetDescription(
                    np.eye(13)[1],
                    np.eye(35)[3],
                    np.eye(8)[3],
                    np.eye(8)[7],
                )
                
            
        assert calc_match_score(desc, target_1.description) > calc_match_score(desc, target_2.description)

    def test_with_near_match(self):
        target_1 = Target3D(
                12.34,
                69.420,
                TargetDescription(
                    np.eye(13)[1],
                    np.eye(35)[3],
                    np.eye(8)[3],
                    np.eye(8)[7],
                )
            )
        
        target_2 = Target3D(
                12.34,
                69.420,
                TargetDescription(
                    np.eye(13)[2],
                    np.eye(35)[2],
                    np.eye(8)[2],
                    np.eye(8)[2],
                )
            )

        desc = TargetDescription(
                    np.eye(13)[2],
                    np.eye(35)[2],
                    np.eye(8)[2],
                    np.eye(8)[7],
                )

        assert calc_match_score(desc, target_1.description) < calc_match_score(desc, target_2.description)

