import unittest
from uavf_2024.imaging.target_tracker import TargetTracker
from uavf_2024.imaging.imaging_types import TargetDescription, Target3D
import numpy as np

class TestTargetTracker(unittest.TestCase):
    def test_with_certain_match(self):
        tracker = TargetTracker(None, None, None, None)
        
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

        # violating private-ness for testing
        tracker._targets = [
           target_1, target_2 
        ]

        desc = TargetDescription(
                    np.eye(13)[1],
                    np.eye(35)[3],
                    np.eye(8)[3],
                    np.eye(8)[7],
                )
            
        assert tracker.closest_match(desc) is target_1

    def test_with_near_match(self):
        tracker = TargetTracker(None, None, None, None)
        
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

        # violating private-ness for testing
        tracker._targets = [
           target_1, target_2 
        ]

        desc = TargetDescription(
                    np.eye(13)[2],
                    np.eye(35)[2],
                    np.eye(8)[2],
                    np.eye(8)[7],
                )
        assert tracker.closest_match(desc) is target_2

