import unittest
from uavf_2024.imaging.localizer import Localizer
from uavf_2024.imaging.imaging_types import FullPrediction
import numpy as np

class TestLocalizer(unittest.TestCase):
    def test_straight_down(self):
        w, h = 400, 300
        localizer = Localizer(45, (w,h))
        camera_pose = np.array([0,0,30,0,90,0])
        pred = FullPrediction(
           w//2,h//2,None,None,None
        )
        actual_position = np.array([0,0,0])
        pred_position = self.localizer.prediction_to_coords(pred, camera_pose).position
        assert np.allclose(actual_position, pred_position)

    def test_harder_case(self):
        w, h = 5312, 2988
        localizer = Localizer(67.6, (w,h))
        camera_pose = np.array([-13,78,0,-86,-90,0])
        pred = FullPrediction(
           1367, 1841,None,None,None
        )
        actual_position = np.array([-19,0,-44])
        pred_position = self.localizer.prediction_to_coords(pred, camera_pose).position
        assert np.allclose(actual_position, pred_position)