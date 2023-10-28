import unittest
from uavf_2024.imaging.localizer import Localizer
from uavf_2024.imaging.imaging_types import FullPrediction
import numpy as np

def test_localization(w,h, camera_pose, hfov, x,y, actual_position):
    localizer = Localizer(hfov, (w,h))
    pred = FullPrediction(
        x,y,None,None,None
    )
    pred_position = localizer.prediction_to_coords(pred, camera_pose).position
    assert np.allclose(actual_position, pred_position)
class TestLocalizer(unittest.TestCase):

    def test_straight_down(self):
        w, h = 400, 300
        test_localization(
            w,h,
            np.array([0,30,0,-90,-90,0]),
            45,
            w//2,h//2,
            np.array([0,0,0])
        )

    def test_45_deg_down_x_axis(self):
        w, h = 400, 300
        test_localization(
            w,h,
            np.array([0,30,0,-45,-90,0]),
            45,
            w//2,h//2,
            np.array([0,0,0])
        )

    def test_45_deg_down_neg_z_axis(self):
        w, h = 400, 300
        test_localization(
            w,h,
            np.array([0,30,0,-135,-90,0]),
            45,
            w//2,h//2,
            np.array([0,0,0])
        )

    def test_90_edge_of_fov(self):
        w,h = 400,300
        test_localization(
            w,h,
            np.array([0,30,0,-90,-90,0]),
            90,
            0,0,
            np.array([22.5,0,-30])
        )

    def test_sim_case(self):
        w, h = 5312, 2988
        
        localizer = Localizer(99.9, (w,h))
        camera_pose = np.array([-13,78,0,-86,-90,0])
        pred = FullPrediction(
           1367, 1841,None,None,None
        )
        actual_position = np.array([-19,0,-44])
        pred_position = localizer.prediction_to_coords(pred, camera_pose).position
        assert np.allclose(actual_position, pred_position, atol=1)