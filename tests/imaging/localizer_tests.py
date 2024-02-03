import unittest
from uavf_2024.imaging.localizer import Localizer
from uavf_2024.imaging.imaging_types import FullPrediction
import numpy as np

def test_localization(w,h, hfov, camera_pose, pos_2d, actual_position):
    localizer = Localizer(hfov, (w,h))
    x,y = pos_2d
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
            45,
            np.array([0,30,0,-90,-90,0]),
            (w//2,h//2),
            np.array([0,0,0])
        )

    def test_45_deg_down_x_axis(self):
        w, h = 400, 300
        test_localization(
            w,h,
            45,
            np.array([0,30,0,-45,-90,0]),
            (w//2,h//2),
            np.array([30,0,0])
        )

    def test_90_edge_of_fov(self):
        w,h = 400,300
        test_localization(
            w,h,
            90,
            np.array([0,30,0,-90,-90,0]),
            (0,0),
            np.array([22.5,0,-30])
        )

    def test_sim_case(self):
        w, h = 5312, 2988
        test_localization(
            w,h,
            99.9,
            np.array([-13,78,0,-86,-90,0]),
            (1367,1841),
            np.array([-19,0,-44])
        )

    def test_sim_case_2(self):
        w, h = 1920, 1080
        test_localization(
            w,h,
            50.94,
            np.array([-88, 76, 40, -100, -106.9, -3.5]),
            (1040, 835),
            np.array([-112.96, 0, 34.93])
        )

    def test_sim_case_3(self):
        w, h = 1920, 1080
        test_localization(
            w,h,
            50.94,
            (-88, 76, 40, -91.2309, -101.4257, 0.512614),
            (1559, 268),
            (-83.81858, 0, 63.73396),
        )