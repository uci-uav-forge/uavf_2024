import unittest
from uavf_2024.imaging import Camera
from scipy.spatial.transform import Rotation as R
import numpy as np

def calc_angular_error(drone_rot: R, cam_rot: tuple[float,float,float], expected_direction: np.ndarray) -> float:
    world_rot = Camera.orientation_in_world_frame(drone_rot, cam_rot)
    forward_direction = world_rot.apply([1,0,0])
    angle_between = np.arccos(np.dot(forward_direction, expected_direction) / (np.linalg.norm(forward_direction) * np.linalg.norm(expected_direction)))
    return angle_between

class TestCameraTransforms(unittest.TestCase):

    def test_with_forward_camera(self):
        drone_rot = R.identity()
        cam_rot = (0,0,0)
        expected_direction = np.array([1,0,0])

        angle_between = calc_angular_error(drone_rot, cam_rot, expected_direction)
        self.assertAlmostEqual(angle_between, 0, places=2)        

    def test_with_down_camera(self):
        drone_rot = R.identity()
        cam_rot = (0,-90,0)
        expected_direction = np.array([0,0,-1])

        angle_between = calc_angular_error(drone_rot, cam_rot, expected_direction)
        self.assertAlmostEqual(angle_between, 0, places=2)



