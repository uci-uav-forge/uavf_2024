import unittest
from uavf_2024.imaging import Camera
from scipy.spatial.transform import Rotation as R

class TestCameraTransforms(unittest.TestCase):
    def test_with_forward_camera(self):
        drone_rot = R.identity()
        cam_rot = (0,0,0)

        world_rotvec = Camera.orientation_in_world_frame(drone_rot, cam_rot).as_rotvec()
        print(world_rotvec)
        self.assertTrue(all(world_rotvec == [1,0,0]))
