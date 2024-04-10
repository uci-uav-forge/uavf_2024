from uavf_2024.imaging import DroneTracker, BoundingBox
import unittest
import numpy as np
from numpy import cos, pi, sin, tan
from scipy.spatial.transform import Rotation as R
import cv2 as cv

class TestDroneTracker(unittest.TestCase):
    def test_stationary_target(self):
        resolution = (1920,1080)
        fov = 90
        focal_len_pixels = resolution[0]/(2*tan(fov/2))
        filter = DroneTracker(resolution, focal_len_pixels)

        n_cam_samples = 10
        # make camera poses in a circle around the origin
        cam_positions = [np.array([cos(2*pi*i/n_cam_samples),0,sin(2*pi*i/n_cam_samples)]) for i in range(n_cam_samples)]
        # make rotations that point the camera at the origin
        cam_rotations = [R.from_euler('xyz', [0, -2*pi*i/n_cam_samples, 0]) for i in range(n_cam_samples)]

        debug_img = np.zeros((1080,1920,3), dtype=np.uint8)

        for cam_pos, cam_rot in zip(cam_positions, cam_rotations):
            # make a measurement
            r = 20
            measurements = [BoundingBox(960,540,2*r,2*r)]
            filter.predict(1)
            filter.update((cam_pos, cam_rot), measurements)


        self.assertTrue(len(filter.tracks) == 1)
        # check that the filter converged to the correct position
        self.assertTrue(np.allclose(filter.tracks[0].kf.x[:3], [0,0,0], atol=0.1))

if __name__ == '__main__':
    unittest.main()
