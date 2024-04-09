from uavf_2024.imaging import DroneTracker
import unittest
import numpy as np
from numpy import cos, pi, sin
from scipy.spatial.transform import Rotation as R
import cv2 as cv

class TestDroneTracker(unittest.TestCase):
    def test_stationary_target(self):
        filter = DroneTracker()

        fov = 90
        resolution = (1920,1080)
        n_cam_samples = 10
        # make camera poses in a circle around the origin
        cam_positions = [np.array([cos(2*pi*i/n_cam_samples),0,sin(2*pi*i/n_cam_samples)]) for i in range(n_cam_samples)]
        # make rotations that point the camera at the origin
        cam_rotations = [R.from_euler('xyz', [0, -2*pi*i/n_cam_samples, 0]) for i in range(n_cam_samples)]

        debug_img = np.zeros((1080,1920,3), dtype=np.uint8)

        for cam_pose, cam_rot in zip(cam_positions, cam_rotations):
            # make a measurement
            r = 20
            measurements = [np.array([960-r,540-r,960+r,540-r])]
            filter.predict(1)
            filter.update((cam_pose, cam_rot), fov, resolution, measurements)
            # skip predict step b/c the target is stationary
            avg_pos = np.mean([p.state[:3] for p in filter.samples], axis=0)
            pos_std = np.std([p.state[:3] for p in filter.samples], axis=0)
            print(avg_pos, pos_std)

        # check that the filter converged to the correct position
        self.assertTrue(np.allclose(avg_pos, [0,0,0], atol=0.1))

if __name__ == '__main__':
    unittest.main()
