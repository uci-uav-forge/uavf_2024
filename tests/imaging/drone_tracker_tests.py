from uavf_2024.imaging import DroneTracker, BoundingBox
import unittest
import numpy as np
from numpy import cos, pi, sin, tan
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class TestDroneTracker(unittest.TestCase):
    def test_stationary_target(self):
        resolution = (1920,1080)
        fov = 90
        focal_len_pixels = resolution[0]/(2*tan(fov/2))
        filter = DroneTracker(resolution, focal_len_pixels)

        n_cam_samples = 10
        # make camera poses in a circle around the origin
        cam_positions = [np.array([sin(2*pi*i/n_cam_samples),0,-cos(2*pi*i/n_cam_samples)]) for i in range(n_cam_samples)]
        # make rotations that point the camera at the origin
        cam_rotations = [R.from_euler('xyz', [0, -2*pi*i/n_cam_samples, 0]) for i in range(n_cam_samples)]

        # visualize the camera positions
        plt.figure()
        ax = plt.axes()
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.plot([0], [0], [0], 'ro')
        for cam_pos, cam_rot in zip(cam_positions, cam_rotations):
            look_direction = cam_rot.apply([0,0,1]) * 0.1
            ax.scatter(cam_pos[0], cam_pos[2])
            ax.plot([cam_pos[0], cam_pos[0]+look_direction[0]], [cam_pos[2], cam_pos[2]+look_direction[2]])
        plt.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker_test_camera_positions.png')

        

        for cam_pos, cam_rot in zip(cam_positions, cam_rotations):
            # make a measurement
            r = 20
            measurements = [BoundingBox(960,540,2*r,2*r)]
            filter.predict(1)
            filter.update((cam_pos, cam_rot), measurements)
            print(filter.tracks[0].kf.x)


        self.assertTrue(len(filter.tracks) == 1)
        # check that the filter converged to the correct position
        self.assertTrue(np.allclose(filter.tracks[0].kf.x[:3], [0,0,0], atol=0.1), f"filter converged to {filter.tracks[0].kf.x}")

if __name__ == '__main__':
    unittest.main()
