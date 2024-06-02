from uavf_2024.imaging.drone_tracker import DroneTracker
from uavf_2024.imaging.particle_filter import BoundingBox
from uavf_2024.imaging.camera_model import CameraModel
import shutil

#shutil.rmtree()
import unittest
import numpy as np
from numpy import cos, pi, sin, tan
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class TestDroneTracker(unittest.TestCase):
    def setUp(self):
        self.resolution = (1920, 1080)
        self.fov = 90
        self.focal_len_pixels = self.resolution[0] / (2 * tan(np.deg2rad(self.fov / 2)))
        self.filter = DroneTracker(self.resolution, self.focal_len_pixels)

    def compute_measurement(self, cam_pose: tuple[np.ndarray, R], state: np.ndarray) -> BoundingBox:
        '''
        `state` is the state of the track, which is a 7-element array
        '''
        # If behind the camera, return a box with 0 area
        if np.dot(state[:3] - cam_pose[0], cam_pose[1].apply([0, 0, 1])) < 0:
            return BoundingBox(0, 0, 0, 0)

        cam = CameraModel(self.focal_len_pixels, 
                          [self.resolution[0] / 2, self.resolution[1] / 2], 
                          cam_pose[1].as_matrix(), 
                          cam_pose[0].reshape(3, 1))

        state_position = state[:3]
        state_radius = state[-1]

        ray_to_center = state_position - cam_pose[0]

        # Monte Carlo to find the circumscribed rectangle around the sphere's projection into the camera
        n_samples = 100

        # Sample points on the sphere
        random_vector = np.random.randn(3, n_samples)
        random_vector -= np.dot(random_vector.T, ray_to_center) * np.repeat([ray_to_center / np.linalg.norm(ray_to_center)], n_samples, axis=0).T
        random_vector = random_vector / np.linalg.norm(random_vector, axis=0) * state_radius

        # Project points into the camera
        projected_points = cam.project(state_position.reshape((3, 1)) + random_vector)
        x_min = np.min(projected_points[0])
        x_max = np.max(projected_points[0])
        y_min = np.min(projected_points[1])
        y_max = np.max(projected_points[1])

        return BoundingBox((x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min)


    def save_visualization(self, test_case_name, fig, vis_num):
        '''Given a test case name and a number for the visual, saves a figure to a particular folder path'''
        folder_path = os.path.join(CURRENT_DIR, 'visualizations', 'drone_tracker', test_case_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f'{test_case_name}_visual_{vis_num}.png')
        fig.savefig(file_path)
        plt.close(fig)

    
    def moving_target_path(self, cam_positions: np.ndarray, cam_rotations: np.ndarray, target_positions: np.ndarray, test_case_name: str):
        '''
        Simulates and visualizes tracking of a moving target by a drone.

        Parameters:
        cam_positions (np.ndarray): Camera positions at each time step.
        cam_rotations (np.ndarray): Camera orientations at each time step.
        target_positions (np.ndarray): Target states (position, velocity, radius) at each time step.
        test_case_name (str): Name for saving visualizations.

        This function follows the following process:
        1. Initialize plot with camera and target positions.
        2. For each time step:
        a. Compute target bounding box from camera pose.
        b. Predict and update filter state.
        c. Visualize and save filter state and covariances.
        3. Plot and save variance of filter state estimates over time.
        4. Assert filter convergence to expected final target position.
        '''

        n_cam_samples = len(cam_positions)

        fig, ax = plt.subplots()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        for step, (cam_pos, cam_rot) in enumerate(zip(cam_positions, cam_rotations)):
            look_direction = cam_rot.apply([0, 0, 1])
            ax.scatter(cam_pos[0], cam_pos[2], label='camera position' if step == 0 else "")
            ax.plot([cam_pos[0], cam_pos[0] + look_direction[0]], [cam_pos[2], cam_pos[2] + look_direction[2]], 'g', label='camera direction' if step == 0 else "")
            ax.plot([target_positions[step][0]], [target_positions[step][2]], 'ro', label='actual position' if step == 0 else "")
        ax.legend()
        self.save_visualization(test_case_name, fig, 1)

        covariances = []
        fig_bounds = 40

        for step, (cam_pos, cam_rot) in enumerate(tqdm(zip(cam_positions, cam_rotations))):
            target_state = target_positions[step]  # Current state of the target

            bbox = self.compute_measurement((cam_pos, cam_rot), target_state)
            self.filter.predict(0.5)
            self.filter.update((cam_pos, cam_rot), [bbox])

            fig = self.filter.tracks[0].filter.visualize(fig_bounds)
            ax = fig.axes[0]
            look_direction = cam_rot.apply([0, 0, 1])
            ax.plot([cam_pos[0], cam_pos[0] + look_direction[0]], [cam_pos[2], cam_pos[2] + look_direction[2]], 'g', label='camera (our drone)')
            ax.plot([target_positions[step][0]], [target_positions[step][2]], 'ro', label='actual position')
            
            mean = self.filter.tracks[0].filter.mean()
            cov = np.diag(self.filter.tracks[0].filter.covariance())
            ax.plot([mean[0]], [mean[2]], 'yo', label='estimated position')
            fig.legend()
            self.save_visualization(test_case_name, fig, step + 2)
            covariances.append(np.diag(self.filter.tracks[0].filter.covariance()))

        fig, ax = plt.subplots()
        labels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'r']
        for i in range(len(covariances[0])):
            ax.plot([c[i] for c in covariances], label=labels[i])
        ax.legend()
        ax.set_title("Variance vs timestep")
        self.save_visualization(test_case_name, fig, len(covariances) + 2)

        self.assertTrue(len(self.filter.tracks) == 1)
        final_state = self.filter.tracks[0].filter.mean()
        expected_final_position = target_positions[-1][:3]
        self.assertTrue(np.allclose(final_state[:3], expected_final_position, atol=0.1), f"filter converged to {final_state}")

    def test_stationary_target(self):
        n_cam_samples = 20
        cam_pos_radius = 10

        cam_positions = [cam_pos_radius * np.array([sin(2 * pi * i / n_cam_samples), 0, -cos(2 * pi * i / n_cam_samples)]) for i in range(n_cam_samples)]
        cam_rotations = [R.from_euler('xyz', [0, -2 * pi * i / n_cam_samples, 0]) for i in range(n_cam_samples)]

        target_positions = np.array([[0, 0, 0, 0, 0, 0, 0.35]] * n_cam_samples)  # Stationary target

        self.moving_target_path(np.array(cam_positions), cam_rotations, target_positions, 'test_stationary_target')

    
    def test_moving_target_straight_line(self):
        cam_positions = np.array([[i, 0, 10] for i in range(20)])
        cam_rotations = [R.from_euler('xyz', [0, 0, 0]) for _ in range(20)]  # Camera looking straight

        target_positions = np.array([[0, i, 10, 0, 1, 0, 0.35] for i in range(20)])

        self.moving_target_path(cam_positions, cam_rotations, target_positions, 'test_moving_target_straight_line')
    
    def test_moving_target_circling_camera(self):
        cam_positions = np.array([[0, 0, 10]] * 20)  # Stationary camera
        cam_rotations = [R.from_euler('xyz', [0, 0, 0]) for _ in range(20)]  # Camera looking straight

        angles = np.linspace(0, 2 * np.pi, 20)
        target_radius = 5
        target_positions = np.array([[target_radius * np.cos(angle), target_radius * np.sin(angle), 10, 
                                    -target_radius * np.sin(angle), target_radius * np.cos(angle), 0, 0.35] for angle in angles])

        self.moving_target_path(cam_positions, cam_rotations, target_positions, 'test_moving_target_circling_camera')
    
    def test_moving_target_spiral_around_camera(self):
        cam_positions = np.array([[0, 0, 10]] * 20)  # Stationary camera
        cam_rotations = [R.from_euler('xyz', [0, 0, 0]) for _ in range(20)]  # Camera looking straight

        # Target moves in a spiral path around the camera
        num_steps = 20
        angles = np.linspace(0, 4 * np.pi, num_steps)  # Two full circles
        target_radius = 5
        z_step = 0.5
        target_positions = np.array([[target_radius * np.cos(angle), target_radius * np.sin(angle), 10 + z_step * i, 
                                    -target_radius * np.sin(angle), target_radius * np.cos(angle), z_step, 0.35] for i, angle in enumerate(angles)])

        self.moving_target_path(cam_positions, cam_rotations, target_positions, 'test_moving_target_spiral_around_camera')
    
    def test_moving_target_spiral_with_camera_movement(self):
        # Target moves in a spiral path around the camera
        num_steps = 20
        angles = np.linspace(0, 4 * np.pi, num_steps)  # Two full circles
        target_radius = 5
        z_step = 0.5
        target_positions = np.array([[target_radius * np.cos(angle), target_radius * np.sin(angle), 10 + z_step * i, 
                                    -target_radius * np.sin(angle), target_radius * np.cos(angle), z_step, 0.35] for i, angle in enumerate(angles)])

        # Camera starts behind the target and speeds up to catch up
        cam_positions = np.array([[target_radius * np.cos(angle - 0.2), target_radius * np.sin(angle - 0.2), 10 + z_step * i] for i, angle in enumerate(angles)])
        cam_rotations = [R.from_euler('xyz', [0, 0, angle - 0.2]) for angle in angles]

        self.moving_target_path(cam_positions, cam_rotations, target_positions, 'test_moving_target_spiral_with_camera_movement')

if __name__ == '__main__':
    unittest.main()