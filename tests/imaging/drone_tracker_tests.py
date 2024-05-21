from uavf_2024.imaging.drone_tracker import DroneTracker
from uavf_2024.imaging.particle_filter import BoundingBox
from uavf_2024.imaging.camera_model import CameraModel

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

    def test_stationary_target(self):
        n_cam_samples = 20
        cam_pos_radius = 10

        cam_positions = [cam_pos_radius * np.array([sin(2 * pi * i / n_cam_samples), 0, -cos(2 * pi * i / n_cam_samples)]) for i in range(n_cam_samples)]
        cam_rotations = [R.from_euler('xyz', [0, -2 * pi * i / n_cam_samples, 0]) for i in range(n_cam_samples)]

        plt.figure()
        ax = plt.axes()
        ax.set_xlim(-cam_pos_radius, cam_pos_radius)
        ax.set_ylim(-cam_pos_radius, cam_pos_radius)
        ax.plot([0], [0], 'ro')
        for cam_pos, cam_rot in zip(cam_positions, cam_rotations):
            look_direction = cam_rot.apply([0, 0, 1])
            ax.scatter(cam_pos[0], cam_pos[2])
            ax.plot([cam_pos[0], cam_pos[0] + look_direction[0]], [cam_pos[2], cam_pos[2] + look_direction[2]])

        os.makedirs(f'{CURRENT_DIR}/visualizations/drone_tracker', exist_ok=True)
        os.makedirs(f'{CURRENT_DIR}/visualizations/drone_tracker/particles', exist_ok=True)
        plt.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker/test_camera_positions.png')

        plt.figure()
        covariances = []
        fig_bounds = 15
        for cam_pos, cam_rot in tqdm(zip(cam_positions, cam_rotations)):
            # Create a measurement
            state = np.array([0, 0, 0, 0, 0, 0, 0.35])  # Example stationary target state
            bbox = self.compute_measurement((cam_pos, cam_rot), state)
            self.filter.predict(0.5)
            self.filter.update((cam_pos, cam_rot), [bbox])

            particles_fig = self.filter.tracks[0].filter.visualize(fig_bounds)
            ax = particles_fig.axes[0]
            ax.plot([0], [0], 'ro', label='actual position')
            look_direction = cam_rot.apply([0, 0, 1])
            ax.plot([cam_pos[0], cam_pos[0] + look_direction[0]], [cam_pos[2], cam_pos[2] + look_direction[2]], 'g', label='camera (our drone)')

            mean = self.filter.tracks[0].filter.mean()
            cov = np.diag(self.filter.tracks[0].filter.covariance())
            ax.plot([mean[0]], [mean[2]], 'yo', label='estimated position')
            particles_fig.legend()
            particles_fig.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker/particles/particles_{len(covariances)}.png')
            del particles_fig
            covariances.append(np.diag(self.filter.tracks[0].filter.covariance()))

        labels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'r']
        for i in range(len(covariances[0])):
            plt.plot([c[i] for c in covariances], label=labels[i])
        plt.legend()
        plt.title("Variance vs timestep")
        plt.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker/test_covariances.png')

        self.assertTrue(len(self.filter.tracks) == 1)
        final_state = self.filter.tracks[0].filter.mean()
        self.assertTrue(np.allclose(final_state[:3], [0, 0, 0], atol=0.1), f"filter converged to {final_state}")

    def test_moving_target(self):
        n_cam_samples = 20
        cam_pos_radius = 10

        cam_positions = [cam_pos_radius * np.array([sin(2 * pi * i / n_cam_samples), 0, -cos(2 * pi * i / n_cam_samples)]) for i in range(n_cam_samples)]
        cam_rotations = [R.from_euler('xyz', [0, -2 * pi * i / n_cam_samples, 0]) for i in range(n_cam_samples)]

        plt.figure()
        ax = plt.axes()
        ax.set_xlim(-cam_pos_radius, cam_pos_radius)
        ax.set_ylim(-cam_pos_radius, cam_pos_radius)
        ax.plot([0], [0], 'ro')
        for cam_pos, cam_rot in zip(cam_positions, cam_rotations):
            look_direction = cam_rot.apply([0, 0, 1])
            ax.scatter(cam_pos[0], cam_pos[2])
            ax.plot([cam_pos[0], cam_pos[0] + look_direction[0]], [cam_pos[2], cam_pos[2] + look_direction[2]])

        os.makedirs(f'{CURRENT_DIR}/visualizations/drone_tracker', exist_ok=True)
        os.makedirs(f'{CURRENT_DIR}/visualizations/drone_tracker/particles', exist_ok=True)
        plt.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker/test_camera_positions.png')

        plt.figure()
        covariances = []
        fig_bounds = 15

        # Define the target's initial state and velocity
        initial_state = np.array([0, -10, 10, 1, 1, 0, 0.35])  # Example moving target state

        for step, (cam_pos, cam_rot) in enumerate(tqdm(zip(cam_positions, cam_rotations))):
            # Update the target's state (simple linear motion model)
            target_state = initial_state.copy()
            target_state[:3] += target_state[3:6] * step * 0.5  # Update position based on velocity and time step

            # Create a measurement
            bbox = self.compute_measurement((cam_pos, cam_rot), target_state)
            self.filter.predict(0.5)
            self.filter.update((cam_pos, cam_rot), [bbox])

            particles_fig = self.filter.tracks[0].filter.visualize(fig_bounds)
            ax = particles_fig.axes[0]
            ax.plot([0], [0], 'ro', label='actual position')
            look_direction = cam_rot.apply([0, 0, 1])
            ax.plot([cam_pos[0], cam_pos[0] + look_direction[0]], [cam_pos[2], cam_pos[2] + look_direction[2]], 'g', label='camera (our drone)')

            mean = self.filter.tracks[0].filter.mean()
            cov = np.diag(self.filter.tracks[0].filter.covariance())
            ax.plot([mean[0]], [mean[2]], 'yo', label='estimated position')
            particles_fig.legend()
            particles_fig.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker/particles/particles_{len(covariances)}.png')
            del particles_fig
            covariances.append(np.diag(self.filter.tracks[0].filter.covariance()))

        labels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'r']
        for i in range(len(covariances[0])):
            plt.plot([c[i] for c in covariances], label=labels[i])
        plt.legend()
        plt.title("Variance vs timestep")
        plt.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker/test_covariances.png')

        self.assertTrue(len(self.filter.tracks) == 1)
        final_state = self.filter.tracks[0].filter.mean()
        expected_final_position = initial_state[:3] + initial_state[3:6] * (n_cam_samples - 1) * 0.5
        self.assertTrue(np.allclose(final_state[:3], expected_final_position, atol=0.1), f"filter converged to {final_state}")
    
    # def test_bounding_box_prediction(self):
    #     num_steps = 100
    #     center = [0, 0, 10]
    #     radius = 10

    #     observer_path = self.circular_path(center, radius, num_steps)
    #     target_path = self.linear_path(np.array([0, -10, 10]), np.array([20, 10, 10]), num_steps)

    #     total_loss = 0

    #     for i in range(num_steps):
    #         cam_position = observer_path[i]
    #         direction_to_target = target_path[i] - cam_position
    #         cam_rotation = R.from_rotvec(np.cross([0, 0, 1], direction_to_target))
    #         bbox = self.compute_measurement((cam_position, cam_rotation), target_path[i])
    #         self.filter.update((cam_position, cam_rotation), [bbox])

    #         estimated_state = self.filter.tracks[0].filter.mean()
    #         error = np.linalg.norm(estimated_state[:3] - target_path[i])
    #         total_loss += error**2  # Sum of squared errors

    #     mean_squared_error = total_loss / num_steps
    #     acceptable_mse = 1.0  # Define an acceptable mean squared error
    #     self.assertLess(mean_squared_error, acceptable_mse, f"Mean squared error too high: {mean_squared_error}")

    @staticmethod
    def linear_path(start_point, end_point, num_steps):
        return np.linspace(start_point, end_point, num_steps)

    @staticmethod
    def circular_path(center, radius, num_steps):
        angles = np.linspace(0, 2 * np.pi, num_steps)
        return np.array([center[0] + radius * np.cos(angles),
                         center[1] + radius * np.sin(angles),
                         center[2] * np.ones_like(angles)]).T

    @staticmethod
    def spiral_path(center, start_radius, num_steps, z_step):
        angles = np.linspace(0, 4 * np.pi, num_steps)  # Complete two full rotations
        radii = np.linspace(start_radius, 0, num_steps)  # Radii decrease to 0
        return np.array([center[0] + radii * np.cos(angles),
                         center[1] + radii * np.sin(angles),
                         center[2] + z_step * np.arange(num_steps)]).T

if __name__ == '__main__':
    unittest.main()

# class TestDroneTracker(unittest.TestCase):

#     def compute_measurement(self, cam_pose: tuple[np.ndarray, Rotation], state: np.ndarray) -> BoundingBox:
#         '''
#         `state` is the state of the track, which is a 7 element array
#         '''
#         # if behind the camera, return a box with 0 area
#         if np.dot(state[:3] - cam_pose[0], cam_pose[1].apply([0,0,1])) < 0:
#             return BoundingBox(0, 0, 0, 0)

#         cam = CameraModel(self.focal_len_pixels, 
#                         [self.resolution[0]/2, self.resolution[1]/2], 
#                         cam_pose[1].as_matrix(), 
#                         cam_pose[0].reshape(3,1))

#         state_position = state[:3]
#         state_radius = state[-1]

#         ray_to_center = state_position - cam_pose[0] 
        
#         # monte carlo to find the circumscribed rectangle around the sphere's projection into the camera
#         # there's probably a better way to do this but I'm not sure what it is
#         # I tried a method where we project 4 points on the boundary and fit a 2d ellipse to their projection
#         # but the ellipse fitting was not working well
#         n_samples = 100

#         # sample points on the sphere
#         random_vector = np.random.randn(3, n_samples)
#         random_vector -= np.dot(random_vector.T, ray_to_center) * np.repeat([ray_to_center / np.linalg.norm(ray_to_center)], n_samples, axis=0).T
#         random_vector = random_vector / np.linalg.norm(random_vector, axis=0) * state_radius

#         # project points into the camera
#         projected_points = cam.project(state_position.reshape((3,1)) + random_vector)
#         x_min = np.min(projected_points[0])
#         x_max = np.max(projected_points[0])
#         y_min = np.min(projected_points[1])
#         y_max = np.max(projected_points[1])

#         return BoundingBox((x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min)



#     def test_stationary_target(self):
#         # object should be radius ~0.35 at the origin, not moving
#         resolution = (1920,1080)
#         fov = 90
#         focal_len_pixels = resolution[0]/(2*tan(fov/2))
#         filter = DroneTracker(resolution, focal_len_pixels)

#         n_cam_samples = 20
#         # make camera poses in a circle around the origin
#         cam_pos_radius = 10
#         cam_positions = [cam_pos_radius*np.array([sin(2*pi*i/n_cam_samples),0,-cos(2*pi*i/n_cam_samples)]) for i in range(n_cam_samples)]
#         # make rotations that point the camera at the origin
#         cam_rotations = [R.from_euler('xyz', [0, -2*pi*i/n_cam_samples, 0]) for i in range(n_cam_samples)]

#         # visualize the camera positions
#         plt.figure()
#         ax = plt.axes()
#         ax.set_xlim(-cam_pos_radius,cam_pos_radius)
#         ax.set_ylim(-cam_pos_radius,cam_pos_radius)
#         ax.plot([0], [0], [0], 'ro')
#         for cam_pos, cam_rot in zip(cam_positions, cam_rotations):
#             look_direction = cam_rot.apply([0,0,1])
#             ax.scatter(cam_pos[0], cam_pos[2])
#             ax.plot([cam_pos[0], cam_pos[0]+look_direction[0]], [cam_pos[2], cam_pos[2]+look_direction[2]])

#         os.makedirs(f'{CURRENT_DIR}/visualizations/drone_tracker', exist_ok=True)
#         os.makedirs(f'{CURRENT_DIR}/visualizations/drone_tracker/particles', exist_ok=True)
#         plt.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker/test_camera_positions.png')

#         plt.figure()
#         covariances = []
#         fig_bounds = 15
#         for cam_pos, cam_rot in tqdm(zip(cam_positions, cam_rotations)):
#             # make a measurement
#             r = 20
#             measurements = [BoundingBox(960,540,2*r,2*r)]
#             filter.predict(0.5)
#             # if len(filter.tracks) > 0:
#             #     particles_fig = filter.tracks[0].filter.visualize(fig_bounds)
#             #     particles_fig.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker/particles/particles_{len(covariances)}a.png')
#             filter.update((cam_pos, cam_rot), measurements)

#             particles_fig = filter.tracks[0].filter.visualize(fig_bounds)
#             # draw the camera position
#             ax = particles_fig.axes[0]
#             ax.plot([0], [0], [0], 'ro', label='actual position')
#             look_direction = cam_rot.apply([0,0,1])
#             ax.plot([cam_pos[0], cam_pos[0]+look_direction[0]], [cam_pos[2], cam_pos[2]+look_direction[2]], 'g', label='camera (our drone)')

#             # plot mean and covariance of the filter
#             mean = filter.tracks[0].filter.mean()
#             cov = np.diag(filter.tracks[0].filter.covariance())
#             ax.plot([mean[0]], [mean[2]], 'yo', label='estimated position')
#             particles_fig.legend()
#             particles_fig.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker/particles/particles_{len(covariances)}.png')
#             del particles_fig
#             covariances.append(np.diag(filter.tracks[0].filter.covariance()))

#         labels = [
#             'x', 'y', 'z', 'vx', 'vy', 'vz', 'r'
#         ]

#         for i in range(len(covariances[0])):
#             plt.plot([c[i] for c in covariances], label=labels[i])
#         plt.legend()
#         plt.title("Variance vs timestep")
#         plt.savefig(f'{CURRENT_DIR}/visualizations/drone_tracker/test_covariances.png')


#         self.assertTrue(len(filter.tracks) == 1)
#         # check that the filter converged to the correct position
#         final_state = filter.tracks[0].filter.mean()
#         self.assertTrue(np.allclose(final_state[:3], [0,0,0], atol=0.1), f"filter converged to {final_state}")

#     def linear_path(start_point, end_point, num_steps):
#         return np.linspace(start_point, end_point, num_steps)
    
#     def circular_path(center, radius, num_steps):
#         angles = np.linspace(0, 2 * np.pi, num_steps)
#         return np.array([center[0] + radius * np.cos(angles),
#                         center[1] + radius * np.sin(angles),
#                         center[2] * np.ones_like(angles)]).T
    
#     def spiral_path(center, start_radius, num_steps, z_step):
#         angles = np.linspace(0, 4 * np.pi, num_steps)  # Complete two full rotations
#         radii = np.linspace(start_radius, 0, num_steps)  # Radii decrease to 0
#         return np.array([center[0] + radii * np.cos(angles),
#                      center[1] + radii * np.sin(angles),
#                      center[2] + z_step * np.arange(num_steps)]).T
    
#     def test_bounding_box_prediction(self):
#         num_steps = 100
#         center = [0, 0, 10]
#         radius = 10

#         observer_path = circular_path(center, radius, num_steps)
#         target_path = linear_path(np.array([0, -10, 10]), np.array([20, 10, 10]), num_steps)

#         resolution = (1920, 1080)
#         fov = 90
#         focal_len_pixels = resolution[0] / (2 * np.tan(np.deg2rad(fov / 2)))
#         drone_tracker = DroneTracker(resolution, focal_len_pixels)

#         for i in range(num_steps):
#             cam_position = observer_path[i]
#             direction_to_target = target_path[i] - cam_position
#             cam_rotation = R.from_rotvec(np.cross([0, 0, 1], direction_to_target))
#             bbox = self.compute_measurement((cam_position, cam_rotation), target_path[i])
#             drone_tracker.update((cam_position, cam_rotation), [bbox])

#             estimated_state = drone_tracker.state_estimate()
#             error = np.linalg.norm(estimated_state[:3] - target_path[i])
#             acceptable_error = 1.0  # Define an acceptable error
#             self.assertLess(error, acceptable_error)


# if __name__ == '__main__':
#     unittest.main()
