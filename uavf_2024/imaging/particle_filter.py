import numpy as np
from scipy.spatial.transform import Rotation
from uavf_2024.imaging.camera_model import CameraModel
from uavf_2024.imaging.imaging_types import BoundingBox
from torchvision.ops import box_iou
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
from dataclasses import dataclass
import torch

@dataclass
class Measurement:
    pose: tuple[np.ndarray, Rotation]
    box: BoundingBox

# Note: this implementation is slow because it's object-oriented, but it's easier to understand
class Particle:
    '''
    State is [x,y,z,dx,dy,dz,radius] for a bounding sphere
    '''
    def __init__(self, state: np.ndarray, likelihood: float, history: list[np.ndarray] = None):
        self.state = state
        self.likelihood = likelihood 
        if history is None:
            self.history = []
        else:
            self.history = history

    def step(self, dt: float, pos_noise_std: float = 0, vel_noise_std: float = 0, radius_noise_std: float = 0):
        '''
        Updates the state of the particle, adding noise to the position, velocity, and radius
        '''
        self.history.append(self.state.copy())
        self.state[0:3] += dt*(self.state[3:6] + np.random.randn(3) * pos_noise_std)
        self.state[3:6] += dt*np.random.randn(3)* vel_noise_std
        self.state[6] += dt*np.random.randn(1)*radius_noise_std
        if np.linalg.norm(self.state[:3]) > 300:
            print("History:")
            for s in self.history:
                print(s)

    def copy(self):
        return Particle(self.state.copy(), self.likelihood, [*self.history])

class ParticleFilter:
    def __init__(self, 
                 initial_measurement: Measurement, 
                 resolution: tuple[int,int], 
                 focal_len_pixels: float, 
                 num_particles: int = 1000, 
                 missed_detection_weight: float = 1e-4, 
                 pos_noise_std: float = 0.1, 
                 vel_noise_std: float = 0, 
                 radius_noise_std: float = 0
        ):
        # not the same as len(self.samples) because we might add samples during `update` but then reduce during `resample`
        self.resolution = resolution
        self.focal_len_pixels = focal_len_pixels
        self.num_samples = num_particles 
        self.missed_detection_weight = missed_detection_weight # likelihood to give a particle if it's not detected
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.radius_noise_std = radius_noise_std

        self.samples: list[Particle] = self.gen_samples_from_measurement(initial_measurement.pose, initial_measurement.box, num_particles)

    def mean(self):
        '''
        Returns the mean of the particles
        '''
        return np.mean([p.state for p in self.samples], axis=0)
    
    def covariance(self):
        '''
        Returns the covariance of the particles
        '''
        return np.cov([p.state for p in self.samples], rowvar=False)

    def gen_samples_from_measurement(self, cam_pose: tuple[np.ndarray, Rotation], measurement: BoundingBox, num_samples: int):
        '''
        Generates `num_samples` particles that are likely to correspond to the given measurement
        '''
        samples = []
        for _ in range(num_samples):
            radius = np.random.uniform(0.2, 0.4)
            samples.append(Particle(self._generate_initial_state(cam_pose, measurement, radius), 1.0))
        return samples

    def _generate_initial_state(self, cam_pose, box: BoundingBox, initial_radius_guess) -> np.ndarray:
        '''
        Returns a state that corresponds to the given cam pose, bounding box, and radius guess

        The state is a 7 element array with the following elements:
        [x,y,z, vx,vy,vz, radius]
        '''

        box_center_ray = np.array([box.x - self.resolution[0]//2, box.y - self.resolution[1]//2, self.focal_len_pixels])
        camera_look_vector = cam_pose[1].apply(box_center_ray)
        camera_look_vector = 0.1 * camera_look_vector / np.linalg.norm(camera_look_vector)
        box_area = box.width * box.height

        # This could probably be done analytically but binary search was easier
        low = 1
        high = 1000
        while low < high:
            distance = (low + high) / 2
            x_guess = cam_pose[0] + distance * camera_look_vector
            projected_box = self.compute_measurement(cam_pose, np.hstack([x_guess, np.array([0,0,0,initial_radius_guess])]))
            projected_box_area = projected_box.width * projected_box.height
            if abs(projected_box_area - box_area) < 1:
                break
            elif projected_box_area > box_area: # if box is too big we need to move further (increase lower bound on distance)
                low = distance
            else:
                high = distance

        initial_velocity_xz = np.random.uniform(-10, 10, 2)
        initial_velocity_y = np.random.uniform(-1, 1, 1)

        return np.hstack([
            x_guess, 
            initial_velocity_xz[0], 
            initial_velocity_y,
            initial_velocity_xz[1],
            np.array([initial_radius_guess])]
        )

    def compute_measurement(self, cam_pose: tuple[np.ndarray, Rotation], state: np.ndarray) -> BoundingBox:
        '''
        `state` is the state of the track, which is a 7 element array
        '''
        cam = CameraModel(self.focal_len_pixels, 
                        [self.resolution[0]/2, self.resolution[1]/2], 
                        cam_pose[1].as_matrix(), 
                        cam_pose[0].reshape(3,1))

        state_position = state[:3]
        state_radius = state[-1]

        ray_to_center = state_position - cam_pose[0] 
        
        # monte carlo to find the circumscribed rectangle around the sphere's projection into the camera
        # there's probably a better way to do this but I'm not sure what it is
        # I tried a method where we project 4 points on the boundary and fit a 2d ellipse to their projection
        # but the ellipse fitting was not working well
        n_samples = 100

        # sample points on the sphere
        random_vector = np.random.randn(3, n_samples)
        random_vector -= np.dot(random_vector.T, ray_to_center) * np.repeat([ray_to_center / np.linalg.norm(ray_to_center)], n_samples, axis=0).T
        random_vector = random_vector / np.linalg.norm(random_vector, axis=0) * state_radius

        # project points into the camera
        projected_points = cam.project(state_position.reshape((3,1)) + random_vector)
        x_min = np.min(projected_points[0])
        x_max = np.max(projected_points[0])
        y_min = np.min(projected_points[1])
        y_max = np.max(projected_points[1])

        return BoundingBox((x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min)

    def update(self, cam_pose: tuple[np.ndarray, Rotation], measurement:Measurement):
        '''
        measurements is a list of 2D integer bounding boxes in pixel coordinates (x1,y1,x2,y2)
        '''

        # add particles to `samples` that would line up with the measurements
        # self.samples.extend(
        #     self.gen_samples_from_measurement(cam_pose, measurement.box, 10)
        #     )

        for i, particle in enumerate(self.samples):

            # compute the likelihood of the particle given the measurement
            # by comparing the measurement to the particle's predicted
            # measurement
            predicted_measurement = self.compute_measurement(cam_pose, particle.state).to_xyxy()
            particle.likelihood = self.compute_likelihood(predicted_measurement, measurement.box.to_xyxy())

        # resample the particles
        self.resample()

    def compute_likelihood(self, predicted_measurement: np.ndarray, measurement: np.ndarray) -> float:
        '''
        Computes the likelihood of the predicted measurement given the actual measurement
        '''
        assert len(predicted_measurement) == 4
        assert len(measurement) == 4
        # compute the intersection over union of the two boxes
        iou = box_iou(torch.tensor(predicted_measurement.reshape([1,4])), torch.tensor(measurement.reshape([1,4])))
        return iou.item()

    def resample(self):
        '''
        Resamples the particles based on the likelihoods
        '''

        likelihoods = np.array([p.likelihood for p in self.samples])
        likelihoods += self.missed_detection_weight
        # normalize the likelihoods
        likelihoods /= np.sum(likelihoods)
        # resample the particles
        new_samples = []
        for _ in range(self.num_samples):
            index = np.random.choice(len(self.samples), p=likelihoods)
            new_samples.append(self.samples[index].copy())
        self.samples = new_samples
        
        
    def predict(self, dt: float):
        for particle in self.samples:
            particle.step(dt, self.pos_noise_std, self.vel_noise_std, self.radius_noise_std)

    def visualize(self) -> Figure:
        fig = Figure()
        ax = fig.add_subplot(111)
        x_min = np.min([p.state[0]-p.state[-1] for p in self.samples])
        x_max = np.max([p.state[0]+p.state[-1] for p in self.samples])
        z_min = np.min([p.state[2]-p.state[-1] for p in self.samples])
        z_max = np.max([p.state[2]+p.state[-1] for p in self.samples])

        for particle in self.samples:
            ax.add_patch(patches.Circle((particle.state[0], particle.state[2]), 0.1, fill=True, alpha = 0.5, color='blue'))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        # ax.set_xlim(-10, 10)
        # ax.set_ylim(-10, 10)
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        return fig