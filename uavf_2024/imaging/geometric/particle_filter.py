import numpy as np
from scipy.spatial.transform import Rotation
from uavf_2024.imaging import Localizer
from torchvision.ops import box_iou
import torch

# Note: this implementation is slow because it's object-oriented, but it's easier to understand
class Particle:
    '''
    State is [x,y,z,dx,dy,dz,radius] for a bounding sphere
    '''
    def __init__(self):
        self.state = np.random.randn(7,1)

    def measure(self, cam_pose: tuple[np.ndarray, Rotation], fov: float, resolution: tuple[int,int]) -> tuple[int,int,int,int]:
        '''
        returns a [x1,y1,x2,y2] bounding box in pixel coordinates of
        how the particle would appear in the camera. Since we're using
        spherical models, the bounding box is the smallest box that
        fully contains the sphere
        '''
        localizer = Localizer(fov, resolution)
        pos_3d = self.state[:3]
        coords_2d = localizer.coords_to_2d(pos_3d, cam_pose)
        # randomly sample around edge of sphere in each direction to get the bounding box
        min_x, min_y = coords_2d
        max_x, max_y = coords_2d
        r = self.state[6]
        for offset in [-r, r]:
            for axis in range(3):
                new_coords = localizer.coords_to_2d(pos_3d + offset*np.eye(3)[axis], cam_pose)
                min_x = min(min_x, new_coords[0])
                min_y = min(min_y, new_coords[1])
                max_x = max(max_x, new_coords[0])
                max_y = max(max_y, new_coords[1])
        return (min_x, min_y, max_x, max_y)

    def step(self, dt: float):
        '''
        Updates the state of the particle
        '''
        self.state[0:3] += self.state[3:6]*dt

    

class ParticleFilter:
    def __init__(self):
        self.samples: list[Particle] = []
        self.false_positive_rate = 0.01
        self.false_negative_rate = 0.01


    def update(self, cam_pose: tuple[np.ndarray, Rotation], fov: float, resolution: tuple[int,int], measurements: list[np.ndarray]):
        '''
        
        measurements is a list of 2D integer bounding boxes in pixel coordinates (x1,y1,x2,y2)
        '''
        likelihoods = np.ones(len(self.samples))
        for i, particle in enumerate(self.samples):
            # compute the likelihood of the particle given the measurements
            for measurement in measurements:
                # compute the likelihood of the particle given the measurement
                # by comparing the measurement to the particle's predicted
                # measurement
                predicted_measurement = particle.measure(cam_pose, fov, resolution)
                likelihoods[i] *= self.compute_likelihood(predicted_measurement, measurement)

        # resample the particles
        self.resample(likelihoods)

    def compute_likelihood(self, predicted_measurement: tuple[int,int,int,int], measurement: np.ndarray) -> float:
        '''
        Computes the likelihood of the predicted measurement given the actual measurement
        '''
        # compute the intersection over union of the two boxes
        iou = box_iou(torch.tensor([predicted_measurement]), torch.tensor([measurement]))
        return iou.item()

    def resample(self, likelihoods: np.ndarray):
        '''
        Resamples the particles based on the likelihoods
        '''
        # normalize the likelihoods
        likelihoods /= np.sum(likelihoods)
        # resample the particles
        new_samples = []
        for _ in range(len(self.samples)):
            index = np.random.choice(len(self.samples), p=likelihoods)
            new_samples.append(self.samples[index])
        self.samples = new_samples
        
        
    def predict(self, dt: float):
        for particle in self.samples:
            particle.step(dt)