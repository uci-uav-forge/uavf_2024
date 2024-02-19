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
    def __init__(self, state: np.ndarray, likelihood: float):
        self.state = state
        self.likelihood = likelihood 

    @staticmethod
    def from_random(
        horizontal_position_range: float = 100,
        vertical_position_range: float = 30,
        horizontal_velocity_range: float = 10,
        vertical_velocity_range: float = 5,
        radius_range: float = 1
    ):
        return Particle([
            np.random.uniform(-horizontal_position_range, horizontal_position_range),
            np.random.uniform(0, vertical_position_range),
            np.random.uniform(-horizontal_position_range, horizontal_position_range),
            np.random.uniform(-horizontal_velocity_range, horizontal_velocity_range),
            np.random.uniform(-vertical_velocity_range, vertical_velocity_range),
            np.random.uniform(-horizontal_velocity_range, horizontal_velocity_range),
            np.random.uniform(0.1, radius_range)
        ], 1.0)

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
                offset_point = pos_3d + offset*np.eye(3)[axis]
                new_coords = localizer.coords_to_2d(offset_point, cam_pose)
                min_x = min(min_x, new_coords[0])
                min_y = min(min_y, new_coords[1])
                max_x = max(max_x, new_coords[0])
                max_y = max(max_y, new_coords[1])
        return (min_x, min_y, max_x, max_y)

    def step(self, dt: float, pos_noise_std: float = 1, vel_noise_std: float = 0.1, radius_noise_std: float = 0.01):
        '''
        Updates the state of the particle, adding noise to the position, velocity, and radius
        '''
        self.state[3:6] += np.random.randn(3)* vel_noise_std
        self.state[0:3] += dt*self.state[3:6] + np.random.randn(3) * pos_noise_std
        self.state[6] += np.random.randn(1)*radius_noise_std

    

class ParticleFilter:
    def __init__(self, num_particles: int, missed_detection_weight: float = 1e-4, pos_noise_std: float = 1, vel_noise_std: float = 0.1, radius_noise_std: float = 0.01):
        self.samples: list[Particle] = [Particle.from_random() for _ in range(num_particles)]

        # not the same as len(self.samples) because we might add samples during `update` but then reduce during `resample`
        self.num_samples = num_particles 
        self.missed_detection_weight = missed_detection_weight # likelihood to give a particle if it's not detected
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.radius_noise_std = radius_noise_std

    def update(self, cam_pose: tuple[np.ndarray, Rotation], fov_deg: float, resolution: tuple[int,int], measurements: list[np.ndarray]):
        '''
        
        measurements is a list of 2D integer bounding boxes in pixel coordinates (x1,y1,x2,y2)
        '''

        cam_position, cam_rot = cam_pose

        w,h = resolution
        focal_len = w/(2*np.tan(np.deg2rad(fov_deg/2)))
        # add particles to `samples` that would line up with the measurements
        for measurement in measurements:
            # create a particle that would line up with the measurement analytically
            x1, y1, x2, y2 = measurement
            x, y = (x1+x2)/2, (y1+y2)/2
            w, h = x2-x1, y2-y1
            bbox_r = (w+h)/2
            initial_direction_vector = np.array([x-w/2,h/2-y,-focal_len])

            # rotate the vector to match the camera's rotation
            rotated_vector = cam_rot.as_matrix() @ initial_direction_vector
            # normalize to unit length
            rotated_vector /= np.linalg.norm(rotated_vector)

            for radius in np.linspace(0.1,1,10):
                # solve for distance to camera
                d = radius * focal_len / bbox_r
                pos = cam_position + d*rotated_vector
                self.samples.append(Particle([
                    pos[0], pos[1], pos[2],
                    0,0,0,
                    radius
                ], 1.0))

        for i, particle in enumerate(self.samples):
            # compute the likelihood of the particle given the measurements
            for measurement in measurements:
                # compute the likelihood of the particle given the measurement
                # by comparing the measurement to the particle's predicted
                # measurement
                predicted_measurement = particle.measure(cam_pose, fov_deg, resolution)
                particle.likelihood = self.compute_likelihood(predicted_measurement, measurement)

        # resample the particles
        self.resample()

    def compute_likelihood(self, predicted_measurement: tuple[int,int,int,int], measurement: np.ndarray) -> float:
        '''
        Computes the likelihood of the predicted measurement given the actual measurement
        '''
        # compute the intersection over union of the two boxes
        iou = box_iou(torch.tensor([predicted_measurement]), torch.tensor([measurement]))
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
            new_samples.append(self.samples[index])
        self.samples = new_samples
        
        
    def predict(self, dt: float):
        for particle in self.samples:
            particle.step(dt, self.pos_noise_std, self.vel_noise_std, self.radius_noise_std)

    def visualize(self, debug_canvas: np.ndarray):
        raise NotImplementedError("This method is not yet implemented") 
