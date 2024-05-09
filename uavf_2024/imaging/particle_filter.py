import numpy as np
from scipy.spatial.transform import Rotation
from uavf_2024.imaging.camera_model import CameraModel
from uavf_2024.imaging.imaging_types import BoundingBox
from uavf_2024.imaging.utils import make_ortho_vectors
from torchvision.ops import box_iou
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class Measurement:
    pose: tuple[np.ndarray, Rotation]
    box: BoundingBox

class ParticleFilter:
    def __init__(self, 
                 initial_measurement: Measurement, 
                 resolution: tuple[int,int], 
                 focal_len_pixels: float, 
                 num_particles: int = 1000, 
                 missed_detection_weight: float = 1e-4, 
                 pos_noise_std: float = 0.1, 
                 vel_noise_std: float = 0.1, 
                 radius_noise_std: float = 0
        ):

        '''
        self.samples is [num_particles, 7]
        where the 7 elements are [x,y,z, vx,vy,vz, radius]
        '''
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resolution = resolution
        self.focal_len_pixels = focal_len_pixels
        self.num_samples = num_particles 
        self.missed_detection_weight = missed_detection_weight # likelihood to give a particle if it's not detected
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.radius_noise_std = radius_noise_std

        self.samples: Tensor = self.gen_samples_from_measurement(initial_measurement.pose, initial_measurement.box, num_particles)

    def mean(self):
        '''
        Returns the mean of the particles
        '''
        return self.samples.mean(dim=0).cpu()
    
    def covariance(self):
        '''
        Returns the covariance of the particles
        '''
        return self.samples.T.cov().cpu()

    def gen_samples_from_measurement(self, cam_pose: tuple[np.ndarray, Rotation], measurement: BoundingBox, num_samples: int):
        '''
        Generates `num_samples` particles that are likely to correspond to the given measurement
        '''
        radii = torch.rand(num_samples) * (2-0.1) + 0.1
        samples = torch.vstack([
            self._generate_initial_state(cam_pose, measurement, r)
            for r in radii
        ])
        return samples.to(self._device)

    def _generate_initial_state(self, cam_pose, box: BoundingBox, initial_radius_guess) -> Tensor:
        '''
        Returns a state that corresponds to the given cam pose, bounding box, and radius guess

        The state is a 7 element array with the following elements:
        [x,y,z, vx,vy,vz, radius]
        '''

        box_center_ray = np.array([box.x - self.resolution[0]//2, box.y - self.resolution[1]//2, self.focal_len_pixels])
        camera_look_vector = Tensor(cam_pose[1].apply(box_center_ray))
        camera_look_vector = 0.1 * camera_look_vector / torch.linalg.norm(camera_look_vector)
        box_area = box.width * box.height

        # This could probably be done analytically but binary search was easier
        low = 1
        high = 1000
        while low < high:
            distance = (low + high) / 2
            x_guess = Tensor(cam_pose[0]) + distance * camera_look_vector
            projected_box = self.compute_measurements(
                cam_pose, 
                torch.hstack([
                    x_guess, 
                    Tensor([0,0,0,initial_radius_guess])
                ])[np.newaxis,:].to(self._device)
            )[0]
            width = projected_box[2] - projected_box[0]
            height = projected_box[3] - projected_box[1]
            projected_box_area = width*height
            if abs(projected_box_area - box_area) < 1:
                break
            elif projected_box_area > box_area: # if box is too big we need to move further (increase lower bound on distance)
                low = distance
            else:
                high = distance

        horizontal_velocity = torch.rand(1)*10
        vertical_velocity = torch.rand(1)*2-1
        angle = torch.rand(1)*2*torch.pi

        return torch.hstack([
            x_guess, 
            horizontal_velocity * torch.cos(angle),
            vertical_velocity,
            horizontal_velocity * torch.sin(angle),
            Tensor([initial_radius_guess])]
        )

    def compute_measurements(self, cam_pose: tuple[np.ndarray, Rotation], states: np.ndarray) -> BoundingBox:
        '''
        `states` is (n, 7)
        returns ndarray of shape (n, 4) where the 4 elements are [x1,y1,x2,y2]
        '''
        cam = CameraModel(self.focal_len_pixels, 
                        [self.resolution[0]/2, self.resolution[1]/2], 
                        cam_pose[1].as_matrix(), 
                        cam_pose[0].reshape(3,1))

        n = states.shape[0]
        positions = states[:, :3]
        radii = states[:, -1]

        cam_position_tensor = Tensor(cam_pose[0]).to(self._device)
        rays_to_center = positions - cam_position_tensor
        
        n_samples = 25
        orthogonal_rays_normalized = make_ortho_vectors(rays_to_center, n_samples)

        pts3 = positions[:, None, :] + orthogonal_rays_normalized * radii[:,None,None]

        pts3_flat = pts3.reshape(-1, 3) # (n*n_samples, 3)

        # project points into the camera
        projected_points = cam.project(pts3_flat.T, self._device).T.reshape(n, n_samples, 2)
        x_mins = torch.min(projected_points[:,:,0], dim=1).values # (n,)
        y_mins = torch.min(projected_points[:,:,1], dim=1).values # (n,)
        x_maxs = torch.max(projected_points[:,:,0], dim=1).values # (n,)
        y_maxs = torch.max(projected_points[:,:,1], dim=1).values # (n,)

        return torch.vstack([x_mins, y_mins, x_maxs, y_maxs]).T # (n, 4)

    def update(self, cam_pose: tuple[np.ndarray, Rotation], measurement:Measurement):
        '''
        measurements is a list of 2D integer bounding boxes in pixel coordinates (x1,y1,x2,y2)
        '''

        # add particles to `samples` that would line up with the measurements
        # self.samples.extend(
        #     self.gen_samples_from_measurement(cam_pose, measurement.box, 10)
        #     )

        measurements = self.compute_measurements(cam_pose, self.samples)
        likelihoods = self.compute_likelihoods(measurements, measurement.box.to_xyxy())

        # resample the particles
        self.resample(likelihoods)

    def compute_likelihoods(self, predicted_measurements: np.ndarray, measurement: np.ndarray) -> Tensor:
        '''
        `predicted_measurements` is (n, 4)
        `measurement` is (4,)
        Computes the likelihood of the predicted measurement given the actual measurement
        Returns (n,) tensor
        '''
        n = predicted_measurements.shape[0]
        # compute the intersection over union of the two boxes
        ious = box_iou(
            Tensor(predicted_measurements),
            Tensor(measurement).unsqueeze(0).to(self._device)
        )
        return ious[:,0]

    def resample(self, likelihoods: Tensor):
        '''
        Resamples the particles based on the likelihoods
        `likelihoods` is a 1D tensor of length `num_samples`
        '''

        likelihoods
        likelihoods += self.missed_detection_weight
        likelihoods = likelihoods
        # normalize the likelihoods
        # resample the particles
        indices = torch.multinomial(likelihoods, self.num_samples, replacement=True)
        new_samples = self.samples[indices].clone().detach()
        self.samples = new_samples
        
        
    def predict(self, dt: float):
        # move the particles according to their velocities
        self.samples[:, :3] += self.samples[:, 3:6] * dt

        self.samples[:, :3] += torch.normal(
            mean = torch.zeros(self.num_samples, 3),
            std = self.pos_noise_std*torch.ones(self.num_samples, 3)
        ).to(self._device)

        self.samples[:, 3:6] += torch.normal(
            mean = torch.zeros(self.num_samples, 3),
            std = self.vel_noise_std*torch.ones(self.num_samples, 3)
        ).to(self._device)

        self.samples[:, 6] += torch.normal(
            mean = torch.zeros(self.num_samples),
            std = self.radius_noise_std*torch.ones(self.num_samples)
        ).to(self._device)

    
    def visualize(self, fig_bounds = None) -> Figure:
        fig = Figure()
        ax = fig.add_subplot(111)

        for particle in self.samples:
            ax.add_patch(patches.Circle((particle[0], particle[2]), 0.1, fill=True, alpha = 0.5, color='blue'))

        if fig_bounds is None:
            x_min = np.min([p[0]-p[-1] for p in self.samples])
            x_max = np.max([p[0]+p[-1] for p in self.samples])
            z_min = np.min([p[2]-p[-1] for p in self.samples])
            z_max = np.max([p[2]+p[-1] for p in self.samples])
            abs_max = max(map(abs, [x_min, x_max, z_min, z_max]))
            ax.set_xlim(-abs_max, abs_max)
            ax.set_ylim(-abs_max, abs_max)
        else:
            ax.set_xlim(-fig_bounds, fig_bounds)
            ax.set_ylim(-fig_bounds, fig_bounds)
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        return fig
