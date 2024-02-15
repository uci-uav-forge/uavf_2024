import numpy as np
from scipy.spatial.transform import Rotation
from uavf_2024.imaging import Localizer

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
        pass