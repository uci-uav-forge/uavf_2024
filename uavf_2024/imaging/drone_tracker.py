from uavf_2024.imaging.imaging_types import BoundingBox
from uavf_2024.imaging.camera_model import CameraModel
from uavf_2024.imaging.particle_filter import ParticleFilter
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
import numpy as np
from torchvision.ops import box_iou
from torch import Tensor
from dataclasses import dataclass

@dataclass
class Measurement:
    pose: tuple[np.ndarray, Rotation]
    box: BoundingBox

class DroneTracker:
    resolution: tuple[int,int]
    focal_len_pixels: float
    def __init__(self, resolution: tuple[int,int], focal_len_pixels: float):
        self.tracks: list[self.Track] = []
        DroneTracker.resolution = resolution
        DroneTracker.focal_len_pixels = focal_len_pixels

    def predict(self, dt: float):
        for track in self.tracks:
            track.predict(dt)

    def update(self, cam_pose: tuple[np.ndarray, Rotation], measurements: list[BoundingBox]):
        # construct matrix of IOU similarity between measurements and tracks
        # iou_matrix[i,j] is the iou between track i and measurement j

        if len(self.tracks) == 0:
            self.tracks.append(self.Track(Measurement(cam_pose, measurements[0])))
        else:
            self.tracks[0].update(Measurement(cam_pose, measurements[0]))
        return
        # TODO: right now we don't take into account the variance of the drone's position/radius when calculating the IOU
        # between the track and the measurement. This means if we don't get a lucky initial guess for the drone's position or
        # if it's moving fast, the track will never match the measurement and it won't be updated.
        # maybe we can do something like un-projecting the bounding box into a cone and then taking the 3d intersection of
        # that and the track's bounding sphere with variance
        iou_matrix = np.zeros((len(self.tracks), len(measurements)))
        for i, track in enumerate(self.tracks):
            for j, measurement in enumerate(measurements):
                iou_matrix[i,j] = self._iou(track, measurement, cam_pose)

        # update tracks with measurements if iou > 0.5
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i,j] > 0.5:
                self.tracks[i].update(measurements[j])
            else:
                self.tracks.append(self.Track(Measurement(cam_pose, measurements[j])))

        # create new tracks for unmatched measurements
        for j in range(len(measurements)):
            if j not in col_ind:
                self.tracks.append(self.Track(Measurement(cam_pose, measurements[j])))

        # prune tracks that don't have 50% seen to alive ratio
        self.tracks = [
            track for track in self.tracks 
            if  track.frames_alive == 0 
                or track.frames_seen/track.frames_alive > 0.5
        ]

    def _iou(self, track, box: BoundingBox, cam_pose: tuple[np.ndarray,Rotation]) -> float:
        '''
        `track` is a DroneTracker.Track object but the type annotation doesn't work
        '''
        track_box = track.simulate_measurement(cam_pose)
        track_box_arr = Tensor([track_box.x - track_box.width//2, track_box.y - track_box.height//2, track_box.x + track_box.width//2, track_box.y + track_box.height//2])
        box_arr = Tensor([box.x - box.width//2, box.y - box.height//2, box.x + box.width//2, box.y + box.height//2])
        return box_iou(track_box_arr.unsqueeze(0), box_arr.unsqueeze(0)).item()
        

    '''
    Nested on purpose to be able to access self.resolution and self.focal_len_pixels
    '''
    class Track:
        def __init__(self, initial_measurement: Measurement):
            # x dimension is [x,y,z, vx,vy,vz, radius]
            self.filter = ParticleFilter(initial_measurement, DroneTracker.resolution, DroneTracker.focal_len_pixels)
            
            self.frames_alive = 0
            self.frames_seen = 0

        def predict(self, dt: float):
            self.frames_alive += 1
            self.filter.predict(dt)

        def update(self, measurement: Measurement):
            self.frames_seen += 1
            self.filter.update(measurement.pose, measurement)

