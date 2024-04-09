from uavf_2024.imaging.imaging_types import BoundingBox
from scipy.spatial.transform import Rotation
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
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
        iou_matrix = np.zeros((len(self.tracks), len(measurements)))
        for i, track in enumerate(self.tracks):
            for j, measurement in enumerate(measurements):
                iou_matrix[i,j] = self._iou(cam_pose, track, measurement)

    def _iou(self, track, measurement: BoundingBox) -> float:
        '''
        `track` is a DroneTracker.Track object but the type annotation doesn't work
        '''
        raise NotImplementedError
        

    '''
    Nested on purpose to be able to access self.resolution and self.focal_len_pixels
    '''
    class Track:
        def __init__(self, initial_measurement: Measurement):
            # x dimension is [x,y,z, vx,vy,vz, radius]
            self.kf = UnscentedKalmanFilter(
                dim_x=7,
                dim_z=4,
                hx=self._measurement_fn,
                fx=self._state_transition,
                points = MerweScaledSigmaPoints(7, 1e-3, 2, 0)
            )

            self.cam_pose = initial_measurement.pose


        def _measurement_fn(self, x: np.ndarray) -> np.ndarray:
            '''
            Returns measurement of the state from self.cam_pose. 
            self.cam_pose needs to be set before calling this function

            The returned ndarray is of shape (7,) with the following elements:
            [x,y,z, vx,vy,vz, radius]
            ''' 
            pass
        
        def simulate_measurement(self, cam_pose: tuple[np.ndarray, Rotation]) -> Measurement:
            '''
            Simulates a measurement with the current state of the track
            '''
            pass

        @staticmethod
        def _state_transition(x: np.ndarray, dt: float) -> np.ndarray:
            pass

        def predict(self, dt: float):
            self.kf.predict(dt)

        def update(self, measurement: Measurement):
            self.cam_pose = measurement.pose
            box_x = measurement.box.x
            box_y = measurement.box.y
            box_w = measurement.box.width
            box_h = measurement.box.height
            self.kf.update(np.array([box_x, box_y, box_w, box_h]))
