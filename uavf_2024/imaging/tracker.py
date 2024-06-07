from __future__ import annotations
from .imaging_types import Target3D, CertainTargetDescriptor
from .utils import calc_match_score
from scipy.special import kl_div
from scipy.optimize import linear_sum_assignment
from itertools import product
import numpy as np

class Track:
    '''
    A Track is a series of measurements of the same target. It is used to keep track of the same target

    Initialization requires at least one measurement
    '''
    def __init__(self, measurements: list[Target3D], id: int):
        assert len(measurements) > 0
        self._measurements = measurements
        self._recalculate_averages()
        self.id = id

    @property
    def position(self):
        '''
        This is the position of the track, calculated by averaging the positions of all the measurements. 
        It is only recalculated when a new measurement is added.
        '''
        return self._position

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def hist(self):
        return self._hist

    def _recalculate_averages(self) -> None:
        self._position = np.mean([measurement.position for measurement in self._measurements], axis=0)
        self._descriptor = np.mean([measurement.descriptor for measurement in self._measurements], axis=0)
        self._hist = np.mean([measurement.hist for measurement in self._measurements], axis=0)

    def add_measurement(self, measurement: Target3D) -> None:
        self._measurements.append(measurement)
        self._recalculate_averages()

    def get_measurements(self) -> list[Target3D]:
        return self._measurements

    def __repr__(self):
        return f"Track {self.id}: {self.position},{self.descriptor}"

    def contributing_measurement_ids(self):
        return [m.id for m in self._measurements]

class TargetTracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self.num_tracks = 0
    
    def _similarity(self, track: Track, detection: Target3D):
        total=0
        for hist_c_1, hist_c_2 in zip(track.hist, detection.hist):
            hist_c_1 /= max(hist_c_1)*256
            hist_c_2 /= max(hist_c_2)*256
            total += np.dot(hist_c_1, hist_c_2)
            print(total)
        return total

    def update(self, detections: list[Target3D]):
        '''
        Assumes that `detections` is all the detections for a single image.
        '''
        # ideas:
        # 1. enforce constraint that each image can't contribute more than 1 detection to a given track
        # 2. use color histograms of crops to match tracks
        tracks_to_be_added = []
        for detection in detections:
            if len(self.tracks) == 0:
                tracks_to_be_added.append(Track([detection], self.num_tracks))
                self.num_tracks+=1
                continue

            closest_track = max(self.tracks, 
                key=lambda track: 
                self._similarity(track, detection)
            )

            # if the track is close enough, add the detection to the track
            if self._similarity(closest_track, detection) > 1e-4:
                closest_track.add_measurement(detection)
            # otherwise, create a new track
            else:
                tracks_to_be_added.append(Track([detection], self.num_tracks))
                self.num_tracks+=1
        self.tracks.extend(tracks_to_be_added)

    def estimate_positions(self, search_candidates: list[CertainTargetDescriptor]) -> list[Track]:
        '''
        Returns closest track in descriptor space for each search candidate
        '''

        
        cost_matrix = np.empty((len(search_candidates), len(self.tracks)))
        for i,j in product(range(len(search_candidates)), range(len(self.tracks))):
            cost_matrix[i,j] = calc_match_score(self.tracks[j].descriptor, search_candidates[i].as_probabilistic())
        #row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        
        col_ind = [
            np.argmax(cost_matrix[i]) for i in range(len(search_candidates))
        ]

        closest_tracks = [
            self.tracks[i] for i in col_ind
        ]

        return closest_tracks

    def confidence_score(self, candidate: CertainTargetDescriptor) -> float:
        return max(calc_match_score(track.descriptor, candidate.as_probabilistic()) for track in self.tracks)