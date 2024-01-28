from __future__ import annotations
from .imaging_types import Target3D, TargetDescription
from .utils import calc_match_score
from dataclasses import dataclass
import numpy as np

class Track:
    '''
    A Track is a series of measurements of the same target. It is used to keep track of the same target

    Initialization requires at least one measurement
    '''
    def __init__(self, measurements: list[Target3D]):
        assert len(measurements) > 0
        self._measurements = measurements
        self._recalculate_averages()

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

    def _recalculate_averages(self) -> None:
        self._position = np.mean([measurement.position for measurement in self._measurements], axis=0)
        self._descriptor = np.mean([measurement.description for measurement in self._measurements], axis=0)

    def add_measurement(self, measurement: Target3D) -> None:
        self._measurements.append(measurement)
        self._recalculate_averages()

    def get_measurements(self) -> list[Target3D]:
        return self._measurements

class TargetTracker:
    def __init__(self, search_candidates: list[TargetDescription], debug_path: str  = None):
        self.search_candidates = search_candidates
        self.tracks: list[Track] = []
        self.debug_path = debug_path

    def update(self, detections: list[Target3D]):
        for detection in detections:
            if len(self.tracks) == 0:
                self.tracks.append(Track([detection]))
                continue

            closest_track = min(self.tracks, key=lambda track: np.linalg.norm(track.position - detection.position))

            # if the track is close enough, add the detection to the track
            if np.linalg.norm(closest_track.position - detection.position) < 3:
                closest_track.add_measurement(detection)
            # otherwise, create a new track
            else:
                self.tracks.append(Track([detection]))

    def estimate_positions(self) -> list[Track]:
        '''
        Returns closest track in descriptor space for each search candidate
        '''
        closest_tracks = [
            max(self.tracks, key=lambda track: calc_match_score(track.descriptor, candidate))
            for candidate in self.search_candidates
        ]

        return closest_tracks