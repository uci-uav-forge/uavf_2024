from uavf_2024.imaging.tracker import TargetTracker
from uavf_2024.imaging.imaging_types import Target3D, TargetDescription
import numpy as np
import unittest


class TargetTrackerTests(unittest.TestCase):
    def test_puts_identical_targets_same_track(self):
        bogus_descriptor = TargetDescription(
                np.eye(9)[0],
                np.eye(36)[0],
                np.eye(8)[0],
                np.eye(8)[0],
            )
        target = Target3D(
            position = np.array([0,0,0]),
            description = bogus_descriptor
        )
        tracker = TargetTracker([])

        tracker.update([target])

        assert len(tracker.tracks) == 1

        tracker.update([target])

        assert len(tracker.tracks) == 1
        assert len(tracker.tracks[0].get_measurements()) == 2