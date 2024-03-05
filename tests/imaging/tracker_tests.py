from uavf_2024.imaging.tracker import TargetTracker
from uavf_2024.imaging.imaging_types import Target3D, CertainTargetDescriptor
import numpy as np
import unittest


class TargetTrackerTests(unittest.TestCase):
    def test_puts_identical_targets_same_track(self):
        bogus_descriptor = CertainTargetDescriptor(
                'red',
                'circle',
                'blue',
                'A'
            ).as_probabilistic()
        target = Target3D(
            position = np.array([0,0,0]),
            descriptor = bogus_descriptor
        )
        tracker = TargetTracker([])

        tracker.update([target])

        assert len(tracker.tracks) == 1

        tracker.update([target])

        assert len(tracker.tracks) == 1
        assert len(tracker.tracks[0].get_measurements()) == 2