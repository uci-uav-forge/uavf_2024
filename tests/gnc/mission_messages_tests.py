import unittest
from uavf_2024.gnc.mission_messages import *


class TestDropzonePlanner(unittest.TestCase):
    def test_conversions(self):
        x = BumpLap(5)
        assert BumpLap.from_string(BumpLap.to_string(x)) == x
        assert BumpLap.from_string("random garbage") == None

        y = RequestPayload("chartreuse", "vantablack", "Q", "pentagon")
        assert RequestPayload.from_string(RequestPayload.to_string(y)) == y
        assert RequestPayload.from_string("garbage") == None