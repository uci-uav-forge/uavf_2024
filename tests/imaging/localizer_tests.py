import unittest
from uavf_2024.imaging.localizer import Localizer
import numpy as np

class TestLocalizer(unittest.TestCase):
    def setUp(self):
        self.localizer = Localizer()
    def test_with_sim_dataset(self):