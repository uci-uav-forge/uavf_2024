import unittest
from uavf_2024.gnc.dropzone_planner import DropzonePlanner
from uavf_2024.gnc.util import *
import matplotlib.pyplot as plt

class MockCommander:
    def __init__(self, lat, lon, bounds):
        self.home_lat = lat
        self.home_lon = lon
        self.dropzone_bounds = bounds
        self.dropzone_bounds_mlocal = [convert_delta_gps_to_local_m((self.home_lat, self.home_lon), x) for x in self.dropzone_bounds]

    def log(self, msg):
        print(msg)

class TestOpportunisticImaging(unittest.TestCase):
    def plot_dropzone_imaging_wps(self, dropzone_filename, start_lat, start_lon, target_x, target_y, current_x, current_y):
        bounds = read_gps(dropzone_filename)
        commander = MockCommander(start_lat, start_lon, bounds)
        planner = DropzonePlanner(commander, 12, 9)

        plt.clf()

        # Dropzone corners are blue
        plt.scatter([p[0] for p in self.dropzone_bounds_mlocal], [p[1] for p in self.dropzone_bounds_mlocal], c = 'blue')
        wps = planner.generate_wps_to_target(target_x, target_y, current_x, current_y)

        # Imaging waypoints that are generated are red
        plt.scatter([p[0] for p in wps], [p[1] for p in wps], c = 'red')

        # Current position is purple
        plt.scatter([wps[0][0]], [wps[0][1]], c = 'purple')

        # Target position is green
        plt.scatter([wps[-1][0]], [wps[-1][1]], c = 'green')

        plt.savefig(f"test/dropzone_imaging_wps_{target_x}_{target_y}.png")

    def test_one(self):
        target_x, target_y = 940, -206
        current_x, current_y = 1000, -211
        self.plot_dropzone_imaging_wps("uavf_2024/gnc/data/AIRDROP_BOUNDARY", 38.31633, -76.55578, target_x, target_y, current_x, current_y)

    def test_two(self):
        target_x, target_y = 1020, -220
        current_x, current_y = 950, -200
        self.plot_dropzone_imaging_wps("uavf_2024/gnc/data/AIRDROP_BOUNDARY", 38.31633, -76.55578, target_x, target_y, current_x, current_y)
