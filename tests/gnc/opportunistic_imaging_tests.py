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
    def setUp(self):
        '''
        Initialize drop zone boundary and home position variables.
        '''
        self.bounds = read_gps("uavf_2024/gnc/data/AIRDROP_BOUNDARY")
        self.home_lat = 38.31633
        self.home_lon = -76.55578

    def plot_dropzone_imaging_wps(self, target_x, target_y, current_x, current_y):
        '''
        Generate PNG image that plots the drop zone boundary, the drone's current position,
        the position of the target (where the payload must be dropped), and the generated 
        opportunistic imaging waypoints.
        '''
        commander = MockCommander(self.home_lat, self.home_lon, self.bounds)
        planner = DropzonePlanner(commander, 12, 9)

        plt.clf()

        dropzone_x_coords = [p[0] for p in commander.dropzone_bounds_mlocal]
        dropzone_y_coords = [p[1] for p in commander.dropzone_bounds_mlocal]
        plt.scatter(dropzone_x_coords, dropzone_y_coords, c = 'blue', label = 'Drop Zone Boundary')
        for i in range(len(dropzone_x_coords)-1):
            plt.plot([dropzone_x_coords[i], dropzone_x_coords[i+1]], [dropzone_y_coords[i], dropzone_y_coords[i+1]], color='blue')
        plt.plot([dropzone_x_coords[-1], dropzone_x_coords[0]], [dropzone_y_coords[-1], dropzone_y_coords[0]], color='blue')
        
        wps = planner.generate_wps_to_target(target_x, target_y, current_x, current_y)
        plt.scatter([p[0] for p in wps], [p[1] for p in wps], c = 'red', label = 'Generated Imaging Waypoints')
        plt.scatter([wps[0][0]], [wps[0][1]], c = 'purple', label = 'Current Position')
        plt.scatter([wps[-1][0]], [wps[-1][1]], c = 'green', label = 'Target Position')

        plt.legend()
        plt.savefig(f"test/dropzone_imaging_wps_{target_x}_{target_y}.png")

    def generate_test_one_image(self):
        target_x, target_y = 940, -206
        current_x, current_y = 1000, -211
        self.plot_dropzone_imaging_wps(target_x, target_y, current_x, current_y)

    def generate_test_two_image(self):
        target_x, target_y = 1020, -220
        current_x, current_y = 950, -200
        self.plot_dropzone_imaging_wps(target_x, target_y, current_x, current_y)
