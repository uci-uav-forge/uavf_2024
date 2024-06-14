import unittest
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from uavf_2024.gnc.util import *

class MockCommander:
    def __init__(self, geofence_file):
        '''
        Initialize geofence variables.
        '''
        self.left_intermediate_waypoint_global = (38.31605966, -76.55154921)
        self.right_intermediate_waypoint_global = (38.31542867, -76.54548898)
        self.geofence_middle_pt = (38.31470980862425, -76.54936361414539)
        self.geofence = read_gps(geofence_file)

    def get_closest_intermediate_point(self, destination_wp):
        '''
        Get an intermediate waypoint to fly to before flying to the destination_wp.
        By flying to the intermediate waypoint before the destination_wp, the
        geofence will not be violated.
        '''
        return self.right_intermediate_waypoint_global if destination_wp[1] > self.geofence_middle_pt[1] else self.left_intermediate_waypoint_global

    def generate_legal_waypoints(self, waypoints):
        '''
        Check if the given waypoints produces a flight path that will violate the geofence
        and return a "legal" set of waypoints that ensures the drone will stay within the
        geofence when it travels to each waypoint.
        '''
        legal_waypoints = []
        geofence_bounds = Polygon(self.geofence)

        for i in range(len(waypoints) - 1):
            start_wp = waypoints[i]
            destination_wp = waypoints[i + 1]
            
            legal_waypoints.append(start_wp)

            if start_wp[0] == destination_wp[0] and start_wp[1] == destination_wp[1]:
                continue

            path = LineString([start_wp, destination_wp])
            if not path.within(geofence_bounds):
                legal_waypoints.append(self.get_closest_intermediate_point(destination_wp))
                
        legal_waypoints.append(waypoints[-1])

        return legal_waypoints
    

class TestGenerateWpsInsideGeofence(unittest.TestCase):
    def setUp(self):
        '''
        Initialize common variables used across unit tests.
        '''
        self.commander = MockCommander("uavf_2024/gnc/data/FLIGHT_BOUNDARY")
        self.geofence_boundary = Polygon(self.commander.geofence)
        self.geofence_latitudes = [pt[0] for pt in self.commander.geofence]
        self.geofence_longitudes = [pt[1] for pt in self.commander.geofence]

    def test_left_intermediate_point_is_valid(self):
        '''
        Verify that the path from commander.left_intermediate_waypoint_global
        to any point on the geofence does not cross the geofence.
        '''
        paths_to_geofence_pts = []
        for geofence_pt in self.commander.geofence:
            paths_to_geofence_pts.append(LineString([geofence_pt, self.commander.left_intermediate_waypoint_global]))
        
        for path in paths_to_geofence_pts:
            self.assertTrue(path.within(self.geofence_boundary))

        # Generate PNG image
        plt.clf()
        
        plt.scatter(self.geofence_longitudes, self.geofence_latitudes, color = 'black', label = 'Geofence')
        for i in range(len(self.geofence_latitudes) - 1):
            plt.plot([self.geofence_longitudes[i], self.geofence_longitudes[i + 1]], [self.geofence_latitudes[i], self.geofence_latitudes[i + 1]], color = 'black')
        plt.plot([self.geofence_longitudes[-1], self.geofence_longitudes[0]], [self.geofence_latitudes[-1], self.geofence_latitudes[0]], color = 'black')

        plt.scatter([self.commander.left_intermediate_waypoint_global[1]], [self.commander.left_intermediate_waypoint_global[0]], color = 'blue', label = 'Left Intermediate Point')

        plt.legend()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig("test/left_intermediate_point.png")

    def test_right_intermediate_point_is_valid(self):
        '''
        Verify that the path from commander.right_intermediate_waypoint_global
        to a point on the geofence boundary does not cross the geofence.
        '''
        paths_to_geofence_pts = []
        for geofence_pt in self.commander.geofence:
            paths_to_geofence_pts.append(LineString([geofence_pt, self.commander.right_intermediate_waypoint_global]))
        
        for path in paths_to_geofence_pts:
            self.assertTrue(path.within(self.geofence_boundary))

        # Generate PNG image
        plt.clf()
        
        plt.scatter(self.geofence_longitudes, self.geofence_latitudes, color = 'black', label = 'Geofence')
        for i in range(len(self.geofence_latitudes) - 1):
            plt.plot([self.geofence_longitudes[i], self.geofence_longitudes[i + 1]], [self.geofence_latitudes[i], self.geofence_latitudes[i + 1]], color = 'black')
        plt.plot([self.geofence_longitudes[-1], self.geofence_longitudes[0]], [self.geofence_latitudes[-1], self.geofence_latitudes[0]], color = 'black')

        plt.scatter([self.commander.right_intermediate_waypoint_global[1]], [self.commander.right_intermediate_waypoint_global[0]], color = 'blue', label = 'Right Intermediate Point')

        plt.legend()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig("test/right_intermediate_point.png")

    def test_path_violates_geofence(self):
        '''
        Test a flight path that violates the geofence. Verify that
        generate_legal_waypoints() produces a new set of waypoints 
        that lie inside the geofence.
        '''

        illegal_path_wps = [(38.31633, -76.55578), (38.31813167, -76.55120605), (38.31813136, -76.54606035), (38.31497788, -76.54205873)]
        legal_path_wps = self.commander.generate_legal_waypoints(illegal_path_wps)
        illegal_path = LineString(illegal_path_wps)

        self.assertFalse(illegal_path.within(self.geofence_boundary))
        self.assertTrue(legal_path_wps.within(self.geofence_boundary))

        self.plot_geofence_and_flight_paths('flight_path_violates_geofence', illegal_path_wps, legal_path_wps, ['red', 'green'], ['Original Illegal Flight Path', 'Generated Legal Flight Path'])

    def test_path_complies_with_geofence(self):
        '''
        Test a flight path that does not violate the geofence. Verify
        that generate_legal_waypoints() returns the original flight path
        since it does not violate the geofence.
        '''
        safe_path_wps = [(38.31633, -76.55578), (38.31587947, -76.55120619), (38.31813149, -76.54777558), (38.31542867, -76.54548898), (38.31452744, -76.54205881)]
        legal_path_wps = self.commander.generate_legal_waypoints(safe_path_wps)
        legal_path = LineString(legal_path_wps)

        for i in range(len(safe_path_wps)):
            self.assertEqual(safe_path_wps[i], legal_path_wps[i])

        self.assertTrue(legal_path.within(self.geofence_boundary))

        self.plot_geofence_and_flight_paths("flight_path_within_geofence", safe_path_wps, legal_path_wps, ['green', 'purple'], ['Original Safe Fight Path', 'Generated Legal Flight Path'])

    def plot_geofence_and_flight_paths(self, png_name: str, original_path: list, generated_path: list, colors: list, labels: list):
        '''
        Generate PNG image that plots the geofence, original_path, and generated_path.
        generated_path is obtained through a call to generate_legal_waypoints(original_path).
        original_path has a label of labels[0] and a color of colors[0]. generated_path
        has a label of labels[1] and a color of colors[1].
        '''
        plt.clf()
        
        plt.scatter(self.geofence_longitudes, self.geofence_latitudes, color = 'black', label = 'Geofence')
        for i in range(len(self.geofence_latitudes) - 1):
            plt.plot([self.geofence_longitudes[i], self.geofence_longitudes[i + 1]], [self.geofence_latitudes[i], self.geofence_latitudes[i + 1]], color = 'black')
        plt.plot([self.geofence_longitudes[-1], self.geofence_longitudes[0]], [self.geofence_latitudes[-1], self.geofence_latitudes[0]], color = 'black')

        original_path_latitudes = [pt[0] for pt in original_path]
        original_path_longitudes = [pt[1] for pt in original_path]
        plt.scatter(original_path_longitudes, original_path_latitudes, color = colors[0], label = labels[0])
        for i in range(len(original_path_latitudes) - 1):
            plt.plot([original_path_longitudes[i], original_path_longitudes[i + 1]], [original_path_latitudes[i], original_path_latitudes[i + 1]], color = colors[0])
        plt.plot([original_path_longitudes[-1], original_path_longitudes[0]], [original_path_latitudes[-1], original_path_latitudes[0]], color = colors[0])

        generated_path_latitudes = [pt[0] for pt in generated_path]
        generated_path_longitudes = [pt[1] for pt in generated_path]
        plt.scatter(generated_path_longitudes, generated_path_latitudes, color = colors[1], label = labels[1])
        for i in range(len(generated_path_latitudes) - 1):
            plt.plot([generated_path_longitudes[i], generated_path_longitudes[i + 1]], [generated_path_latitudes[i], generated_path_latitudes[i + 1]], color = colors[1])
        plt.plot([generated_path_longitudes[-1], generated_path_longitudes[0]], [generated_path_latitudes[-1], generated_path_latitudes[0]], color = colors[1])

        plt.legend()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f"test/{png_name}.png")