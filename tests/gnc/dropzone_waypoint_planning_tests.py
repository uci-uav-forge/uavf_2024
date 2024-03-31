import unittest
from uavf_2024.gnc.dropzone_planner import DropzonePlanner
from uavf_2024.gnc.util import *
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt


# Mocks funcitonality of commander node used by dropzone planner.
class MockCommander:
    def __init__(self, lat, lon, bounds):
        self.got_global_pos = True
        self.got_pose = True
        self.home_lat = lat
        self.home_lon = lon
        self.home_global_pos = (lat,lon)
        self.dropzone_bounds = bounds

        self.dropzone_bounds_mlocal = [convert_delta_gps_to_local_m((self.home_lat, self.home_lon), x) for x in self.dropzone_bounds]

    def local_to_gps(self, local):
        return convert_local_m_to_delta_gps((self.home_lat,self.home_lon) , local)

    def get_cur_xy(self):
        return np.array([-44.61928939819336 -37.73757553100586])

    def log(self, msg):
        print(msg)



class TestDropzonePlanner(unittest.TestCase):
    def sanity_check_conversions(self, label, gps_filename, start_lat, start_lon):
        wpts = read_gps(gps_filename)
        commander = MockCommander(start_lat, start_lon, wpts)

        twice_converted = list(map(commander.local_to_gps, commander.dropzone_bounds_mlocal))

        plt.clf()
        plt.plot([gp[0] for gp in wpts], [gp[1] for gp in wpts])
        plt.plot([gp[0] for gp in twice_converted], [gp[1] for gp in twice_converted])
        plt.plot([start_lat], [start_lon], '*')
        plt.savefig(f"test/test_conversions_{label}.png")

        for i in range(len(wpts)):
            assert np.linalg.norm(twice_converted[i] - wpts[i]) < 0.001, "Converting twice should not greatly change position."

    def sanity_check_dropzone_plan(self, area_name, dropzone_filename, start_lat, start_lon):
        wpts = read_gps(dropzone_filename)
        commander = MockCommander(start_lat, start_lon, wpts)
        planner = DropzonePlanner(commander, 12, 9)

        polygon = Polygon(commander.dropzone_bounds_mlocal)
        plan = planner.gen_dropzone_plan()
        plt.clf()
        plt.plot([gp[0] for gp in commander.dropzone_bounds_mlocal + [commander.dropzone_bounds_mlocal[0]]], [gp[1] for gp in commander.dropzone_bounds_mlocal + [commander.dropzone_bounds_mlocal[0]]])
        plt.plot([gp[0] for gp,yw in plan], [gp[1] for gp, yw in plan])
        plt.savefig(f"test/test_{area_name}.png")

        print("Planned ", plan)
        print("Boundary wpts (gps)", wpts)

        for xy, yaw in plan:
            assert polygon.exterior.distance(Point(xy)) <= 24, "All planned waypoints should be within or near the dropzone (meters)"
        

        
        polygon2 = Polygon(wpts)
        gps = [commander.local_to_gps(xy) for xy,yw in plan]

        plt.clf()
        plt.plot([gp[0] for gp in gps], [gp[1] for gp in gps])
        plt.plot([wp[0] for wp in wpts + [wpts[0]]], [wp[1] for wp in wpts + [wpts[0]]])
        plt.savefig(f"test/test_{area_name}_gps.png")

        print(f"GPS is {gps}")

        for gp in gps:
            assert polygon2.exterior.distance(Point(gp)) <= .05, "All planned waypoints should be within or near the dropzone (gps)"
        pass

    def test_conversions(self):
        self.sanity_check_conversions("arc_club_field", "uavf_2024/gnc/data/ARC/CLUB_FIELD/AIRDROP_BOUNDARY", 33.6423003, -117.8268298)
        self.sanity_check_conversions("maryland", "uavf_2024/gnc/data/AIRDROP_BOUNDARY", 38.31633, -76.55578)

    def test_arc_club_field(self):
        self.sanity_check_dropzone_plan("arc_club_field", "uavf_2024/gnc/data/ARC/CLUB_FIELD/AIRDROP_BOUNDARY", 33.6423003, -117.8268298)
        
    def test_maryland(self):
       self.sanity_check_dropzone_plan("maryland", "uavf_2024/gnc/data/AIRDROP_BOUNDARY", 38.31633, -76.55578)
