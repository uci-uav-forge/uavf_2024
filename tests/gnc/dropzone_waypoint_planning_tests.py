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
    '''
    Waypoints: [(33.6419614, -117.8273124, 20.0), array([  33.641858989078614, -117.82649540918828 ,   20.               ]), array([  33.64179097235305, -117.82644250730468,   20.              ]), array([  33.6417229556041 , -117.82638960550426,   20.              ]), array([  33.64165493883178, -117.82633670378704,   20.              ]), array([  33.64158692203606, -117.826283802153  ,   20.              ]), array([  33.64151890521697, -117.82623090060216,   20.              ]), array([  33.64156490023077, -117.82611382118294,   20.              ]), array([  33.64163291709945, -117.82616672266985,   20.              ]), array([  33.64170093394475, -117.82621962423997,   20.              ]), array([  33.641768950766654, -117.82627252589326 ,   20.               ]), array([  33.64183696756518, -117.82632542762974,   20.              ]), array([  33.641904984340336, -117.82637832944943 ,   20.               ]), array([  33.641950979490844, -117.82626124958605 ,   20.               ]), array([  33.641882962666116, -117.8262083478303  ,   20.               ]), array([  33.64181494581799, -117.82615544615774,   20.              ]), array([  33.641746928946496, -117.82610254456839 ,   20.               ]), array([  33.64167891205162, -117.82604964306222,   20.              ]), array([  33.64161089513336, -117.82599674163923,   20.              ])] Yaws: [nan, 146.95428199048845, 146.95428199048845, 146.95428199048845, 146.95428199048845, 146.95428199048845, 146.95428199048845, 326.9542819904884, 326.9542819904884, 326.9542819904884, 326.9542819904884, 326.9542819904884, 326.9542819904884, 146.95428199048845, 146.95428199048845, 146.95428199048845, 146.95428199048845, 146.95428199048845, 146.95428199048845]
    '''
    def sanity_check_dropzone_plan(self, area_name, dropzone_filename, start_lat, start_lon):
        wpts = read_gps(dropzone_filename)
        commander = MockCommander(start_lat, start_lon, wpts)
        planner = DropzonePlanner(commander, 12, 9)

        polygon = Polygon(commander.dropzone_bounds_mlocal)
        plan = planner.gen_dropzone_plan()

        array = np.array

        xy2 = [(array([ 31.077328944095182, -48.90399809992625 ]), 146.95428191266546), (array([ 35.985101510482885, -56.4481195488832  ]), 146.95428191266546), (array([ 40.892874076870584, -63.99224099784014 ]), 146.95428191266546), (array([ 45.80064664325828, -71.5363624467971 ]), 146.95428191266546), (array([ 50.70841920964598, -79.08048389575404]), 146.95428191266546), (array([ 55.61619177603368, -86.624605344711  ]), 146.95428191266546), (array([ 66.47773771194771, -81.52295660317367]), 326.95428191266546), (array([ 61.56996514556001, -73.97883515421671]), 326.95428191266546), (array([ 56.66219257917231, -66.43471370525977]), 326.95428191266546), (array([ 51.754420012784614, -58.890592256302824]), 326.95428191266546), (array([ 46.846647446396915, -51.346470807345874]), 326.95428191266546), (array([ 41.938874880009216, -43.80234935838892 ]), 326.95428191266546), (array([ 52.800420815923246, -38.70070061685159 ]), 146.95428191266546), (array([ 57.708193382310945, -46.24482206580854 ]), 146.95428191266546), (array([ 62.615965948698644, -53.78894351476549 ]), 146.95428191266546), (array([ 67.52373851508634 , -61.333064963722435]), 146.95428191266546), (array([ 72.43151108147404, -68.87718641267938]), 146.95428191266546), (array([ 77.33928364786175, -76.42130786163634]), 146.95428191266546)]

        xy3 = [array([ 72.31493638025228, -73.17124293577768]), array([ 49.12227884204777, -87.59031368508366]), array([ 23.146284353674467, -47.66057855917786 ]), array([ 49.122011945152465, -35.45982569595279 ])]

        plt.clf()
        plt.plot([gp[0] for gp in commander.dropzone_bounds_mlocal + [commander.dropzone_bounds_mlocal[0]]], [gp[1] for gp in commander.dropzone_bounds_mlocal + [commander.dropzone_bounds_mlocal[0]]])
        plt.plot([gp[0] for gp,yw in plan], [gp[1] for gp, yw in plan])
        plt.plot([xy[0] for xy,_ in xy2], [xy[1] for xy,_ in xy2])
        plt.plot([xy[0] for xy in xy3], [xy[1] for xy in xy3])
        plt.savefig(f"test/test_{area_name}.png")

        print("Planned ", plan)
        print(wpts)

        for xy, yaw in plan:
            assert polygon.exterior.distance(Point(xy)) <= 12, f"All planned waypoints should be within or near the dropzone but {xy} is not."
        

        
        polygon2 = Polygon(wpts)
        gps = [commander.local_to_gps(xy) for xy,yw in plan]

        plt.clf()
        plt.plot([gp[0] for gp in gps], [gp[1] for gp in gps])
        plt.plot([wp[0] for wp in wpts + [wpts[0]]], [wp[1] for wp in wpts + [wpts[0]]])

        bad_wpts = [(33.6419614, -117.8273124, 20.0), array([  33.641858989078614, -117.82649540918828 ,   20.               ]), array([  33.64179097235305, -117.82644250730468,   20.              ]), array([  33.6417229556041 , -117.82638960550426,   20.              ]), array([  33.64165493883178, -117.82633670378704,   20.              ]), array([  33.64158692203606, -117.826283802153  ,   20.              ]), array([  33.64151890521697, -117.82623090060216,   20.              ]), array([  33.64156490023077, -117.82611382118294,   20.              ]), array([  33.64163291709945, -117.82616672266985,   20.              ]), array([  33.64170093394475, -117.82621962423997,   20.              ]), array([  33.641768950766654, -117.82627252589326 ,   20.               ]), array([  33.64183696756518, -117.82632542762974,   20.              ]), array([  33.641904984340336, -117.82637832944943 ,   20.               ]), array([  33.641950979490844, -117.82626124958605 ,   20.               ]), array([  33.641882962666116, -117.8262083478303  ,   20.               ]), array([  33.64181494581799, -117.82615544615774,   20.              ]), array([  33.641746928946496, -117.82610254456839 ,   20.               ])]
        plt.plot([gp[0] for gp in bad_wpts], [gp[1] for gp in bad_wpts])
        plt.savefig(f"test/test_{area_name}_gps.png")

        print(f"GPS is {gps}")

        for gp in gps:
            assert polygon2.exterior.distance(Point(gp)) <= .1, "Planned wps should be within or near the dropzone."
        pass

    def test_arc_club_field(self):
        self.sanity_check_dropzone_plan("arc_club_field", "uavf_2024/gnc/data/ARC/UPPER_FIELD_DROPZONE", 33.6422999, -117.82683)
        

    #def test_maryland(self):
    #    self.sanity_check_dropzone_plan("maryland", "uavf_2024/gnc/data/AIRDROP_BOUNDARY", 38.31633, -76.55578)
