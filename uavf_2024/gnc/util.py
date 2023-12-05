from geographiclib.geodesic import Geodesic
import numpy as np

def read_gps(fname):
    with open(fname) as f:
        return [tuple(map(float, line.split(','))) for line in f]

def convert_delta_gps_to_local_m(gp1, gp2):
    geod = Geodesic.WGS84
    g0 = geod.Inverse(*gp1, gp2[0], gp1[1])
    g1 = geod.Inverse(gp2[0], gp1[1], *gp2)

    return np.array([g1['s12'], g0['s12']])

def convert_local_m_to_delta_gps(gp0, dm):
    geod = Geodesic.WGS84

    azi = np.degrees(np.arctan2(dm[0], -dm[1]))
    len = np.linalg.norm(dm)

    gr = geod.Direct(*gp0, azi, len)
    print(azi, len)

    return np.array([gr['lat2'], gr['lon2']])

