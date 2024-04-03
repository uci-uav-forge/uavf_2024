from uavf_2024.gnc.payload import Payload
from uavf_2024.imaging import ProbabilisticTargetDescriptor
from geographiclib.geodesic import Geodesic
import numpy as np
import math

def read_gps(fname):
    with open(fname) as f:
        return [tuple(map(float, line.split(','))) for line in f]

def read_payload_list(fname):
    with open(fname) as f:
        return [Payload(line.split(',')) for line in f]

def convert_delta_gps_to_local_m(gp1, gp2):
    geod = Geodesic.WGS84
    g0 = geod.Inverse(*gp1, gp2[0], gp1[1])
    g1 = geod.Inverse(gp2[0], gp1[1], *gp2)

    return np.array([g1['s12']*math.copysign(1, g1['azi1']), -g0['s12']*math.copysign(1, g0['azi1'])])

def convert_local_m_to_delta_gps(gp0, dm):
    geod = Geodesic.WGS84
    gr = geod.Direct(*gp0, 0, dm[1])
    gr2 = geod.Direct(gr['lat2'],gr['lon2'],90, dm[0])
    return np.array([gr2['lat2'], gr2['lon2']])

def calculate_turn_angles_deg(path_coordinates):
    norm_vectors = []

    for i in range(len(path_coordinates) - 1):
        tail_x, tail_y = path_coordinates[i][0], path_coordinates[i][1]
        head_x, head_y = path_coordinates[i + 1][0], path_coordinates[i + 1][1]
        
        result_vector = np.array([head_x - tail_x, head_y - tail_y])
        norm_vectors.append(result_vector / np.linalg.norm(result_vector))

    turn_angles = []
    for i in range(len(norm_vectors) - 1):
        turn_angles.append(np.degrees(np.arccos(np.dot(norm_vectors[i], norm_vectors[i + 1]))))
    
    return turn_angles