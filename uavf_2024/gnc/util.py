from uavf_2024.imaging import CertainTargetDescriptor
from geographiclib.geodesic import Geodesic
from shapely.geometry import Point, Polygon
import numpy as np
import math

# def is_point_within_fence(point, fence):
#     # Convert the list of tuples representing the border to a Shapely Polygon
#     fence_polygon = Polygon(fence)
#     point = Point(point)
#     return fence_polygon.contains(point)
#
# def validate_gps_data(data, geofence):
#     for point_tuple in data:
#         if not is_point_within_fence(point_tuple, geofence):
#             return False
#     return True
#
# def read_geofence(fname):
#     # Creates a list of tuples of (lat, lon) of the geofence
#     with open(fname) as f:
#         return [tuple(map(float, line.split(','))) for line in f]
#
# def read_gps(fname, geofence):
#     with open(fname) as f:
#         data = [tuple(map(float, line.split(','))) for line in f]
#     if validate_gps_data(data, geofence): # Passes coords to validate that they're within the geofence
#         return data
#     else:
#         raise ValueError("Invalid GPS data format or outside geofence boundaries.")

def read_payload_list(fname):
    payload_list = []
    with open(fname) as f:
        for line in f:
            shape_col, shape, letter_col, letter = line.split()
            payload_list.append(CertainTargetDescriptor(shape_col, shape, letter_col, letter))

    return payload_list

def is_inside_bounds_local(bounds, pt):
    p = Point(pt[0], pt[1])
    boundary = Polygon(bounds)

    return p.within(boundary)

def convert_delta_gps_to_local_m(gp1, gp2):

    # sets up the following triangle to convert to local coordinates:
    #  X - - - - - - - - - - - gp2
    #  |          (g1)
    #  | (g0)
    #  |
    # gp1

    # azi1 corresponds to the angle of each line:
    # it will be (+90=east, -90=west) for g1 and (0 = north, 180 = south) for g0
    # s12 is the distance.


    geod = Geodesic.WGS84
    g0 = geod.Inverse(*gp1, gp2[0], gp1[1])
    g1 = geod.Inverse(gp2[0], gp1[1], *gp2)
    return np.array([g1['s12']*(1 if g1['azi1'] > 0 else -1), g0['s12']*(-1 if g0['azi1'] > 90 else 1)])

def convert_local_m_to_delta_gps(gp0, dm):

    # first travel dm[1] meters north/south
    # then travel dm[0] meters east/west

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