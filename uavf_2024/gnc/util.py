from geographiclib.geodesic import Geodesic
import gpxpy
import gpxpy.gpx
import math
import numpy as np
from shapely.geometry import Point, Polygon
from uavf_2024.imaging import CertainTargetDescriptor

def read_gps(fname):
    with open(fname) as f:
        return [tuple(map(float, line.split(','))) for line in f]

def read_payload_list(fname):
    payload_list = []
    with open(fname) as f:
        for line in f:
            shape_col, shape, letter_col, letter = line.split()
            payload_list.append(CertainTargetDescriptor(shape_col, shape, letter_col, letter))
    
    return payload_list

def extract_track_label(track: gpxpy.gpx.GPXTrack):
    '''
    Depending on the given track's name, return one of the four following labels:
    'Mission', 'Airdrop Boundary', 'Flight Boundary', 'Unknown Track'.
    '''
    if track.name.endswith('Mission'):
        return 'Mission'
    elif track.name.endswith('Airdrop Boundary'):
        return 'Airdrop Boundary'
    elif track.name.endswith('Flight Boundary'):
        return 'Flight Boundary'
    else:
        return 'Unknown Track'

def extract_coordinates(track: gpxpy.gpx.GPXTrack):
    '''
    Return the list of coordinates that make up the given track.
    '''
    coordinates = []

    if track.name.endsWith('Mission'):
        for segment in track.segments:
            for point in segment.points:
                coordinates.append((point.latitude, point.longitude, point.elevation))
    else:
        for segment in track.segments:
            for point in segment.points:
                coordinates.append((point.latitude, point.longitude))

    return coordinates

def read_gpx_file(file_name: str):
    '''
    Return a dictionary with key-value pairs that represent the main tracks 
    in the given GPX file. The key is the track's label, which will either be
    Flight Boundary, Airdrop Boundary, Mission, or Unknown Track. The value 
    is a list of GPS points that describe the associated track.
    '''
    gpx_file = open(file_name, 'r')
    gpx = gpxpy.parse(gpx_file)
    track_map = {extract_track_label(track): extract_coordinates(track) for track in gpx.tracks}
    return track_map

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