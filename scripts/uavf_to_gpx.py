import gpxpy
import gpxpy.gpx
import argparse
from uavf_2024.gnc.util import *

ap = argparse.ArgumentParser()
ap.add_argument('mission_wpts')
ap.add_argument('dropzone_wpts')
ap.add_argument('output_filename')
args = ap.parse_args()


mission_gp = [x[:2] for x in read_gps(args.mission_wpts)]
dropzone_gp = [x for x in read_gps(args.dropzone_wpts)]

# Create a new GPX object
gpx = gpxpy.gpx.GPX()

for coords in (mission_gp, dropzone_gp):
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    # Create a new GPX track segment
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    # Add points to the segment
    for latitude, longitude in coords:
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude, longitude))

# You can add metadata and other details here if needed

# Generate the GPX string
gpx_string = gpx.to_xml()

# Save the GPX string to a file
filename = args.output_filename
with open(filename, "w") as file:
    file.write(gpx_string)