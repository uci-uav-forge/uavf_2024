from typing import List, Tuple
from libuavf_2024.gnc.util import convert_local_m_to_delta_gps
import numpy as np
import math

class DropzonePlanner:
    # Handles all logic related to controlling drone motion during the payload drop.
    def __init__(self, commander, image_width_m: float, image_height_m: float):
        self.commander = commander
        self.image_width_m = image_width_m
        self.image_height_m = image_height_m
    
    def gen_dropzone_plan(self):
        # Generates dropzone plan with yaws included, in meters.

        # Step 1: find closest corner of dropzone.
        # Set up some helpers to reorient relative to that.
        while not self.commander.got_global_pos or not self.commander.got_pose:
            pass

        dropzone_coords = self.commander.dropzone_bounds_mlocal

        pose = self.commander.cur_pose.pose

        cur_xy = np.array([pose.position.x, pose.position.y])

        closest_idx = min(range(4), key = lambda i: np.linalg.norm(dropzone_coords[i] - cur_xy))

        dist_m1, dist_p1 = \
            [np.linalg.norm(dropzone_coords[closest_idx] - dropzone_coords[(closest_idx + k) % 4]) for k in (-1, 1)]

        drop_h, drop_w = max(dist_m1,dist_p1), min(dist_m1,dist_p1)

        if dist_m1 > dist_p1:
            far_idx = (closest_idx - 1) % 4
            near_idx = (closest_idx + 1) % 4
        else:
            far_idx = (closest_idx + 1) % 4
            near_idx = (closest_idx - 1) % 4
        
        w_unit = dropzone_coords[near_idx] - dropzone_coords[closest_idx]
        w_unit /= np.linalg.norm(w_unit)

        h_unit = dropzone_coords[far_idx] - dropzone_coords[closest_idx]
        h_unit /= np.linalg.norm(h_unit)

        fwd_yaw = np.degrees(np.arccos(np.dot(np.array([1,0]), h_unit)))
        fwd_yaw += 90
        fwd_yaw %= 360
        

        # Step 2: Generate a "zigzag" pattern to sweep the entire dropzone.
        result_wps = []
        for col in range(math.ceil(drop_w/self.image_width_m)):
            wps_col = []

            offset_w = self.image_width_m * (col + 1/2)
            for row in range(math.ceil(drop_h/self.image_height_m)):
                offset_h = self.image_height_m * (row + 1/2)

                wps_col.append(dropzone_coords[closest_idx] + w_unit * offset_w + h_unit * offset_h)
            
            if col % 2:
                result_wps += [(x, (fwd_yaw + 180)%360) for x in wps_col[::-1]]
            else:
                result_wps += [(x, fwd_yaw) for x in wps_col]

        return result_wps

    def conduct_air_drop(self):
        # Called when a waypoint lap has been finished.
        # Expects that the drone is in the air.
        # Moves to drop zone from current position,
        # scans drop zone,
        # navigates to the target best matching the current payload,
        # and releases it.

        dropzone_plan = self.gen_dropzone_plan()
        self.commander.log("planned wps", [self.commander.local_to_gps(wp) for wp, _ in dropzone_plan])
        self.commander.call_imaging_at_wps = True
        self.commander.do_waypoints([self.commander.local_to_gps(wp) for wp, yaw in dropzone_plan], [yaw for wp, yaw in dropzone_plan])
        self.commander.call_imaging_at_wps = False
        detections = self.commander.gather_imaging_detections()

        self.commander.log(detections)

        best_match = max(detections, key = self.match_score)
        self.commander.do_waypoints([self.commander.local_to_gps((best_match.x, best_match.y))])
        self.commander.release_payload()
        
    
    def match_score(self, detection):
        return detection.letter_conf[self.commander.args.payload_letter_id] \
            + detection.shape_conf[self.commander.args.payload_shape_id] \
            + detection.letter_color_conf[self.commander.args.payload_letter_color_id] \
            + detection.shape_color_conf[self.commander.args.payload_shape_color_id]

