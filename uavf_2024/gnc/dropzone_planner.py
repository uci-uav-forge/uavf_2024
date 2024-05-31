import numpy as np
import math
from uavf_2024.imaging.tracker import TargetTracker
from uavf_2024.gnc.util import is_inside_bounds_local
altitude = 20.0

class DropzonePlanner:
    '''
    Handles all logic related to controlling drone motion during the payload drop.
    '''

    def __init__(self, commander: 'CommanderNode', image_width_m: float, image_height_m: float):
        self.commander = commander
        self.image_width_m = image_width_m
        self.image_height_m = image_height_m
        self.target_tracker = TargetTracker()
        self.detections = []
        self.current_payload_index = 0
        self.has_scanned_dropzone = False
        self.dist_btwn_img_wps = 2

        self.dropped_payloads = []

        np.set_printoptions(precision=20)
    
    def gen_dropzone_plan(self):
        '''
        Generates dropzone plan with yaws included, in meters.
        '''

        # Step 1: Find closest corner of dropzone.
        # Set up some helpers to reorient relative to that.
        while not self.commander.got_global_pos or not self.commander.got_pose:
            pass

        dropzone_coords = self.commander.dropzone_bounds_mlocal

        cur_xy = self.commander.get_cur_xy()
        self.commander.log(f"bounds: {dropzone_coords}")
        self.commander.log(f"homepos: {self.commander.home_global_pos}")
        self.commander.log(f"current xy: {cur_xy}")

        closest_idx = min(range(4), key = lambda i: np.linalg.norm(dropzone_coords[i] - cur_xy))

        self.commander.log(f"Closest corner is {dropzone_coords[closest_idx]}")
        self.commander.log(f"GPS: {self.commander.local_to_gps(dropzone_coords[closest_idx])}")

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

        self.commander.log(f"Dropzone dimensions are {drop_h}x{drop_w}")
        self.commander.log(f"Dropzone unit vectors are {w_unit} and {h_unit}")

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

    def scan_dropzone(self):
        '''
        Will be executed after the drone completes the first waypoint lap. 
        Drone will move to drop zone from its current position and scan the
        entire drop zone. Will update target_tracker with the new detections.
        '''
        dropzone_plan = self.gen_dropzone_plan()

        self.commander.log(f"Local coords: {dropzone_plan}")
        self.commander.log(f"Planned waypoints: {[self.commander.local_to_gps(wp) for wp, _ in dropzone_plan]}")

        self.commander.call_imaging_at_wps = True
        self.commander.execute_waypoints([np.concatenate((self.commander.local_to_gps(wp), np.array([altitude]))) for wp, yaw in dropzone_plan], [yaw for wp, yaw in dropzone_plan])
        self.commander.call_imaging_at_wps = False
        detections = self.commander.gather_imaging_detections()
        self.target_tracker.update(detections)

        self.commander.log(f"Imaging detections: {detections}")

    def generate_wps_to_target(self, target_x, target_y, cur_x, cur_y):
        '''
        Will generate a path of waypoints from the drone's current position
        to the target. These waypoints represent the locations at which
        the drone should take images of the drop zone below.
        '''
        current_x, current_y = cur_x, cur_y
        x_distance, y_distance = current_x - target_x, current_y - target_y

        divisor = abs(x_distance) / self.dist_btwn_img_wps
        if abs(y_distance) > abs(x_distance):
            divisor = abs(y_distance) / self.dist_btwn_img_wps

        x_step = x_distance / divisor
        y_step = y_distance / divisor
        self.commander.log(f"x_step: {x_step}, y_step: {y_step}")
        
        waypoints = [(target_x, target_y)]
        new_wp = (target_x + x_step, target_y + y_step)
        running_x_dist, running_y_dist = abs(x_step), abs(y_step)
        while (is_inside_bounds_local(self.commander.dropzone_bounds_mlocal, new_wp) and running_x_dist < abs(x_distance) and running_y_dist < abs(y_distance)):
            waypoints.append(new_wp)
            new_wp = (new_wp[0] + x_step, new_wp[1] + y_step)
            running_x_dist += abs(x_step)
            running_y_dist += abs(y_step)

        waypoints.reverse()
        return waypoints

    def conduct_air_drop(self):
        '''
        Will be executed each time the drone completes a waypoint lap.
        Drone will navigate to the target that best matches its current 
        payload and will release the payload.
        '''

        # Scan the drop zone if the drone has completed its first waypoint lap
        if not self.has_scanned_dropzone:
            self.scan_dropzone()
            self.has_scanned_dropzone = True
        
        # Find the target that best matches the payload
        best_match = self.target_tracker.estimate_positions(self.commander.payloads)[self.current_payload_index]
        best_match_x, best_match_y = best_match.position[0], best_match.position[1]
        self.commander.log(f"best_match_x: {best_match_x}, best_match_y: {best_match_y}")

        # Generate path of waypoints to the target to take images at
        next_wps = self.generate_wps_to_target(best_match_x, best_match_y, self.commander.cur_pose.pose.position.x, self.commander.cur_pose.pose.position.y)
        self.commander.log(f"Opportunistic imaging waypoints: {next_wps}")
        
        # Fly along the path of waypoints to the target
        self.commander.call_imaging_at_wps = True
        self.commander.execute_waypoints([np.concatenate((self.commander.local_to_gps(wp), np.array([altitude]))) for wp in next_wps])
        self.commander.call_imaging_at_wps = False
        detections = self.commander.gather_imaging_detections()
        self.target_tracker.update(detections)

        # Release the payload
        self.commander.release_payload()
        self.commander.log(f"Released payload {self.current_payload_index}")
    
    def advance_current_payload_index(self):
        self.dropped_payloads.append(self.current_payload_index)
        self.current_payload_index = max(
            (i for i in range(len(self.commander.payloads)) if i not in self.dropped_payloads),
            key = lambda i: self.target_tracker.confidence_score(self.commander.payloads[i]))