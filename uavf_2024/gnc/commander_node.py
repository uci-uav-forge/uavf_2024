from datetime import datetime
import geometry_msgs.msg 
import libuavf_2024.srv
import logging
import mavros_msgs.msg
import mavros_msgs.srv
import numpy as np
import rclpy
import rclpy.node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
import sensor_msgs.msg
from shapely.geometry import Point, Polygon, LineString
import std_msgs.msg
import time
from uavf_2024.gnc.dropzone_planner import DropzonePlanner
from uavf_2024.gnc.util import read_gps, convert_delta_gps_to_local_m, convert_local_m_to_delta_gps, read_payload_list
from uavf_2024.imaging.imaging_types import ROSDetectionMessage, Target3D

TAKEOFF_ALTITUDE = 20.0

class CommanderNode(rclpy.node.Node):
    '''
    Manages subscriptions to ROS2 topics and services necessary for the main GNC node. 
    '''

    def __init__(self, args):
        '''
        Initialize all clients, subscriptions, and variables needed. Check out 
        http://wiki.ros.org/mavros and http://wiki.ros.org/mavros_msgs to read 
        more about the MAVROS messages and services the drone relies on.
        '''
        super().__init__('uavf_commander_node')

        np.set_printoptions(precision=8)
        logging.basicConfig(filename='commander_node_{:%Y-%m-%d}.log'.format(datetime.now()), encoding='utf-8', level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1)
        
        self.got_pos = False

        self.arm_client = self.create_client(mavros_msgs.srv.CommandBool, 'mavros/cmd/arming')   
        self.mode_client = self.create_client(mavros_msgs.srv.SetMode, 'mavros/set_mode')
        self.takeoff_client = self.create_client(mavros_msgs.srv.CommandTOL, 'mavros/cmd/takeoff')
        self.waypoints_client = self.create_client(mavros_msgs.srv.WaypointPush, 'mavros/mission/push')
        self.clear_mission_client = self.create_client(mavros_msgs.srv.WaypointClear, 'mavros/mission/clear')
        self.cmd_long_client = self.create_client(mavros_msgs.srv.CommandLong, 'mavros/cmd/command')

        self.cur_state = None
        self.state_sub = self.create_subscription(
            mavros_msgs.msg.State,
            'mavros/state',
            self.got_state_cb,
            qos_profile)

        self.got_pose = False
        self.world_position_sub = self.create_subscription(
            geometry_msgs.msg.PoseStamped,
            '/mavros/local_position/pose',
            self.got_pose_cb,
            qos_profile)

        self.got_global_pos = False
        self.global_position_sub = self.create_subscription(
            sensor_msgs.msg.NavSatFix,
            '/mavros/global_position/global',
            self.got_global_pos_cb,
            qos_profile)

        self.last_wp_seq = -1
        self.reached_sub = self.create_subscription(
            mavros_msgs.msg.WaypointReached,
            'mavros/mission/reached',
            self.reached_cb,
            qos_profile)
        
        self.imaging_client = self.create_client(
            libuavf_2024.srv.TakePicture,
            '/imaging_service')
        
        self.mission_wps = read_gps(args.mission_file)
        self.dropzone_bounds = read_gps(args.dropzone_file)
        self.payloads = read_payload_list(args.payload_list)
        self.geofence = read_gps(args.geofence_file)
        
        self.dropzone_planner = DropzonePlanner(self, args.image_width_m, args.image_height_m)
        self.args = args

        self.call_imaging_at_wps = False
        self.imaging_futures = []

        self.left_intermediate_waypoint_global = (38.31605966, -76.55154921)
        self.right_intermediate_waypoint_global = (38.31542867, -76.54548898)
        self.geofence_middle_pt = (38.31470980862425, -76.54936361414539)
        
        self.turn_angle_limit = 170
    
    def log(self, *args, **kwargs):
        '''
        Log message. Intended for convenience and debugging purposes.
        '''
        logging.info(*args, **kwargs)
    
    def got_state_cb(self, state):
        '''
        Set self.cur_state. This is the callback function for the subscription to 
        the mavros/state topic.
        '''
        self.cur_state = state
    
    def reached_cb(self, reached):
        '''
        Set self.last_wp_seq. This is the callback function for the subscrption to the 
        mavros/mission/reached topic.
        '''
        if reached.wp_seq > self.last_wp_seq:
            self.log(f"Reached waypoint {reached.wp_seq}")
            self.last_wp_seq = reached.wp_seq

            if self.call_imaging_at_wps:
                self.imaging_futures.append(self.imaging_client.call_async(libuavf_2024.srv.TakePicture.Request()))
    
    def got_pose_cb(self, pose):
        '''
        Obtain the current local position. This is the callback function for the 
        subscription to the /mavros/local_position/pose topic.
        '''
        self.cur_pose = pose
        self.cur_rot = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w,]).as_rotvec()
        self.got_pose = True

    def got_global_pos_cb(self, pos):
        '''
        Obtain the global home position and set self.home_global_pos. This is the 
        callback function for the subscription to the /mavros/global_position/global topic.
        '''
        # Todo this feels messy - there should be a cleaner way to get home-pos through MAVROS.
        self.last_global_pos = pos
        if not self.got_global_pos:
            self.home_global_pos = pos
            print(self.home_global_pos)
            
            self.dropzone_bounds_mlocal = [convert_delta_gps_to_local_m((pos.latitude, pos.longitude), x) for x in self.dropzone_bounds]
            self.log(f"Dropzone bounds in local coords {self.dropzone_bounds_mlocal}")

            self.got_global_pos = True
    
    def local_to_gps(self, local):
        '''
        Convert local coordinates to global coordinates by simply calling 
        the convert_local_m_to_delta_gps() utility function.
        '''
        return convert_local_m_to_delta_gps((self.home_global_pos.latitude, self.home_global_pos.longitude) , local)

    def get_cur_xy(self):
        '''
        Get current drone position.
        '''
        pose = self.cur_pose.pose
        return np.array([pose.position.x, pose.position.y])
    
    def execute_waypoints(self, waypoints, yaws = None):
        '''
        Fly to each of the waypoints.
        '''
        if yaws is None:
            yaws = [float('NaN')] * len(waypoints)

        self.last_wp_seq = -1

        self.log("Pushing waypoints")

        waypoints = [(self.last_global_pos.latitude, self.last_global_pos.longitude, TAKEOFF_ALTITUDE)] +  waypoints
        yaws = [float('NaN')] + yaws
        self.log(f"Waypoints: {waypoints} Yaws: {yaws}")

        waypoint_msgs = [
                mavros_msgs.msg.Waypoint(
                    frame = mavros_msgs.msg.Waypoint.FRAME_GLOBAL_REL_ALT,
                    command = mavros_msgs.msg.CommandCode.NAV_WAYPOINT,
                    is_current = False,
                    autocontinue = True,

                    param1 = 0.0,
                    param2 = 5.0,
                    param3 = 0.0,
                    param4 = yaw,

                    x_lat = wp[0],
                    y_long = wp[1],
                    z_alt = wp[2])

                for wp,yaw in zip(waypoints, yaws)]
        
        self.clear_mission_client.call(mavros_msgs.srv.WaypointClear.Request())

        self.log("Delaying before pushing waypoints.")
        time.sleep(1)
        self.log("Pushing waypoints.")
        
        self.waypoints_client.call(mavros_msgs.srv.WaypointPush.Request(start_index = 0, waypoints = waypoint_msgs))
        self.log("Delaying before setting mode.")
        time.sleep(1)
        self.log("Setting mode.")
        # mavros/px4 doesn't consistently set the mode the first time this function is called...
        # retry or fail the script.
        for _ in range(1000):
            self.mode_client.call(mavros_msgs.srv.SetMode.Request(base_mode = 0, custom_mode = 'AUTO.MISSION'))
            time.sleep(0.2)
            if self.cur_state != None and self.cur_state.mode == 'AUTO.MISSION':
                self.log("Success setting mode")
                break
        else:
            self.log("Failure setting mode, quitting.")
            quit()

        self.log("Waiting for mission to finish.")

        while self.last_wp_seq != len(waypoints)-1:
            pass
    
    def release_payload(self):
        '''
        Send signal to release the payload. The drone has reached the target in the drop 
        zone that matches the payload.
        '''
        deg1 = 180
        deg2 = 100

        deg_to_actuation = lambda x: (x/180)*2 - 1
        self.log("waiting for cmd long client...")
        self.cmd_long_client.wait_for_service()
        for t_deg in list(range(deg1+1,deg2-1,-1)) + list(range(deg2,deg1)):
            self.log(f"setting to {t_deg}") 
            a = deg_to_actuation(t_deg)
            self.cmd_long_client.call(
                mavros_msgs.srv.CommandLong.Request(
                    command = 187,
                    confirmation = 1,
                    param1 = a,
                    param2 = a,
                    param3 = a,
                    param4 = a,
                    param5 = a,
                    param6 = 0.0,
                    param7 = 0.0
                )
            )

            time.sleep(.1)
    
    def gather_imaging_detections(self):
        '''
        Collect the imaging detections stored in self.imaging_futures. self.imaging_futures
        contains the results collected from calling the TakePicture service.
        '''
        detections = []
        self.log("Waiting for imaging detections.")
        
        for future in self.imaging_futures:
            while not future.done():
                pass
            for detection in future.result().detections:
                detections.append(Target3D.from_ros(ROSDetectionMessage(detection.timestamp, detection.x, detection.y, detection.z,
                                                                        detection.shape_conf, detection.letter_conf, 
                                                                        detection.shape_color_conf, detection.letter_color_conf, detection.id)))
        
        self.imaging_futures = []
        self.log(f"Successfully retrieved imaging detections: {detections}")
        
        return detections
    
    def wait_for_takeoff(self):
        '''
        Wait for a signal indicating that the drone should takeoff and start next lap.
        '''
        self.log('Waiting for takeoff')

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
            path = LineString([start_wp, destination_wp])
            
            legal_waypoints.append(start_wp)
            if not path.within(geofence_bounds):
                legal_waypoints.append(self.get_closest_intermediate_point(destination_wp))
                
        legal_waypoints.append(waypoints[-1])

        return legal_waypoints


    def execute_mission_loop(self):
        '''
        Execute one mission loop which consists of flying one waypoint lap, flying to the drop zone,
        releasing the payload in the drop zone, and flying back to the home position.
        '''
        while not self.got_global_pos:
            pass

        if self.args.servo_test:
            self.release_payload()
            return
            
        self.dropzone_planner.gen_dropzone_plan()

        for lap in range(len(self.payloads)):
            self.log(f"Lap {lap}")

            # Wait for takeoff
            self.wait_for_takeoff()

            # Fly waypoint lap
            self.execute_waypoints(self.mission_wps)

            if self.args.exit_early:
                return

            # Fly to drop zone and release current payload
            self.dropzone_planner.conduct_air_drop()

            # Fly back to home position
            self.execute_waypoints([(self.home_global_pos.latitude, self.home_global_pos.longitude, TAKEOFF_ALTITUDE)])
