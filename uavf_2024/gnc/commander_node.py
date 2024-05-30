import std_msgs.msg
import mavros_msgs.msg
import mavros_msgs.srv
import rclpy
import rclpy.node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import sensor_msgs.msg
import geometry_msgs.msg 
import libuavf_2024.srv
from uavf_2024.gnc.util import read_gps, convert_delta_gps_to_local_m, convert_local_m_to_delta_gps, calculate_turn_angles_deg, read_payload_list
from uavf_2024.gnc.dropzone_planner import DropzonePlanner
from scipy.spatial.transform import Rotation as R
import time
import logging
from datetime import datetime
import numpy as np
from std_srvs.srv import Empty

class CommanderNode(rclpy.node.Node):
    '''
    Manages subscriptions to ROS2 topics and services necessary for the main GNC node. 
    '''

    def __init__(self, args):
        super().__init__('uavf_commander_node')

        np.set_printoptions(precision=8)
        logging.basicConfig(filename='commander_node_{:%Y-%m-%d}.log'.format(datetime.now()), encoding='utf-8', level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )
        self.got_pos = False

        self.arm_client = self.create_client(mavros_msgs.srv.CommandBool, 'mavros/cmd/arming')
        
        self.mode_client = self.create_client(mavros_msgs.srv.SetMode, 'mavros/set_mode')

        self.takeoff_client = self.create_client(mavros_msgs.srv.CommandTOL, 'mavros/cmd/takeoff')

        self.waypoints_client = self.create_client(mavros_msgs.srv.WaypointPush, 'mavros/mission/push')
        self.clear_mission_client = self.create_client(mavros_msgs.srv.WaypointClear, 'mavros/mission/clear')


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
        
        self.payload_client = self.create_client(
            Empty,
            '/payload_drop_service')
        
        self.mission_wps = read_gps(args.mission_file)
        self.dropzone_bounds = read_gps(args.dropzone_file)
        self.payloads = read_payload_list(args.payload_list)

        self.dropzone_planner = DropzonePlanner(self, args.image_width_m, args.image_height_m)
        self.args = args

        self.call_imaging_at_wps = False
        self.imaging_futures = []

        self.turn_angle_limit = 170
    
    def log(self, *args, **kwargs):
        logging.info(*args, **kwargs)
    
    def global_pos_cb(self, global_pos):
        self.got_pos = True
        self.last_pos = global_pos
    
    def got_state_cb(self, state):
        self.cur_state = state
    
    def reached_cb(self, reached):
        if reached.wp_seq > self.last_wp_seq:
            self.log(f"Reached waypoint {reached.wp_seq}")
            self.last_wp_seq = reached.wp_seq

            if self.call_imaging_at_wps:
                self.imaging_futures.append(self.imaging_client.call_async(libuavf_2024.srv.TakePicture.Request()))
    
    def got_pose_cb(self, pose):
        self.cur_pose = pose
        self.cur_rot = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w,]).as_rotvec()
        self.got_pose = True

    def got_global_pos_cb(self, pos):
        #Todo this feels messy - there should be a cleaner way to get home-pos through MAVROS.
        self.last_global_pos = pos
        if not self.got_global_pos:
            self.home_global_pos = pos
            print(self.home_global_pos)
            
            self.dropzone_bounds_mlocal = [convert_delta_gps_to_local_m((pos.latitude, pos.longitude), x) for x in self.dropzone_bounds]
            self.log(f"Dropzone bounds in local coords {self.dropzone_bounds_mlocal}")

            self.got_global_pos = True
    
    def local_to_gps(self, local):
        return convert_local_m_to_delta_gps((self.home_global_pos.latitude,self.home_global_pos.longitude) , local)
    
    def execute_waypoints(self, waypoints, yaws = None, altitude = 0):
        if yaws is None:
            yaws = [float('NaN')] * len(waypoints)

        self.last_wp_seq = -1

        self.log("Pushing waypoints")

        
        waypoints = [(self.last_global_pos.latitude, self.last_global_pos.longitude, altitude)] +  waypoints
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
                    param4 = 0.0;
                    param4 = yaw,

                    x_lat = wp[0],
                    y_long = wp[1],
                    z_alt = wp[2])

                for wp,yaw in zip(waypoints, yaws)]

        
        self.clear_mission_client.call(mavros_msgs.srv.WaypointClear.Request())

        self.log("Delaying before setting mode")
        time.sleep(1)
        self.log("Pushing waypoints and setting mode.")

        
        self.waypoints_client.call(mavros_msgs.srv.WaypointPush.Request(start_index = 0, waypoints = waypoint_msgs))
        # mavros/px4 doesn't consistently set the mode the first time this function is called...
        # retry or fail the script.
        for _ in range(1000):
            self.mode_client.call(mavros_msgs.srv.SetMode.Request( \
                base_mode = 0,
                custom_mode = 'AUTO.MISSION'
            ))
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
        # mocked out for now.
        self.payload_client(libuavf_2024.srv.TakePicture.Request())
        request = Empty.Request()
        self.payload_client.call(request)
        self.log("Payload release requested")
    
    def gather_imaging_detections(self):
        detections = []
        self.log("Waiting for imaging detections.")
        for future in self.imaging_futures:
            while not future.done():
                pass
            detections += future.result().detections
        self.imaging_futures = []
        self.log(f"Successfully retrieved imaging detections: {detections}")
        return detections
    
    def wait_for_takeoff(self):
        '''
        Will be executed before the start of each lap. Will wait for a signal
        indicating that the drone has taken off and is ready to fly the next lap.
        '''
        self.log('Waiting for takeoff')

    def execute_mission_loop(self):
        while not self.got_global_pos:
            pass

        for lap in range(len(self.payloads)):
            self.log(f'Lap {lap}')

            # Wait for takeoff
            self.wait_for_takeoff()

            # Fly waypoint lap
            self.execute_waypoints(self.mission_wps)

            if self.args.exit_early:
                return

            # Fly to drop zone and release current payload
            self.dropzone_planner.conduct_air_drop()

            # Fly back to home position
            self.execute_waypoints([(self.home_global_pos.latitude, self.home_global_pos.longitude)])