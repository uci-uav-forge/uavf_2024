import std_msgs.msg
import mavros_msgs.msg
import mavros_msgs.srv
import rclpy
import rclpy.node
from rclpy.qos import *
import sensor_msgs.msg
import geometry_msgs.msg 
import libuavf_2024.srv
from uavf_2024.imaging.imaging_types import ROSDetectionMessage, Target3D
from uavf_2024.gnc.util import read_gps, convert_delta_gps_to_local_m, convert_local_m_to_delta_gps, read_payload_list, read_gpx_file
from uavf_2024.gnc.dropzone_planner import DropzonePlanner
from scipy.spatial.transform import Rotation as R
import time
import logging
from datetime import datetime
import numpy as np
from uavf_2024.gnc.mission_messages import *

TAKEOFF_ALTITUDE = 20.0

class CommanderNode(rclpy.node.Node):
    '''
    Manages subscriptions to ROS2 topics and services necessary for the main GNC node. 
    '''

    def __init__(self, args):
        super().__init__('uavf_commander_node')

        np.set_printoptions(precision=8)
        logging.basicConfig(filename='commander_node_{:%Y-%m-%d-%m-%s}.log'.format(datetime.now()), format='%(asctime)s %(message)s', encoding='utf-8', level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_ALL,
            depth = 1)
        
        self.got_pos = False

        self.arm_client = self.create_client(mavros_msgs.srv.CommandBool, 'mavros/cmd/arming')   
        self.mode_client = self.create_client(mavros_msgs.srv.SetMode, 'mavros/set_mode')
        self.takeoff_client = self.create_client(mavros_msgs.srv.CommandTOL, 'mavros/cmd/takeoff')
        self.waypoints_client = self.create_client(mavros_msgs.srv.WaypointPush, 'mavros/mission/push')
        self.clear_mission_client = self.create_client(mavros_msgs.srv.WaypointClear, 'mavros/mission/clear')
        self.cmd_long_client = self.create_client(mavros_msgs.srv.CommandLong, 'mavros/cmd/command')

        self.msg_sub = self.create_subscription(
            mavros_msgs.msg.StatusText,
            'mavros/statustext/recv',
            self.status_text_cb,
            qos_profile
        )

        self.msg_pub = self.create_publisher(
            mavros_msgs.msg.StatusText,
            'mavros/statustext/send',
            qos_profile
        )

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
        
        self.gpx_track_map = read_gpx_file(args.gpx_file)
        self.mission_wps, self.dropzone_bounds, self.geofence = self.gpx_track_map['Mission'], self.gpx_track_map['Airdrop Boundary'], self.gpx_track_map['Flight Boundary']
        self.payloads = read_payload_list(args.payload_list)

        self.dropzone_planner = DropzonePlanner(self, args.image_width_m, args.image_height_m)
        self.args = args

        self.call_imaging_at_wps = False
        self.imaging_futures = []

        self.turn_angle_limit = 170

        self.cur_lap = -1
    
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
                self.do_imaging_call()
    
    def do_imaging_call(self):
        self.imaging_futures.append(self.imaging_client.call_async(libuavf_2024.srv.TakePicture.Request()))
    
    def got_pose_cb(self, pose):
        self.cur_pose = pose
        self.cur_rot = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w,]).as_rotvec()
        self.got_pose = True

    def got_global_pos_cb(self, pos):
        #Todo this feels messy - there should be a cleaner way to get home-pos through MAVROS.
        self.last_global_pos = pos
        self.log(f"got global pos: {pos.latitude} {pos.longitude}")
        if not self.got_global_pos:
            self.home_global_pos = pos
            print(self.home_global_pos)
            
            self.dropzone_bounds_mlocal = [convert_delta_gps_to_local_m((pos.latitude, pos.longitude), x) for x in self.dropzone_bounds]
            self.log(f"Dropzone bounds in local coords {self.dropzone_bounds_mlocal}")

            self.got_global_pos = True
    
    def status_text_cb(self, statustext):
        self.log(f"recieved statustext: {statustext}")
        bump_lap = BumpLap.from_string(statustext.text)
        if bump_lap is not None:
            self.cur_lap = max(self.cur_lap, bump_lap.lap_index)
    
    def local_to_gps(self, local):
        return convert_local_m_to_delta_gps((self.home_global_pos.latitude,self.home_global_pos.longitude) , local)

    def get_cur_xy(self):
        pose = self.cur_pose.pose
        return np.array([pose.position.x, pose.position.y])
    
    def execute_waypoints(self, waypoints, yaws = None, do_set_mode=True):
        if yaws is None:
            yaws = [float('NaN')] * len(waypoints)

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

        if do_set_mode:
            self.log("Delaying before setting mode.")
            time.sleep(1)
            self.log("Setting mode.")
            # mavros/px4 doesn't consistently set the mode the first time this function is called...
            # retry or fail the script.
            for _ in range(1000):
                self.mode_client.call(mavros_msgs.srv.SetMode.Request( \
                    base_mode = 0,
                    custom_mode = 'AUTO.MISSION'))
                self.last_wp_seq = -1
                time.sleep(0.2)
                if self.cur_state != None and self.cur_state.mode == 'AUTO.MISSION':
                    self.log("Success setting mode")
                    break
            else:
                self.log("Failure setting mode, quitting.")
                quit()
        else:
            self.log("Didn't set mode, waiting for manual flip to MISSION mode.")

        self.log("Waiting for mission to finish.")

        while self.last_wp_seq != len(waypoints)-1:
            pass
    
    def release_payload(self):
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
    
    def request_load_payload(self, payload):
        payload_request = RequestPayload(shape=payload.shape, shape_col=payload.shape_col, letter=payload.letter, letter_col=payload.letter_col)
        request_msg = payload_request.to_string()
        self.log(f"Requesting {request_msg}.")
        for chunk in [request_msg[i:i+30] for i in range(0,len(request_msg),30)]:
            self.msg_pub.publish(mavros_msgs.msg.StatusText(severity=mavros_msgs.msg.StatusText.NOTICE, text=chunk))

    def execute_mission_loop(self):
        while not self.got_global_pos:
            pass

        if self.args.servo_test:
            self.release_payload()
            return

        if self.args.call_imaging:
            while True:
                self.do_imaging_call()
                detections = self.gather_imaging_detections()
                for detection in detections:
                    detection_gp = self.local_to_gps(detection.position)
                    self.log(f"For detection {detection} would go to {detection_gp}")
                time.sleep(self.args.call_imaging_period)
            
        self.dropzone_planner.gen_dropzone_plan()
        self.request_load_payload(self.payloads[0])
        for lap in range(len(self.payloads)):
            self.log(f"Lap {lap}")

            if lap > 0:
                self.dropzone_planner.advance_current_payload_index()
                self.request_load_payload(self.payloads[self.dropzone_planner.current_payload_index])
            
            # Fly waypoint lap
            self.execute_waypoints(self.mission_wps, do_set_mode=False)

            if self.args.exit_early:
                return

            # Fly to drop zone and release current payload
            self.dropzone_planner.conduct_air_drop()

            # Fly back to home position
            self.execute_waypoints([(self.home_global_pos.latitude, self.home_global_pos.longitude, TAKEOFF_ALTITUDE)])
