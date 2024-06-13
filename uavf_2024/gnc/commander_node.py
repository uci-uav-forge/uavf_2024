from datetime import datetime
import geometry_msgs.msg 
import libuavf_2024.srv
import logging
import mavros_msgs.msg
import mavros_msgs.srv
import numpy as np
from pygeodesy.geoids import GeoidPGM
import rclpy
import rclpy.node
from rclpy.qos import *
from scipy.spatial.transform import Rotation as R
import sensor_msgs.msg
from shapely.geometry import Point, Polygon, LineString
import std_msgs.msg
import time
from uavf_2024.imaging.imaging_types import ROSDetectionMessage, Target3D
from uavf_2024.gnc.util import pose_to_xy, convert_delta_gps_to_local_m, convert_local_m_to_delta_gps, read_payload_list, read_gpx_file, validate_points
from uavf_2024.gnc.dropzone_planner import DropzonePlanner
from uavf_2024.gnc.mission_messages import *

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
        logging.basicConfig(filename='commander_node_{:%Y-%m-%d-%m-%s}.log'.format(datetime.now()), format='%(asctime)s %(message)s', encoding='utf-8', level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_ALL,
            depth = 1)

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
            qos_profile)

        self.msg_pub = self.create_publisher(
            mavros_msgs.msg.StatusText,
            'mavros/statustext/send',
            qos_profile)

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

        self.got_home_pos = False
        self.home_position_sub = self.create_subscription(
            mavros_msgs.msg.HomePosition,
            'mavros/home_position/home',
            self.home_position_cb,
            qos_profile)

        self.setpoint_pub = self.create_publisher(
            geometry_msgs.msg.PoseStamped,
            'mavros/setpoint_position/local',
            qos_profile)

        self.default_altitude_asml = 23.0

        self.gpx_track_map = read_gpx_file(args.gpx_file)
        self.mission_wps, self.dropzone_bounds, self.geofence = self.gpx_track_map['Mission'], self.gpx_track_map['Airdrop Boundary'], self.gpx_track_map['Flight Boundary']
        validate_points(self.mission_wps, self.geofence)
        validate_points(self.dropzone_bounds, self.geofence, False)
        self.payloads = read_payload_list(args.payload_list)
        
        self.dropzone_planner = DropzonePlanner(self, args.image_width_m, args.image_height_m)
        self.args = args

        self.call_imaging_at_wps = False
        self.imaging_futures = []

        self.cur_lap = -1
        self.in_loop = False

        self.got_home_local_pos = False
        self.home_local_pos = None

        self.egm96 = GeoidPGM('/usr/share/GeographicLib/geoids/egm96-5.pgm', kind = -3)
    
        self.last_imaging_time = None
        
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
        if self.in_loop and state.mode not in ["AUTO.MISSION", "AUTO.LOITER"]:
            self.log_statustext(f"Bad mode; {state.mode}. Crashing")
            quit()

    def reached_cb(self, reached):
        '''
        Set self.last_wp_seq. This is the callback function for the subscrption to the 
        mavros/mission/reached topic.
        '''
        if reached.wp_seq > self.last_wp_seq:
            self.log(f"Reached waypoint {reached.wp_seq}")
            self.last_wp_seq = reached.wp_seq
            if self.call_imaging_at_wps:
                self.do_imaging_call()

    def do_imaging_call(self):
        self.imaging_futures.append(self.imaging_client.call_async(libuavf_2024.srv.TakePicture.Request()))
    
    def got_pose_cb(self, pose):
        '''
        Obtain the current local position. This is the callback function for the 
        subscription to the /mavros/local_position/pose topic.
        '''
        self.cur_pose = pose
        self.cur_rot = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w,]).as_rotvec()
        self.got_pose = True
        timestamp = time.time()
        if self.call_imaging_at_wps and (self.last_imaging_time is None or timestamp - self.last_imaging_time < 0.3):
            self.do_imaging_call()
            self.last_imaging_time = timestamp
        if not self.got_home_local_pos:
            self.got_home_local_pos = True
            self.home_local_pose = self.cur_pose
            self.log(f"home local pose is {self.home_local_pose}")

    def got_global_pos_cb(self, pos):
        '''
        This is the callback function for the subscription to the /mavros/global_position/global topic.
        '''
        # Todo this feels messy - there should be a cleaner way to get home-pos through MAVROS.
        self.last_global_pos = pos
        self.got_global_pos = True

    def home_position_cb(self, pos):
        '''
        This is the callback function for the subscription to the mavros/home_position/home topic.
        '''
        if not self.got_home_pos:
            self.log(f"home pos is {pos}")
        self.home_pos = pos
        self.got_home_pos = True
    
    def status_text_cb(self, statustext):
        '''
        This is the callback function for the subscription to the mavros/statustext/recv topic.
        '''
        self.log(f"recieved statustext: {statustext}")
        bump_lap = BumpLap.from_string(statustext.text)
        if bump_lap is not None:
            self.cur_lap = max(self.cur_lap, bump_lap.lap_index)

    def local_to_gps(self, local):
        '''
        Convert local coordinates to global coordinates by simply calling 
        the convert_local_m_to_delta_gps() utility function.
        '''
        local = np.array(local)
        return convert_local_m_to_delta_gps((self.home_pos.geo.latitude, self.home_pos.geo.longitude), local - pose_to_xy(self.home_local_pose))

    def gps_to_local(self, gps):
        return convert_delta_gps_to_local_m((self.home_pos.geo.latitude, self.home_pos.geo.longitude), gps) + pose_to_xy(self.home_local_pose)

    def get_cur_xy(self):
        '''
        Get current drone position.
        '''
        return pose_to_xy(self.cur_pose)
    
    def geoid_height(self, lat, lon):
       '''
       Calculates AMSL to ellipsoid conversion offset.
       Uses EGM96 data with 5' grid and cubic interpolation.
       The value returned can help you convert from meters 
       above mean sea level (AMSL) to meters above
       the WGS84 ellipsoid.
   
       If you want to go from AMSL to ellipsoid height, add the value.
   
       To go from ellipsoid height to AMSL, subtract this value.
       '''
       return self.egm96.height(lat, lon)

    def execute_waypoints(self, waypoints, yaws = None, do_set_mode = True):
        '''
        Fly to each of the waypoints.
        '''
        if self.args.is_maryland:
            waypoints = self.generate_legal_waypoints(waypoints)

        if yaws is None:
            yaws = [float('NaN')] * len(waypoints)

        self.log("Pushing waypoints")

        waypoints = [(self.last_global_pos.latitude, self.last_global_pos.longitude, self.default_altitude_asml)] + waypoints
        validate_points(waypoints, self.geofence)
        
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
        time.sleep(.5)
        self.log("Pushing waypoints.")
        
        self.waypoints_client.call(mavros_msgs.srv.WaypointPush.Request(start_index = 0, waypoints = waypoint_msgs))
        self.log("Delaying before resetting mission progress.")
        time.sleep(.5)
        self.last_wp_seq = -1
        self.log("Set mission progress")
        if do_set_mode:
            self.log("Delaying before setting mode.")
            time.sleep(.5)
            self.log("Setting mode.")
            # mavros/px4 doesn't consistently set the mode the first time this function is called...
            # retry or fail the script.
            for _ in range(1000):
                self.mode_client.call(mavros_msgs.srv.SetMode.Request( \
                    base_mode = 0,
                    custom_mode = 'AUTO.MISSION'))
                time.sleep(0.05)
                if (self.cur_state != None and self.cur_state.mode == 'AUTO.MISSION') or self.last_wp_seq >= -1:
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
        '''
        Send signal to release the payload. The drone has reached the target in the drop 
        zone that matches the payload.
        '''
        deg1 = 180
        deg2 = 100

        deg_to_actuation = lambda x: (x/180)*2 - 1
        self.log("waiting for cmd long client...")
        self.cmd_long_client.wait_for_service()
        self.log("waiting for full stop before payload drop.")
        time.sleep(2)
        self.log_statustext("Releasing payload.")
        for t_deg in list(range(deg1+1,deg2-1,-1)) + list(range(deg2,deg1)):
            #self.log(f"setting to {t_deg}") 
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
            
            time.sleep(.05)

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
    
    def request_load_payload(self, payload):
        '''
        Wait for a signal indicating that the drone should takeoff and start next lap.
        '''
        payload_request = RequestPayload(shape=payload.shape, shape_col=payload.shape_col, letter=payload.letter, letter_col=payload.letter_col)
        request_msg = payload_request.to_string()
        self.log(f"Requesting {request_msg}.")
        self.log_statustext(request_msg)
    
    def log_statustext(self, msg):
        self.log(msg)
        for chunk in [msg[i:i+30] for i in range(0,len(msg),30)]:
            self.msg_pub.publish(mavros_msgs.msg.StatusText(severity=mavros_msgs.msg.StatusText.NOTICE, text=chunk))
    
    def setpoint(self, x, y, z):
        self.setpoint_pub.publish(geometry_msgs.msg.PoseStamped(pose=geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=x,y=y,z=z))))

    def demo_setpoint_loop(self):
        for _ in range(200):
            self.setpoint(0.0,0.0,40.0)
            time.sleep(0.05)
        self.log('setting mode')
        
        self.mode_client.call(mavros_msgs.srv.SetMode.Request( \
                    base_mode = 0,
                    custom_mode = 'OFFBOARD'))
        t0 = time.time()
        while True:
            dt = time.time() - t0
            dt %= 40
            x,y,z = 0.0,0.0,40.0
            p = dt % 10
            if dt < 10:
                x = p-5
                y = -5
            elif dt < 20:
                x=5
                y=p-5 
            elif dt < 30:
                x=5-p
                y=5
            else:
                x=-5
                y=5-p

            self.setpoint(float(x),float(y),float(z))
            time.sleep(0.05)

    def get_closest_intermediate_point(self, destination_wp):
        '''
        Get an intermediate waypoint to fly to before flying to the destination_wp.
        By flying to the intermediate waypoint before the destination_wp, the
        geofence will not be violated.
        '''
        geoid_separation = self.geoid_height(self.last_global_pos.latitude, self.last_global_pos.longitude)
        current_altitude = self.last_global_pos.altitude - geoid_separation
        altitude_asml = max(self.default_altitude_asml, current_altitude)

        left_intermediate_waypoint_global = (38.31605966, -76.55154921, altitude_asml)
        right_intermediate_waypoint_global = (38.31542867, -76.54548898, altitude_asml)
        geofence_middle_pt = (38.31470980862425, -76.54936361414539)

        return right_intermediate_waypoint_global if destination_wp[1] > geofence_middle_pt[1] else left_intermediate_waypoint_global

    def generate_legal_waypoints(self, waypoints: list):
        '''
        Check if the given waypoints produces a flight path that will violate the geofence
        and return a legal set of waypoints that ensures the drone will stay within the
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
        
        while not self.got_global_pos or not self.got_home_pos:
            pass

        if self.args.demo_setpoint_loop:
            self.demo_setpoint_loop()
            return

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

        self.request_load_payload(self.payloads[0])
        self.dropzone_bounds_mlocal = [self.gps_to_local(x) for x in self.dropzone_bounds]
        self.log(f"dropzone bounds = {self.dropzone_bounds_mlocal}")
        for lap in range(len(self.payloads)):
            if lap > 0:
                self.dropzone_planner.advance_current_payload_index()
                self.request_load_payload(self.payloads[self.dropzone_planner.current_payload_index])

            
            self.log_statustext(f"Pushing mission for {lap}")
            self.execute_waypoints(self.mission_wps, do_set_mode=False)

            if self.args.exit_early:
                return

            self.log_statustext(f"Beginning dropzone lap.")
            # Fly to drop zone and release current payload
            self.dropzone_planner.conduct_air_drop()
            
            self.log_statustext("Returning home.")
            # Fly back to home position
            self.execute_waypoints([(self.home_pos.geo.latitude, self.home_pos.geo.longitude, self.default_altitude_asml)])
            self.in_loop = False
