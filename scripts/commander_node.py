#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import TrajectorySetpoint, VehicleAttitudeSetpoint, VehicleRatesSetpoint,\
    OffboardControlMode,  VehicleCommand, VehicleStatus, VehicleOdometry, VehicleGlobalPosition
from uavf_ros2_msgs.msg import GpsAltitudePosition, NedEnuOdometry, NedEnuSetpoint, NedEnuWaypoint

from px4_offboard_mpc.conversions import convert_quaternion_to_euler_angles, convert_NED_ENU_in_inertial,\
    convert_NED_ENU_in_body, convert_body_to_inertial_frame, convert_inertial_to_body_frame
import numpy as np
from scipy.spatial.transform import Rotation


class CommanderNode(Node):
    ''' Receives input from every ongoing process.
        Output to PX4 Interface Node.
    '''

    def __init__(self):
        super().__init__('commander_node')
        ''' Initialize publishers, subscribers, and class attributes.
        '''
        # Configure QoS profile according to PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


        ''' This section talks to our ROS network '''
        # publisher for gps and altitude feedback in desired format
        self.gps_alt_pub = self.create_publisher(
            GpsAltitudePosition, '/commander/gps_altitude_position', qos_profile)
        # publisher position, velocity, angle, and angle rate feedback in desired format
        self.ned_enu_odom_pub = self.create_publisher(
            NedEnuOdometry, '/commander/ned_enu_odometry', qos_profile)
        # tell the trajectory planner where to plan for
        self.traj_plan_pub = self.create_publisher(
            NedEnuOdometry, '/commander/trajectory_planner_command', qos_profile)
        
        # subscribe to topics owned by slave process nodes
        self.traj_plan_sub = self.create_subscription(
            NedEnuSetpoint, '/trajectory_planner/ned_enu_setpoint', self.traj_plan_cb, qos_profile)
        self.wp_tracker_sub = self.create_subscription(
            NedEnuWaypoint, '/waypoint_tracker/ned_enu_waypoint', self.wp_tracker_cb, qos_profile
        )
        
        
        ''' This section talks to PX4 '''
        # publishers for heartbeat and commands to px4
        self.offboard_ctrl_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        # publishers for position, velocity, angle, and angle rate setpoints to px4
        self.traj_setpt_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.att_setpt_pub = self.create_publisher(
            VehicleAttitudeSetpoint, 'fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.rate_setpt_pub = self.create_publisher(
            VehicleRatesSetpoint, 'fmu/in/vehicle_rates_setpoint', qos_profile)
        
        # subscribers for px4 status, global position, and odometry (NED)
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_cb, qos_profile)
        self.global_pos_sub = self.create_subscription(
            VehicleGlobalPosition, '/fmu/out/vehicle_global_position', self.global_pos_cb, qos_profile)
        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos_profile)
        
        self.traj_planner_is_ENU = True
        self.all_wp_reached = False
        self.waypoint_is_ENU = False
        self.waypoint = np.zeros(3, dtype=np.float32)
        self.status = VehicleStatus()

        '''
        self.setpt_count = 0
        self.takeoff_height = -5.0

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_cb)
        '''
    
    
    def make_decision(self):
        ''' Reads in class attributes that are updated by worker processes.
            Publishes the offboard control heartbeat. Arrives at a decision 
            based on an arbitrarily defined switch state.
        '''
        self.publish_heartbeat()
        if not self.all_wp_reached:
            self.publish_trajectory_planner_command(self, self.waypoint, self.waypoint_is_ENU)
        else:
            self.land()
            self.disarm()

    
    def traj_plan_cb(self, ned_enu_setpt):
        ''' Reads the setpoint calculated by the trajectory planner
            and publishes the corresponding setpoints to PX4.
            Incomplete.
        '''
        self.traj_planner_is_ENU = ned_enu_setpt.is_enu
        pass
    

    def wp_tracker_cb(self, ned_enu_wp):
        ''' Gets the "NED-ENU-ness" of the waypoint tracker
            and the waypoint to go to.
        '''
        if not ned_enu_wp.all_waypoints_reached:
            self.waypoint_is_ENU = ned_enu_wp.is_enu
            self.waypoint = ned_enu_wp.position_waypoint
        else:
            self.destroy_subscription(self.wp_tracker_sub)
            self.all_wp_reached = ned_enu_wp.all_waypoints_reached


    def status_cb(self, status):
        ''' Updates status, might use this in a future feature.
        '''
        self.status = status
    

    def global_pos_cb(self, global_pos):
        ''' Gets relevant gps and altitude information
            and publishes it to our ROS network.
        '''
        self.publish_gps_altitude_feedback(global_pos)
    

    def odom_cb(self, odom):
        ''' Gets ned odometry, converts to enu odometry, and publishes them.
        '''
        # getting ned states
        pos_ned = np.float32(odom.position)
        vel_ned = np.float32(odom.velocity)
        quat_ned = np.float32(odom.q)
        ang_ned = convert_quaternion_to_euler_angles(quat_ned)
        body_ang_rate_ned = np.float32(odom.angular_velocity)
        inertial_ang_rate_ned = convert_body_to_inertial_frame(body_ang_rate_ned, ang_ned)
        odom_ned_list = [pos_ned, vel_ned, ang_ned, body_ang_rate_ned, inertial_ang_rate_ned]

        # getting enu states
        pos_enu = convert_NED_ENU_in_inertial(pos_ned)
        vel_enu = convert_NED_ENU_in_inertial(vel_ned)
        ang_enu = convert_NED_ENU_in_inertial(ang_ned)
        body_ang_rate_enu = convert_NED_ENU_in_body(body_ang_rate_ned)
        inertial_ang_rate_enu = convert_body_to_inertial_frame(body_ang_rate_enu, ang_enu)
        odom_enu_list = [pos_enu, vel_enu, ang_enu, body_ang_rate_enu, inertial_ang_rate_enu]
        
        # publish the odometries
        self.publish_ned_enu_odometry_feedback(odom_ned_list, odom_enu_list)
    

    def publish_trajectory_planner_command(self, pos_decision:np.ndarray, pos_is_ENU:bool):
        ''' Publishes position command to the trajectory planner after
            arriving at a decision. Converts between NED and ENU depending
            on the the "NED-ENU-ness" of the waypoint tracker.
        '''
        if not pos_is_ENU:
            # getting ned states
            pos_ned = np.float32(pos_decision)
            vel_ned = np.zeros(3, dtype=np.float32)
            '''NEED TO ADD IN FEATURE FOR CALCULATING YAW'''
            ang_ned = np.float32([0, 0, 0]) 
            body_ang_rate_ned = np.zeros(3, dtype=np.float32)
            inertial_ang_rate_ned = np.zeros(3, dtype=np.float32)
            # getting enu states
            pos_enu = convert_NED_ENU_in_inertial(pos_ned)
            vel_enu = np.zeros(3, dtype=np.float32)
            ang_enu = convert_NED_ENU_in_inertial(ang_ned)
            body_ang_rate_enu = np.zeros(3, dtype=np.float32)
            inertial_ang_rate_enu = np.zeros(3, dtype=np.float32)

        else:
            pos_enu = np.float32(pos_decision)
            vel_enu = np.zeros(3, dtype=np.float32)
            '''NEED TO ADD IN FEATURE FOR CALCULATING YAW'''
            ang_enu = np.float32([0, 0, 0]) 
            body_ang_rate_enu = np.zeros(3, dtype=np.float32)
            inertial_ang_rate_enu = np.zeros(3, dtype=np.float32)
            # getting enu states
            pos_ned = convert_NED_ENU_in_inertial(pos_ned)
            vel_ned = np.zeros(3, dtype=np.float32)
            ang_ned = convert_NED_ENU_in_inertial(ang_ned)
            body_ang_rate_ned = np.zeros(3, dtype=np.float32)
            inertial_ang_rate_ned = np.zeros(3, dtype=np.float32)
        
        msg = NedEnuOdometry()
        msg.position_ned = pos_ned
        msg.velocity_ned = vel_ned
        msg.euler_angle_ned = ang_ned
        msg.body_angle_rate_ned = body_ang_rate_ned
        msg.inertial_angle_rate_ned = inertial_ang_rate_ned

        msg.position_enu = pos_enu
        msg.velocity_enu = vel_enu
        msg.euler_angle_enu = ang_enu
        msg.body_angle_rate_enu = body_ang_rate_enu
        msg.inertial_angle_rate_enu = inertial_ang_rate_enu
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.traj_plan_pub.publish(msg)
        self.get_logger().info(f"Publishing NED and ENU command to trajectory planner.")


    def publish_gps_altitude_feedback(self, global_pos_msg:VehicleGlobalPosition):
        ''' Publishes latitude, longitude in degrees; altitude in meters (ASML)
        '''
        msg = GpsAltitudePosition()
        msg.lat = global_pos_msg.lat
        msg.lon = global_pos_msg.lon
        msg.alt = global_pos_msg.alt
        msg.alt_ellipsoid = global_pos_msg.alt_ellipsoid
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.gps_alt_pub.publish(msg)
        self.get_logger().info(f"Publishing GPS and altitude feedback.")


    def publish_ned_enu_odometry_feedback(self, odom_ned_list:list, odom_enu_list:list):
        ''' Unpacks the odometry lists, generated the message, and publishes it.
        '''
        pos_ned, vel_ned, ang_ned, body_ang_rate_ned, inertial_ang_rate_ned = odom_ned_list
        pos_enu, vel_enu, ang_enu, body_ang_rate_enu, inertial_ang_rate_enu = odom_enu_list

        # generating ned and enu message
        msg = NedEnuOdometry()
        msg.position_ned = pos_ned
        msg.velocity_ned = vel_ned
        msg.euler_angle_ned = ang_ned
        msg.body_angle_rate_ned = body_ang_rate_ned
        msg.inertial_angle_rate_ned = inertial_ang_rate_ned

        msg.position_enu = pos_enu
        msg.velocity_enu = vel_enu
        msg.euler_angle_enu = ang_enu
        msg.body_angle_rate_enu = body_ang_rate_enu
        msg.inertial_angle_rate_enu = inertial_ang_rate_enu
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.ned_enu_odom_pub.publish(msg)
        self.get_logger().info(f"Publishing NED and ENU feedback.")


    def publish_trajectory_setpoint(self, pos:np.ndarray, vel:np.ndarray, is_ENU):
        ''' (x, y, z)
            Publish the 3D position and velocity setpoints. 
            Set "is_ENU" to True if inputting ENU coordinates. 
        '''
        assert len(pos) == 3
        assert len(vel) == 3
        pos_f32 = np.float32(pos)
        vel_f32 = np.float32(vel)

        if is_ENU:
            pos_f32 = self.convert_NED_ENU_in_inertial(pos_f32)
            vel_f32 = self.convert_NED_ENU_in_inertial(vel_f32)

        msg = TrajectorySetpoint()
        msg.position = pos_f32
        msg.velocity = vel_f32
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.traj_setpt_pub.publish(msg)
        self.get_logger().info(f"Publishing trajectory setpoint.")


    def publish_euler_angle_setpoint(self, ang:np.ndarray, is_ENU):
        ''' (Roll, Pitch, Yaw)
            Publish 3D euler angle attitude setpoint. 
            Set "is_ENU" to True if inputting ENU coordinates. 
        '''
        assert len(ang) == 3
        ang_f32 = np.float32(ang)
        if is_ENU:
            ang_f32 = self.convert_NED_ENU_in_inertial(ang_f32)
        
        msg = VehicleAttitudeSetpoint()
        msg.roll_body = ang_f32[0]
        msg.pitch_body = ang_f32[1]
        msg.yaw_body = ang_f32[2]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.att_setpt_pub.publish(msg)
        self.get_logger().info(f"Publishing euler angle attitude setpoint.")


    def publish_euler_angle_rate_setpoint(self, ang_rate, is_ENU, is_inertial):
        ''' (Roll, Pitch, Yaw)
            Publish 3D euler angular rate setpoint in body frame.
            Set "is_inertial" to True if inputting inertial frame angle rates. 
            Set "is_ENU" to True if inputting ENU coordinate angle rates. 
        '''
        assert len(ang_rate) == 3
        ang_rate_f32 = np.float32(ang_rate)
        if is_ENU:
            ang_rate_f32 = self.convert_NED_ENU_in_body(ang_rate_f32)
        if is_inertial:
            ang_rate_f32 = self.convert_inertial_to_body_frame(ang_rate_f32, is_ENU=is_ENU)
        
        msg = VehicleRatesSetpoint()
        msg.roll = ang_rate_f32[0]
        msg.pitch = ang_rate_f32[1]
        msg.yaw = ang_rate_f32[2]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.rate_setpt_pub.publish(msg)
        self.get_logger().info(f"Publishing euler angle rates setpoint.")
    
    
    def publish_heartbeat(self) -> None:
        ''' Enables and disables the desired states to be controlled.
            Serves as the heartbeat to maintain offboard control, must keep publishing this. 
            May parameterize its options in the constructor in the future.
        '''
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_ctrl_mode_pub.publish(msg)


    def publish_command(self, command, **params) -> None:
        ''' Publish a vehicle command. 
            Setting vehicle state uses this.
        '''
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)


    def arm(self) -> None:
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')


    def disarm(self) -> None:
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')


    def engage_offboard_mode(self) -> None:
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")


    def land(self) -> None:
        self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")


    """
    def timer_cb(self) -> None:
        '''Callback function for the timer.'''
        self.publish_heartbeat()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.vehicle_local_position.z > self.takeoff_height and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

        elif self.vehicle_local_position.z <= self.takeoff_height:
            self.land()
            exit(0)

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
    """


def main(args=None):
    print('Starting commander node...')
    rclpy.init(args=args)
    node = CommanderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)