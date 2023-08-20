#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import TrajectorySetpoint, VehicleAttitudeSetpoint, VehicleRatesSetpoint,\
    OffboardControlMode,  VehicleCommand, VehicleStatus, VehicleLocalPosition, VehicleOdometry
from uavf_msgs.msg import NedEnuOdometry, CommanderOutput

from px4_offboard_mpc.ned_enu_conversions import convert_NED_ENU_in_inertial, convert_NED_ENU_in_body, \
    convert_body_to_inertial_frame, convert_inertial_to_body_frame
import numpy as np
from scipy.spatial.transform import Rotation


class PX4InterfaceNode(Node):
    ''' Numpy ROS 2 node for interfacing with PX4 Offboard Control. 
        Your programs should only directly interface with the Commander node,
        which talks to this node.
    '''

    def __init__(self):
        super().__init__('px4_interface_node')

        # Configure QoS profile according to PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


        ''' This section talks to our ROS network '''
        # publish position, velocity, angle, and angle rate feedback in desired format
        self.ned_enu_odom_pub = self.create_publisher(
            NedEnuOdometry, '/px4_interface/out/ned_enu_odometry', qos_profile)
        # subscriber that receives command
        self.commander_sub = self.create_subscription(
           CommanderOutput, '/px4_interface/in/commander_output', self.commander_cb, qos_profile)


        ''' This section talks to PX4 '''
        # subscribe to px4 status
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_cb, qos_profile)
        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.convert_odom_cb, qos_profile)

        # publish heartbeat and commands to px4
        self.offboard_ctrl_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        # publish position, velocity, angle, and angle rate setpoints to px4
        self.traj_setpt_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.att_setpt_pub = self.create_publisher(
            VehicleAttitudeSetpoint, 'fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.rate_setpt_pub = self.create_publisher(
            VehicleRatesSetpoint, 'fmu/in/vehicle_rates_setpoint', qos_profile)
        
        self.status = VehicleStatus()

        '''
        self.setpt_count = 0
        self.takeoff_height = -5.0

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_cb)
        '''


    def commander_cb(self, commander_output):
        if commander_output.disarm:
            self.disarm()
        elif commander_output.arm:
            self.arm()
        elif commander_output.land:
            self.land()
        else:
            is_ENU = commander_output.is_enu
            self.publish_trajectory_setpoint(
                commander_output.position_setpoint, commander_output.velocity_setpoint, is_ENU)
            self.publish_euler_angle_setpoint(
                commander_output.euler_angle_setpoint, is_ENU
            )
            self.publish_euler_angle_rate_setpoint(
                commander_output.euler_angle_rate_setpoint, is_ENU, commander_output.is_inertial
            )
            self.publish_offboard_control_mode()


    def status_cb(self, status):
        self.status = status
    

    def convert_odom_cb(self, odom):
        # getting ned states
        pos_ned = np.float32(odom.position)
        vel_ned = np.float32(odom.velocity)
        quat_ned = np.float32(odom.q)
        rot = Rotation.from_quat(quat_ned)
        ang_ned = np.float32(rot.as_euler('xyz'))
        body_ang_rate_ned = np.float32(odom.angular_velocity)
        inertial_ang_rate_ned = convert_body_to_inertial_frame(body_ang_rate_ned, ang_ned)

        # getting enu states
        pos_enu = convert_NED_ENU_in_inertial(pos_ned)
        vel_enu = convert_NED_ENU_in_inertial(vel_ned)
        # quat_enu
        ang_enu = convert_NED_ENU_in_inertial(ang_ned)
        body_ang_rate_enu = convert_NED_ENU_in_body(body_ang_rate_ned)
        inertial_ang_rate_enu = convert_body_to_inertial_frame(body_ang_rate_enu, ang_enu)

        # generating ned and enu message
        msg = NedEnuOdometry()
        msg.position_ned = pos_ned
        msg.velocity_ned = vel_ned
        msg.euler_angle_ned = ang_ned
        msg.body_angle_rate_ned = body_ang_rate_ned
        msg.inertial_angle_rate_ned = inertial_ang_rate_ned
        msg.quaternion_ned = quat_ned

        msg.position_enu = pos_enu
        msg.velocity_enu = vel_enu
        msg.euler_angle_enu = ang_enu
        msg.body_angle_rate_enu = body_ang_rate_enu
        msg.inertial_angle_rate_enu = inertial_ang_rate_enu
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # publishing and logging
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
        self.get_logger().info(f"Publishing position setpoints {pos_f32}")
        return
    

    def publish_quaternion_NED_setpoint(self, q_NED:np.ndarray):
        ''' Publish 4D quaternion attitude setpoint. 
        '''
        assert len(q_NED) == 4
        q_NED_f32 = np.float32(q_NED)

        msg = VehicleAttitudeSetpoint()
        msg.q_d = q_NED_f32
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.att_setpt_pub.publish(msg)
        self.get_logger().info(f"Publishing quaternion attitude setpoint {q_NED_f32}")
        return


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
        self.get_logger().info(f"Publishing euler angle attitude setpoint {ang_f32}")
        return


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
        self.get_logger().info(f"Publishing euler angle rates setpoint {ang_rate_f32}")
        return
    
    
    def publish_offboard_control_mode(self) -> None:
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
        return


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
        return


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
        self.publish_offboard_control_mode()

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
    print('Starting px4 interface node...')
    rclpy.init(args=args)
    node = PX4InterfaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)

