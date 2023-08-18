import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import TrajectorySetpoint, VehicleAttitudeSetpoint, VehicleRatesSetpoint,\
    OffboardControlMode,  VehicleCommand, VehicleStatus, VehicleLocalPosition, VehicleOdometry

import numpy as np
from scipy.spatial.transform import Rotation


class OffboardControlNode(Node):
    ''' Numpy ROS 2 node for interfacing with PX4 Offboard Control. 
        Use numpy vectors for the "get" and "set" methods. 
        PX4 operates in NED frames, so flag "is_ENU" as True 
        for getter and setter methods if other programs work in frames.
    '''

    def __init__(self) -> None:
        super().__init__('offboard_control_node')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # status subscriber
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_cb, qos_profile)
        # state subscriber
        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos_profile)

        # Status and command publishers
        self.offboard_ctrl_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        # State publishers
        self.traj_setpt_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.att_setpt_pub = self.create_publisher(
            VehicleAttitudeSetpoint, 'fmu/in/vehicle_attitude_setpoint', qos_profile)
        self.rate_setpt_pub = self.create_publisher(
            VehicleRatesSetpoint, 'fmu/in/vehicle_rates_setpoint', qos_profile)

        # Initialize variables
        self.status = VehicleStatus()
        self.odom = VehicleOdometry()

        '''
        self.setpt_count = 0
        self.takeoff_height = -5.0

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_cb)
        '''


    def status_cb(self, status):
        self.status = status
    

    def odom_cb(self, odom):
        self.odom = odom
    

    def convert_NED_ENU_in_inertial(self, x) -> np.ndarray:
        ''' Converts a state between NED or ENU inertial frames.
            This operation is commutative. 
        '''
        assert len(x) == 3
        new_x = np.float32(
            [x[1], x[0], -x[2]])
        return new_x
    

    def convert_NED_ENU_in_body(self, x) -> np.ndarray:
        ''' Converts a state between NED or ENU body frames.
            (More formally known as FRD or RLU body frames)
            This operation is commutative. 
        '''
        assert len(x) == 3
        new_x =  np.float32(
            [x[0], -x[1], -x[2]])
        return new_x


    def convert_body_to_inertial_frame(self, ang_rate_body, is_ENU=False) -> np.ndarray:
        ''' Converts body euler angle rates to their inertial frame counterparts.
        '''
        assert len(ang_rate_body) == 3
        ang_rate_body_f32 = np.float32(ang_rate_body)

        ang = self.get_euler_angle(is_ENU=is_ENU)
        phi, theta, psi = ang
        W_inv = np.float32([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0,               np.cos(phi),              -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)],
        ])
        ang_rate_inertial = W_inv @ ang_rate_body_f32
        return ang_rate_inertial
    

    def convert_inertial_to_body_frame(self, ang_rate_inertial, is_ENU=False) -> np.ndarray:
        ''' Converts inertial euler angle rates to their body frame counterparts.
        '''
        assert len(ang_rate_inertial) == 3
        ang_rate_inertial_f32 = np.float32(ang_rate_inertial)

        ang = self.get_euler_angle(is_ENU=is_ENU)
        phi, theta, psi = ang
        W = np.float32([
            [1,           0,             -np.sin(theta)],
            [0,  np.cos(phi), np.cos(theta)*np.sin(phi)],
            [0, -np.sin(phi), np.cos(theta)*np.cos(phi)]
        ])
        ang_rate_body = W @ ang_rate_inertial_f32
        return ang_rate_body


    def get_position(self, is_ENU=False) -> np.ndarray:
        ''' (x, y, z)
            Position feedback as a 3D cartesian vector.
            Outputs in NED inertial frame by default.
            Set "is_ENU" to True to get coordinates in ENU coordinates.
        '''
        pos = np.float32(self.odom.position)
        if is_ENU: 
            pos = self.convert_NED_ENU_in_inertial(pos)
        return pos
    

    def get_velocity(self, is_ENU=False) -> np.ndarray:
        ''' (x, y, z)
            Velocity feedback as a 3D cartesian vector.
            Outputs in NED inertial frame by default.
            Set "is_ENU" to True to get velocities in ENU coordinates.
        '''
        vel = np.float32(self.odom.velocity)
        if is_ENU: 
            vel = self.convert_NED_ENU_in_inertial(vel)
        return vel


    def get_quaternion(self) -> np.ndarray:
        ''' Attitude feedback as a 4D quaternion.
            Outputs in NED inertial frame by default.
            Will implement conversion to ENU later.
        '''
        q = np.float32(self.odom.q)
        return q
    

    def get_euler_angle(self, is_ENU=False) -> np.ndarray:
        ''' (Roll, Pitch, Yaw) 
            Attitude feedback as a 3D vector of euler angles.
            Outputs in NED inertial frame by default.
            Set "is_ENU" to True to get angles in ENU coordinates.
        '''
        q = self.get_quaternion()
        rot = Rotation.from_quat(q)
        ang = np.float32(rot.as_euler('xyz'))
        if is_ENU: 
            ang = self.convert_NED_ENU_in_inertial(ang)
        return ang


    def get_euler_angle_rate(self, is_ENU=False, is_inertial_frame=False) -> np.ndarray:
        ''' (Roll, Pitch, Yaw)
            Angular velocity feedback as a 3D vector of euler angle rates.
            Outputs in NED body frame (formally known as FRD body frame) by default.
            Set "is_inertial_frame" to True to get angle rates in the inertial frame.
            Set "is_ENU" to True to get angle rates in ENU coordinates.
        '''
        ang_rate = np.float32(self.odom.angular_velocity)
        if is_ENU:
            ang_rate = self.convert_NED_ENU_in_body(ang_rate)
        if is_inertial_frame:
            ang_rate = self.convert_body_to_inertial_frame(ang_rate, is_ENU=is_ENU)
        return ang_rate


    def set_trajectory_setpoint(self, pos:np.ndarray, vel:np.ndarray, is_ENU=False):
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
    

    def set_quaternion_NED_setpoint(self, q_NED:np.ndarray):
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


    def set_euler_angle_setpoint(self, ang:np.ndarray, is_ENU=False):
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


    def set_euler_angle_rate_setpoint(self, ang_rate, is_ENU=False, is_inertial=False):
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
    
    
    def set_offboard_control_mode(self) -> None:
        ''' Enables and disables the desired states to be controlled.
            May parameterize this in the constructor in the future.
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


    def set_command(self, command, **params) -> None:
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
        self.set_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')


    def disarm(self) -> None:
        self.set_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')


    def engage_offboard_mode(self) -> None:
        self.set_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")


    def land(self) -> None:
        self.set_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")


    """
    def timer_cb(self) -> None:
        '''Callback function for the timer.'''
        self.set_offboard_control_mode()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.vehicle_local_position.z > self.takeoff_height and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.set_position_setpoint(0.0, 0.0, self.takeoff_height)

        elif self.vehicle_local_position.z <= self.takeoff_height:
            self.land()
            exit(0)

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
    """


def main():
    '''
    The loop goes:
        get setpoint: next objective
        get state: pos_enu, vel_enu, angle_enu, ang_rate_enu
        mpc -> next_control_and_state(state, setpoint)
        set next state
    '''

    '''
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControlNode()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()
    '''


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
