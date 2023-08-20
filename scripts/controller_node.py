#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_offboard_mpc.sqp_nlmpc import derive_quad_dynamics, SQP_NLMPC
from uavf_msgs.msg import NedEnuOdometry, NedEnuSetpoint

import numpy as np


class ControllerNode(Node):
    ''' Initialize your controller in "init_controller".
        Make sure your controller has a method "next_state"
    '''

    def __init__(self, is_ENU:bool, is_inertial:bool, time_step:float):
        super().__init__('controller_node')

        # Configure QoS profile according to PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.ned_enu_odom_sub = self.create_subscription(
            NedEnuOdometry, '/px4_interface/out/ned_enu_odometry', self.odom_cb, qos_profile)
        self.commander_sub = self.create_subscription(
            NedEnuOdometry, '/commander/out/ned_enu_odometry', self.commander_cb, qos_profile)
        self.controller_setpt_pub = self.create_publisher(
            NedEnuSetpoint, '/commander/in/controller_setpoint', qos_profile)
        
        self.controller = self.init_controller(time_step)
        self.is_ENU = is_ENU
        self.is_inertial = is_inertial
        self.time_step = time_step                          # runtime for 1 mpc loop in seconds
        self.time_tracker = self.get_clock().now().nanoseconds 
        self.curr_state = np.zeros(12, dtype=np.float32)
        self.setpoint = np.zeros(12, dtype=np.float32)


    def init_controller(self, time_step):
        ''' Any type of controller can be inserted into "init controller".
            The resulting object must have a "get_next_state" function 
            with arguments of initial state, setpoint, and a 
            return type that is a 12-dimensional state vector.
        '''
        # deriving dynamics model
        m = 0.027 # kg
        l = 0.040 # m
        Ixx = 2.3951 * 10**(-5)
        Iyy = 2.3951 * 10**(-5)
        Izz = 3.2347 * 10**(-5)
        kf = 0.005022
        km = 1.858 * 10**(-5)
        nl_quad_model = derive_quad_dynamics(m,l,Ixx,Iyy,Izz,kf,km)

        # initializing controller
        Q = np.diag([4,4,4, 2,2,2, 1,1,1, 1,1,1])
        R = 0.1 * np.diag([1,1,1,1] )
        mpc = SQP_NLMPC(
            nl_quad_model, Q, R, 
            time_step=time_step, num_nodes=10, u_max=10000)
        return mpc
    

    def odom_cb(self, ned_enu_odom):
        ''' Takes in the appropriate reference frame for the 
            position, angle, velocity, and angular rate states.
        '''
        if self.is_ENU:
            self.curr_state[0:3] = ned_enu_odom.position_enu
            self.curr_state[3:6] = ned_enu_odom.euler_angle_enu
            self.curr_state[6:9] = ned_enu_odom.velocity_enu
            if self.is_inertial:
                self.curr_state[9:12] = ned_enu_odom.inertial_angle_rate_enu
            else:
                self.curr_state[9:12] = ned_enu_odom.body_angle_rate_enu
        else:
            self.curr_state[0:3] = ned_enu_odom.position_ned
            self.curr_state[3:6] = ned_enu_odom.euler_angle_ned
            self.curr_state[6:9] = ned_enu_odom.velocity_ned
            if self.is_inertial:
                self.curr_state[9:12] = ned_enu_odom.inertial_angle_rate_ned
            else:
                self.curr_state[9:12] = ned_enu_odom.body_angle_rate_ned
        print(self.curr_state)
    

    def commander_cb(self, ned_enu_setpt):
        ''' Takes in the appropriate reference frame for the
            setpoint, runs the controller, and publishes
            to the commander. Loops once every dt seconds.
        '''
        now = self.get_clock().now().nanoseconds 
        dt = (now - self.time_tracker) / 10**9

        if dt >= self.time_step:
            if self.is_ENU:
                self.setpoint[0:3] = ned_enu_setpt.position_enu
                self.setpoint[3:6] = ned_enu_setpt.euler_angle_enu
                self.setpoint[6:9] = ned_enu_setpt.velocity_enu
                if self.is_inertial:
                    self.setpoint[9:12] = ned_enu_setpt.inertial_angle_rate_enu
                else:
                    self.setpoint[9:12] = ned_enu_setpt.body_angle_rate_enu
            else:
                self.setpoint[0:3] = ned_enu_setpt.position_ned
                self.setpoint[3:6] = ned_enu_setpt.euler_angle_ned
                self.setpoint[6:9] = ned_enu_setpt.velocity_ned
                if self.is_inertial:
                    self.setpoint[9:12] = ned_enu_setpt.inertial_angle_rate_ned
                else:
                    self.setpoint[9:12] = ned_enu_setpt.body_angle_rate_ned

            next_state = self.controller.get_next_state(x0=self.curr_state, x_set=self.setpoint)
            self.publish_controller_setpoint(next_state)
            self.time_tracker = self.get_clock().now().nanoseconds 
    

    def publish_controller_setpoint(self, setpoint:np.ndarray):
        ''' Assigns the next desired state vector to a setpoint message.
        '''
        assert len(setpoint) == 12
        setpoint_f32 = np.float32(setpoint)

        msg = NedEnuSetpoint()
        msg.is_enu = self.is_ENU
        msg.is_inertial = self.is_inertial
        msg.position_setpoint = setpoint_f32[0:3]
        msg.euler_angle_setpoint = setpoint_f32[3:6]
        msg.velocity_setpoint = setpoint_f32[6:9]
        msg.euler_angle_rate_setpoint = setpoint_f32[9:12]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.controller_setpt_pub.publish(msg)
        self.get_logger().info(f"Publishing controller setpoint to commander: {msg}")
        return
        

def main(args=None):
    is_ENU = True
    is_inertial = True
    time_step = 0.1     # seconds

    print('Starting controller node...')
    rclpy.init(args=args)
    node = ControllerNode(is_ENU, is_inertial, time_step)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)

