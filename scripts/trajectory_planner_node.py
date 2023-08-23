#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from uavf_ros2_msgs.msg import NedEnuOdometry, NedEnuSetpoint
from px4_offboard_mpc.sqp_nlmpc import derive_quad_dynamics, SQP_NLMPC

import numpy as np


class TrajectoryPlannerNode(Node):
    ''' Initialize your planner in "init_planner".
        Make sure your planner has a method "next_state"
    '''

    def __init__(self, is_ENU:bool, is_inertial:bool, time_step:float):
        super().__init__('trajectory_planner_node')

        # Configure QoS profile according to PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.ned_enu_odom_sub = self.create_subscription(
            NedEnuOdometry, '/commander/out/ned_enu_odometry', self.odom_cb, qos_profile)
        self.commander_sub = self.create_subscription(
            NedEnuOdometry, '/commander/out/trajectory_planner_command', self.commander_cb, qos_profile)
        self.ned_enu_setpt_pub = self.create_publisher(
            NedEnuSetpoint, '/commander/in/trajectory_planner_setpoint', qos_profile)
        
        self.planner = self.init_planner(time_step)
        self.is_ENU = is_ENU
        self.is_inertial = is_inertial
        self.time_step = time_step  # runtime for 1 mpc loop in seconds
        self.time_tracker = self.get_clock().now().nanoseconds 

        # the state vector consists of: x,y,z, roll,pitch,yaw, 
        # xdot,ydot,dot, rolldot, pitchdot, yawdot
        self.curr_state = np.zeros(12, dtype=np.float32)


    def init_planner(self, time_step):
        ''' Any type of planner can be inserted into "init_planner".
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

        # initializing mpc
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
        self.curr_state = self.get_state_from_odometry(ned_enu_odom)
    

    def commander_cb(self, ned_enu_setpt):
        ''' Takes in the appropriate reference frame for the
            setpoint, runs the planner, and publishes
            to the commander. Loops once every dt seconds.
        '''
        now = self.get_clock().now().nanoseconds 
        dt = (now - self.time_tracker) / 10**9

        if dt >= self.time_step:
            setpoint = self.get_state_from_odometry(ned_enu_setpt)
            next_state = self.planner.get_next_state(
                x0=self.curr_state, x_set=setpoint, timer=True)
            self.publish_planner_setpoint(next_state)
            self.time_tracker = self.get_clock().now().nanoseconds 
    

    def get_state_from_odometry(self, ned_enu_msg):
        ''' Assigns the appropriate values to the 
            state vector according to the reference frame.
        '''
        state = np.zeros(12, dtype=np.float32)
        if self.is_ENU:
            state[0:3] = ned_enu_msg.position_enu
            state[3:6] = ned_enu_msg.euler_angle_enu
            state[6:9] = ned_enu_msg.velocity_enu
            if self.is_inertial:
                state[9:12] = ned_enu_msg.inertial_angle_rate_enu
            else:
                state[9:12] = ned_enu_msg.body_angle_rate_enu
        else:
            state[0:3] = ned_enu_msg.position_ned
            state[3:6] = ned_enu_msg.euler_angle_ned
            state[6:9] = ned_enu_msg.velocity_ned
            if self.is_inertial:
                state[9:12] = ned_enu_msg.inertial_angle_rate_ned
            else:
                state[9:12] = ned_enu_msg.body_angle_rate_ned
        return state


    def publish_planner_setpoint(self, setpoint:np.ndarray):
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

        self.ned_enu_setpt_pub.publish(msg)
        self.get_logger().info(f"Publishing planner setpoint to commander.")
        

def main(args=None):
    is_ENU = True
    is_inertial = True
    time_step = 0.1     # seconds

    print('Starting trajectory planner node...')
    rclpy.init(args=args)
    node = TrajectoryPlannerNode(is_ENU, is_inertial, time_step)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)

