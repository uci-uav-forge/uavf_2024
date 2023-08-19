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

    def __init__(self, is_ENU:bool, time_step:float):
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
        self.setpoint_pub = self.create_publisher(
            NedEnuSetpoint, '/controller/out/ned_enu_setpoint', qos_profile)
        
        self.controller = self.init_controller(time_step)
        self.is_ENU = is_ENU
        self.time_step = time_step                          # runtime for 1 mpc loop in seconds
        self.time_tracker = self.get_clock().now().nanoseconds * 10**9  # in seconds


    def init_controller(self, time_step):
        ''' This method is intentionally kept vague. 
            The idea is that if desired, any type of 
            controller can be inserted into the generic
            "init_controller"and "next_state" functions.
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
            time_step=time_step, num_nodes=10, u_max=10000
        )
        return mpc
    

    def odom_cb(self, ned_enu_odom):
        return


def main(args=None):
    '''
    Startup:
        send offboard control mode msg 10 times,
        engage offboard mode,
        arm

    The loop goes:
        get setpoint: next objective
        get state: pos_enu, vel_enu, angle_enu, ang_rate_enu
        mpc -> next_control_and_state(state, setpoint)
        set next state
    '''

    is_ENU = True
    time_step = 0.1

    print('Starting controller node...')
    rclpy.init(args=args)
    node = ControllerNode(is_ENU, time_step)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)


        