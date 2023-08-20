#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from uavf_msgs.msg import NedEnuOdometry, NedEnuSetpoint, CommanderOutput

import numpy as np


class WaypointTrackerNode(Node):
    ''' Reads in the static mission waypoints in an 
        input file and keeps track of the 
        current and next waypoint to go to.
    '''

    def __init__(self, mission_file, is_ENU:bool):
        super().__init__('commander_node')

        # Configure QoS profile according to PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.ned_enu_odom_sub = self.create_subscription(
            NedEnuOdometry, '/px4_interface/out/ned_enu_odometry', self.odom_cb, qos_profile)
        self.controller_setpt_pub = self.create_publisher(
            NedEnuSetpoint, '/commander/in/waypoint_tracker', qos_profile)
        
        # do something to generate queue or list from mission_file: self.file
        # define coordinate transform betwee gps coords in mission_file to enu and ned coords in px4 telem
        # self.curr_pos = np.zeros(3, dtype=np.float32)
        

def main(args=None):
    print('Starting waypoint tracker node...')
    rclpy.init(args=args)
    node = WaypointTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)