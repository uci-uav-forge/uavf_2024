#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from uavf_ros2_msgs.msg import GpsAltitudePosition, NedEnuOdometry, NedEnuWaypoint

import numpy as np
import json


class WaypointTrackerNode(Node):
    ''' Reads in the static mission waypoints in an 
        input file and keeps track of the 
        current and next waypoint to go to.
        Initialize with the acceptable error between
        current position and waypoint in terms of meters.
    '''

    def __init__(self, waypoint_list:list, epsilon=10.0):
        super().__init__('waypoint_tracker_node')

        # Configure QoS profile according to PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.ned_enu_odom_sub = self.create_subscription(
            NedEnuOdometry, '/commander/out/ned_enu_odometry', self.odom_cb, qos_profile)
        self.waypoint_pub = self.create_publisher(
            NedEnuWaypoint, '/commander/in/waypoint_tracker', qos_profile)
        
        self.is_ENU = True
        self.epsilon = epsilon
        self.wp_list = waypoint_list
        self.iter = 0
    

    def odom_cb(self, odom):
        ''' Gets the current position, evaluates
            the current waypoint, and publishes a waypoint.
        '''
        curr_pos_enu = np.float32(odom.position_enu)
        waypoint, all_wps_reached = self.evaluate_waypoint(curr_pos_enu)
        self.publish_waypoint(waypoint, all_wps_reached)
        

    def publish_waypoint(self, waypoint:np.ndarray, all_waypoints_reached:bool):
        ''' Builds the waypoint message and publishes it 
            to the commander node.
        '''
        msg = NedEnuWaypoint()
        msg.is_enu = self.is_ENU
        msg.all_waypoints_reached = all_waypoints_reached
        msg.position_waypoint = np.float32(waypoint)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.waypoint_pub.publish(msg)
        self.get_logger().info(f"Publishing waypoint to commander.")
        return


    def evaluate_waypoint(self, curr_pos:np.ndarray):
        ''' Compares the current position to the current waypoint
            and decides on the waypoint to send to the commander.
        '''
        curr_wp = np.float32(self.wp_list[self.iter])
        dist = np.linalg.norm(curr_wp - curr_pos)
        if dist < self.epsilon:
            self.iter += 1
        try:
            nxt_wp = np.float32(self.wp_list[self.iter])
            all_wps_reached = False
        # if there are no more waypoints to go to:
        except IndexError:
            self.destroy_subscription(self.ned_enu_odom_sub)
            self.destroy_publisher(self.waypoint_pub)
            all_wps_reached = True
        return nxt_wp, all_wps_reached

    '''
    def init_waypoint_list(self, mission_file_name):
        wps = []
        f = open(mission_file_name)
        data = json.load(f)
        for wp in data['waypoints']:
            wps += np.float32(wp)
        return wps
    '''


def main(args=None):
    wp_list = [
        [0, 0, 20], 
        [0, 10, 20],
        [10, 10, 20],
        [10, 0, 20],
        [0, 0, 20]
    ]
    print('Starting waypoint tracker node...')
    rclpy.init(args=args)
    node = WaypointTrackerNode(waypoint_list=wp_list, epsilon=10)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)