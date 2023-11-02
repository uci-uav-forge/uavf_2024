#!/usr/bin/env python3

import std_msgs.msg
import mavros_msgs.msg
import mavros_msgs.srv
import rclpy
import rclpy.node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import sensor_msgs.msg
from threading import Thread
import sys

class CommanderNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('uavf_commander_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )

        # set up some clients - not all of these are used right now.

        self.global_pos_sub = self.create_subscription(
            sensor_msgs.msg.NavSatFix,
            'ap/geopose/filtered',
            self.global_pos_cb,
            qos_profile)
        self.got_pos = False

        self.arm_client = self.create_client(mavros_msgs.srv.CommandBool, 'mavros/cmd/arming')
        
        self.mode_client = self.create_client(mavros_msgs.srv.SetMode, 'mavros/set_mode')

        self.takeoff_client = self.create_client(mavros_msgs.srv.CommandTOL, 'mavros/cmd/takeoff')

        self.waypoints_client = self.create_client(mavros_msgs.srv.WaypointPush, 'mavros/mission/push')
    
    def global_pos_cb(self, global_pos):
        self.got_pos = True
        self.last_pos = global_pos

if __name__ == '__main__':
    rclpy.init()
    node = CommanderNode()

    spinner = Thread(target = rclpy.spin, args = (node,))
    

    with open(sys.argv[1]) as f:
        waypoints = [tuple(map(float, line.split(','))) for line in f]


    spinner.start()

    
    print('Pushing waypoints')

    node.waypoints_client.call(mavros_msgs.srv.WaypointPush.Request(start_index = 0,
        waypoints = 
            [
                mavros_msgs.msg.Waypoint(
                    frame = mavros_msgs.msg.Waypoint.FRAME_GLOBAL_REL_ALT,
                    command = mavros_msgs.msg.CommandCode.NAV_WAYPOINT,
                    is_current = True,
                    autocontinue = True,

                    param1 = 0.0,
                    param2 = 5.0,
                    param3 = 0.0,
                    param4 = float('NaN'),

                    x_lat = wp[0],
                    y_long = wp[1],
                    z_alt = 20.0
                )

                for wp in waypoints
            ]
    ))

    print("Sent waypoints, setting node")


    node.mode_client.call(mavros_msgs.srv.SetMode.Request( \
        base_mode = 0,
        custom_mode = 'AUTO'
    ))

    print('Set mode.')


    node.destroy_node()
    rclpy.shutdown()