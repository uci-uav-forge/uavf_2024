#!/usr/bin/env python3

from uavf_2024.gnc.commander_node import CommanderNode
from uavf_2024.gnc.util import read_gps
import mavros_msgs.msg
import mavros_msgs.srv
import rclpy
import rclpy.node
from threading import Thread
import sys



if __name__ == '__main__':
    rclpy.init()
    node = CommanderNode()

    spinner = Thread(target = rclpy.spin, args = (node,))
    

    waypoints = read_gps(sys.argv[1])


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