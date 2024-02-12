#!/usr/bin/env python3

from uavf_2024.gnc.commander_node import CommanderNode
import mavros_msgs.msg
from geometry_msgs.msg import PoseStamped, Point
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import rclpy.node
from scipy.spatial.transform import Rotation as R
from typing import Callable

class BasicDrone(rclpy.node.Node):
    def __init__(self, custom_pose_callback: Callable):
        super().__init__('uavf_basic_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )

        self.world_position_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.got_pose_cb,
            qos_profile)

        self.got_pose = False
        self.custom_pose_callback = custom_pose_callback

    def got_pose_cb(self, pose: PoseStamped):
        self.cur_pose = pose
        self.cur_position = pose.pose.position
        self.cur_rot = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w,])
        self.got_pose = True
        self.custom_pose_callback(self.cur_position, self.cur_rot, self.log)

    def log(self, *args, **kwargs):
        self.get_logger().info(*args, **kwargs)

if __name__ == "__main__":
    rclpy.init()

    def my_pose_callback(position: Point, rotation: R, log_fn: Callable):
        log_fn(str(position))
        log_fn(str(rotation.as_rotvec()))

    drone = BasicDrone(my_pose_callback)
    rclpy.spin(drone)