#!/usr/bin/env python3

from uavf_2024.gnc.commander_node import CommanderNode
from geometry_msgs.msg import PoseStamped, Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import rclpy.node
from scipy.spatial.transform import Rotation as R
from typing import Callable
from uavf_2024.imaging import Tracker, Localizer
from libuavf_2024.srv import TakePicture

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

        self.imaging_client = self.create_client(TakePicture, 'imaging_service')
        while not self.imaging_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for imaging service...')
        self.get_logger().info("Finished intializing imaging client")

    def got_pose_cb(self, pose: PoseStamped):
        self.cur_pose = pose
        self.cur_position = pose.pose.position
        self.cur_rot = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w,])
        self.got_pose = True
        self.req = TakePicture.Request()
        res = self.send_request()
        self.get_logger().info(str(res.detections))
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