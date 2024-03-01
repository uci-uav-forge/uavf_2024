#!/usr/bin/env python3

from uavf_2024.gnc.commander_node import CommanderNode
from geometry_msgs.msg import PoseStamped, Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import rclpy.node
from scipy.spatial.transform import Rotation as R
from typing import Callable
from uavf_2024.imaging import TargetTracker, Localizer
from libuavf_2024.srv import TakePicture
from libuavf_2024.msg import TargetDetection

class BasicDrone(rclpy.node.Node):
    def __init__(self):
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

        self.imaging_client = self.create_client(TakePicture, 'imaging_service')
        while not self.imaging_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for imaging service...')
        self.get_logger().info("Finished intializing imaging client")

        self.tracker = 

    def got_pose_cb(self, pose: PoseStamped):
        self.cur_pose = pose
        self.cur_position = pose.pose.position
        self.cur_rot = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w,])
        self.got_pose = True
        self.req = TakePicture.Request()
        res = self.send_request()
        detections: TargetDetection = res.detections
        self.get_logger().info(str(res.detections))

    def log(self, *args, **kwargs):
        self.get_logger().info(*args, **kwargs)

if __name__ == "__main__":
    rclpy.init()


    drone = BasicDrone()
    rclpy.spin(drone)