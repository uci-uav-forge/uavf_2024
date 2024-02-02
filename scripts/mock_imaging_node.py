#!/usr/bin/env python3

# example usage: ros2 run libuavf_2024 mock_imaging_node.py /home/ws/libuavf_2024/uavf_2024/gnc/data/AIRDROP_BOUNDARY 12 9

# generates and mocks 5 unique targets

# Checks if targets would be in FOV and then returns "perfect" results.

# todo:
# add some noise (error in position, error in class estimates, variable delay for results)

import rclpy
from rclpy.node import Node
from libuavf_2024.msg import TargetDetection
from libuavf_2024.srv import TakePicture
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from uavf_2024.gnc.util import read_gps, convert_delta_gps_to_local_m
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import NavSatFix
import numpy as np
import random
import time


def sample_point(dropzone_bounds):
    # sample a point uniformly at random.

    return dropzone_bounds[0] \
        + random.random()*(dropzone_bounds[-1] - dropzone_bounds[0]) \
        + random.random()*(dropzone_bounds[1] - dropzone_bounds[0])

def gen_fake_targets(dropzone_bounds):
    cur_targets = []
    for _ in range(5):
        id = None
        while id == None or id in [ct[0] for ct in cur_targets]:
            id = (random.randrange(8),random.randrange(36),random.randrange(8),random.randrange(8))
        
        p = None
        while p is None or (len(cur_targets) and min(np.linalg.norm(p2 - p) for ct,p2 in cur_targets) < 5):
            p = sample_point(dropzone_bounds)
        cur_targets.append((id,p))
    return cur_targets


class MockImagingNode(Node):
    def __init__(self, args) -> None:
        super().__init__('imaging_node')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )
        self.got_pose = False
        self.world_position_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.got_pose_cb, qos_profile)
        self.got_global_pos = False
        self.global_position_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self.got_global_pos_cb, qos_profile)
        self.imaging_service = self.create_service(TakePicture, '/imaging_service', self.imaging_callback)

        self.dropzone_bounds = read_gps(args.dropzone_file)
        self.img_w_m = args.img_w_m
        self.img_h_m = args.img_h_m

    def got_pose_cb(self, pose):
        self.cur_pose = pose
        self.cur_rot = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w,]).as_rotvec()
        self.got_pose = True

    def got_global_pos_cb(self, pos):
        #Todo this feels gross - there should be a cleaner way to get home-pos through MAVROS.
        if not self.got_global_pos:
            self.got_global_pos = True
            self.home_global_pos = pos
            
            self.dropzone_bounds_mlocal = [convert_delta_gps_to_local_m((pos.latitude, pos.longitude), x) for x in self.dropzone_bounds]
            print("Dropzone bounds in local coords: ", self.dropzone_bounds_mlocal)
            
            self.targets = gen_fake_targets(self.dropzone_bounds_mlocal)
            print(self.targets)

    
    def imaging_callback(self, request, response: list[TargetDetection]):
        response.detections = [
            
        ]

        cur_xy = np.array([self.cur_pose.pose.position.x,-self.cur_pose.pose.position.y])

        for target in self.targets:
            delta = target[1] - cur_xy
            heading = np.array([np.cos(self.cur_rot[-1]), np.sin(self.cur_rot[-1])])

            amt_fwd = np.dot(heading, delta)

            amt_side = (np.linalg.norm(delta)**2 - amt_fwd**2)**0.5

            print(cur_xy, target[1], delta, amt_fwd, amt_side)

            if abs(amt_fwd) < self.img_w_m/2 and abs(amt_side) < self.img_h_m/2:
                print("target hit!")
                response.detections.append(TargetDetection(
                    timestamp = int(1000*time.time()),
                    x = target[1][0],
                    y = target[1][1],
                    z = -1.0, # why z????
                    shape_conf = [int(i == target[0][0]) for i in range(8)],
                    letter_conf = [int(i == target[0][1]) for i in range(36)],
                    shape_color_conf = [int(i == target[0][2]) for i in range(8)],
                    letter_color_conf = [int(i == target[0][3]) for i in range(8)],
                ))
        return response

def main(args=None) -> None:
    ap = argparse.ArgumentParser('mock_imaging_node.py')
    ap.add_argument('dropzone_file')
    ap.add_argument('img_w_m', type = float)
    ap.add_argument('img_h_m', type = float)
    rclpy.init(args=args)
    node = MockImagingNode(ap.parse_args())
    print("spin spin")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()