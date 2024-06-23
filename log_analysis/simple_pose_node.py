from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import MagneticField
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import rclpy

from typing import Any, Callable
import time

class GenericSubscriber():
    def __init__(self, node : Node, topic, t_type, callback: Callable, cb_group):        
        self.callback = callback

        self.profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        depth = 1
        )
        node.create_subscription(
            t_type,
            topic,
            self.callback,
            self.profile,
            callback_group=cb_group
        )

class main_node(Node):

    def __init__(self):
        super().__init__("test")
        cb_group1 = MutuallyExclusiveCallbackGroup()
        cb_group2 = MutuallyExclusiveCallbackGroup()
        test = GenericSubscriber(self, "/mavros/local_position/pose", PoseStamped, self.pose_cb, cb_group1)
        test2 = GenericSubscriber(self, "/mavros/imu/mag", MagneticField, self.global_pose_cb, cb_group2)
    
    def run(self):
        rclpy.spin(self)

    def pose_cb(self, pose : PoseStamped):
        print(f"Local: {pose.header.stamp.sec} {pose.header.stamp.nanosec}")
        starttime = time.time()
        while time.time()-starttime < 0.1:
            self.executor.spin_once(timeout_sec=0)
            pass

    def global_pose_cb(self, pose : MagneticField):
        #print(f"Global: {pose.header.stamp.sec}")
        pass

def main():
    rclpy.init()
    node = main_node()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    # node.run()

if __name__ == '__main__':
    main()
   
    