from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.node import Node
import rclpy

import pandas as pd
from matplotlib import pyplot as plt

import csv
from typing import Any, Callable


class PoseSubscriber(Node):
    _QOS_PROFILE = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        depth = 1
    )
    
    def __init__(self, callback: Callable[[PoseStamped], Any]):
        super().__init__("pose_subscriber") #type: ignore
        
        self.callback = callback
        
        self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self.callback,
            __class__._QOS_PROFILE
        )
        
        rclpy.spin(self)


def capture_pose_timestamps():
    rclpy.init()
    
    with open(LOG_FILE, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["time"])
        
        def write_pose(message: PoseStamped):
            time_seconds = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
            print(f"Writing time {time_seconds}")
            writer.writerow([time_seconds])
            f.flush()
            
        subscriber = PoseSubscriber(write_pose)


def visualize_timestamps_csv():
    df = pd.read_csv(LOG_FILE)
    
    times = df["time"]
    
    first_time = times[0]
    last_time = times[len(times) - 1]
    
    print(f"first time: {first_time}")
    print(f"list time: {last_time}")
    print(f"Got {len(times)} points over {round(last_time - first_time)} seconds. Average rate {len(times) / (last_time - first_time)} points/second")
    
    plt.plot(list(range(len(times))), times)
    plt.suptitle("Pose timestamps vs. pose index")
    plt.ylabel("Pose timestamp (UNIX seconds)")
    
    plt.savefig(FIG_SAVE)

    
LOG_FILE = "poses.csv"
FIG_SAVE = "pose_times_replay.png"

if __name__ == '__main__':
    # capture_pose_timestamps()
    visualize_timestamps_csv()
   
    