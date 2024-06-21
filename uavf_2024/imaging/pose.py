from bisect import bisect_left
import json
import time
from typing import Any, Callable, NamedTuple

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, PoseStamped
from scipy.spatial.transform import Rotation, Slerp

from uavf_2024.async_utils import RosLoggingProvider


class PoseDatum(NamedTuple):
    """
    Our representation of the pose data from the Cube.
    """
    position: Point
    rotation: Rotation
    time_seconds: float
    
    def to_json(self):
        return {
            "position": [self.position.x, self.position.y, self.position.z],
            "rotation": list(self.rotation.as_quat()),
            "time_seconds": self.time_seconds
        }


QOS_PROFILE = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    depth = 1
)


# TODO: Refactor to MAVSDK
class PoseProvider(RosLoggingProvider[PoseStamped, PoseDatum]):        
    def _subscribe_to_topic(self, action: Callable[[PoseStamped], Any]) -> None:
        self.node.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            action,
            QOS_PROFILE
        )
        
    def log_to_file(self, item: PoseDatum):
        logs_dir = self.get_logs_dir()
        if logs_dir is None:
            raise ValueError("Logging directory not given in initialization.")
        
        path = logs_dir / f"{item.time_seconds:.02f}.json"
        with open(path, "w") as f:
            json.dump(item.to_json(), f)
            
    def format_data(self, message: PoseStamped) -> PoseDatum:
        quaternion = message.pose.orientation
        
        return PoseDatum(
            position = message.pose.position,
            rotation = Rotation.from_quat(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w]),
            time_seconds = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        )
        
    def _interpolate_from_buffer(self, time_seconds: float, wait: bool = False) -> PoseDatum | None:
        """
        Returns the pose datum interpolated between the two data points before and after the time given.
        
        If this interpolation is not possible because the pose is too old, returns the oldest pose.
        If this interpolation is not possible because the poses are not old enough, wait until enough data is available if wait is enabled.
            Otherwise, return the newest pose
        If there is no pose available, wait if enabled. Otherwise, return None.
        """
        if self._buffer.count == 0:
            return None
        
        data = self._buffer.get_all_reversed()
        
        closest_idx = bisect_left([d.time_seconds for d in data], time_seconds)
        
        if closest_idx == 0:
            return data[0]
        
        # Poll every 100ms
        while closest_idx == len(data):
            if not wait:
                return data[closest_idx - 1]

            time.sleep(0.1)
            data = self._buffer.get_all_reversed()
            closest_idx = bisect_left([d.time_seconds for d in data], time_seconds)
            
        pt_before = data[closest_idx - 1]
        pt_after = data[closest_idx]

        # Interpolate between the two points
        proportion = (time_seconds - pt_before.time_seconds) / (pt_after.time_seconds - pt_before.time_seconds)
        position = Point(
            x = pt_before.position.x + proportion * (pt_after.position.x - pt_before.position.x),
            y = pt_before.position.y + proportion * (pt_after.position.y - pt_before.position.y),
            z = pt_before.position.z + proportion * (pt_after.position.z - pt_before.position.z)
        )
        key_times = [pt_before.time_seconds, pt_after.time_seconds]
        rotations = Rotation.concatenate([pt_before.rotation, pt_after.rotation])
        slerp = Slerp(key_times, rotations)
        rotation = Rotation.from_quat(slerp([time_seconds]).as_quat()[0])
        # go to and from quaternion to force returned
        # object to be single rotation

        return PoseDatum(
            position = position,
            rotation = rotation,
            time_seconds = time_seconds
        )

    def get_interpolated(self, time_seconds: float) -> tuple[PoseDatum, bool]:
        '''
        Gets the pose interpolated at `time_seconds`. Blocks if we don't have enough data in the queue
        to interpolate yet, for a maximum of 5 seconds before giving up and returning the latest datum
        regardless of its timestamp

        '''
        for _ in range(50):
            interp_pose = self._interpolate_from_buffer(time_seconds, True)
            if interp_pose is not None:
                return (interp_pose, True)
            else:
                time.sleep(0.1)
        return (self._buffer.get_fresh(), False)
