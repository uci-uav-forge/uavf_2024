import os
from queue import Queue
from pathlib import Path
import threading
import time
from typing import Any, Callable, Generic, NamedTuple, TypeVar
from collections import deque
import csv
from uuid import UUID

from scipy.spatial.transform import Rotation

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Point


class PoseDatum(NamedTuple):
    """
    Our representation of the pose data from the Cube.
    """
    position: Point
    rotation: Rotation
    time_seconds: float


class _PoseBuffer:
    def __init__(self, capacity: int = 4):
        if capacity < 1:
            raise ValueError(f"Buffer capacity cannot be less than 1. Got {capacity}")
        self._capacity = capacity
        self._queue: deque[PoseDatum] = deque(maxlen=self.capacity)
        
        # Lock for the whole queue. 
        # Not necessary if only the front is popped because that is thread-safe. 
        self.lock = threading.Lock()
        
    @property
    def capacity(self):
        return self._capacity
    
    @property
    def count(self):
        return len(self._queue)
        
    def __bool__(self):
        return bool(self.count)
    
    def put(self, datum: PoseDatum):
        with self.lock:
            # If the queue is too long, it'll automatically discard 
            # the item at the other end.
            self._queue.append(datum)
        
    def get_fresh(self, offset: int = 0):
        """
        Gets the item at the freshness offset specified (if specified).
        Otherwise, get the freshest datum
        """
        if offset < 0:
            raise ValueError(f"Offset cannot be less than 0. Got {offset}")
        
        with self.lock:
            return self._queue[-(offset + 1)]
    
    def get_all(self) -> list[PoseDatum]:
        """
        Returns all items in the buffer in the order of freshest first.
        
        Can be useful if we want a more refined search.
        """
        with self.lock:
            return list(reversed(self._queue))
        

InputT = TypeVar("InputT")
class Subscriptions(Generic[InputT]):
    """
    Manages subscriptions in a thread-safe way.
    
    This class can be used in the future to subsume ROS' subscription
    functionality when we stay within Python.
    """
    def __init__(self):
        self._callbacks: dict[UUID, Callable[[InputT], Any]] = {}
        self.lock = threading.Lock()
    
    def add(self, callback: Callable[[InputT], Any]) -> Callable[[], None]:
        """
        Adds the callback to the collection of subscriptions to be called
        when there is a notification.
        
        Returns a function to unsubscribe.
        """
        subscription_id = UUID()
        
        with self.lock:
            def unsubscribe():
                del self._callbacks[subscription_id]
            
            self._callbacks[subscription_id] = callback
        
        return unsubscribe

    def notify(self, new_value: InputT):
        """
        Calls all of the callbacks with the new value.
        
        Locks so that subscriptions will have to wait after a round of notifications.
        """
        with self.lock:
            for callback in self._callbacks.values():
                callback(new_value)


class PoseProvider:
    """
    Logs and buffers the world position for reading.
    
    Provides a method to subscribe to changes as well.
    """ 
    def __init__(
        self, 
        logs_path: str | os.PathLike | Path | None = None, 
        buffer_size = 5
    ):
        """
        Parameters:
            logs_path: The parent directory to which to log.
            buffer_size: The number of world positions to keep in the buffer
                for offsetted access.
        """
        self.logs_path = Path(logs_path) if logs_path else None
        
        if self.logs_path:
            if not self.logs_path.exists():
                self.logs_path.mkdir(parents=True)
            elif not self.logs_path.is_dir():
                raise FileExistsError(f"{self.logs_path} exists but is not a directory")
        
        # daemon=True allows the thread to be terminated when the class instance is deleted.
        self._logger_thread = threading.Thread(target=self._log_task, daemon=True)
        self._logs_queue: Queue[PoseDatum] = Queue()
        
        self._buffer = _PoseBuffer(buffer_size)
        
        # This is encapsulated so as not to expose Node's interface
        # The type error is just from rclpy and is unavoidable
        self._world_pos_node = Node("world_pos_node") # type: ignore
        
        # Initialize Quality-of-Service profile for subscription
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        self._world_pos_node.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.handle_pose_update,
            qos_profile
        )
        
        self._subscribers: Subscriptions[PoseDatum] = Subscriptions()
        
    def handle_pose_update(self, pose: PoseStamped) -> None:
        quaternion = pose.pose.orientation
        formatted = PoseDatum(
            position = pose.pose.position,
            rotation = Rotation.from_quat(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w]),
            time_seconds = pose.header.stamp.sec + pose.header.stamp.nanosec / 1e9
        )
        self._buffer.put(formatted)
        self._subscribers.notify(formatted)
        
    def get(self, offset: int = 0):
        """
        Gets the item at the freshness offset specified (if specified).
        Otherwise, get the freshest datum
        """
        return self._buffer.get_fresh(offset)
    
    def _log_task(self):
        """
        Task to pull from the _logs_queue and write the pose to file.
        """
        if self.logs_path is None:
            return
        
        with open(self.logs_path / "poses.csv") as f:
            writer = csv.writer(f)
            writer.writerow(PoseDatum._fields)
            
            while True:
                datum = self._logs_queue.get()
                writer.writerow(datum)
    
    def subscribe(self, callback: Callable[[PoseDatum], Any]):
        self._subscribers.add(callback)
        
    def wait_for_pose(self, timeout_seconds: float = float('inf')):
        """
        Waits until the first pose is added to the buffer.
        """
        start = time.time()
        
        while self._buffer.count == 0:
            if time.time() - start >= timeout_seconds:
                raise TimeoutError("Timed out waiting for pose")
            
            time.sleep(0.1)

