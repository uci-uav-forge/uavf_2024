from abc import abstractmethod
from collections import deque
import logging
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable, Generic, TypeVar, NamedTuple
from geometry_msgs.msg import Point, PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from libuavf_2024.msg import UAVFPose
from rclpy.node import Node

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
    
    def to_ros(self):
        message = UAVFPose()
        message.orientation = self.rotation.as_quat()
        message.position = np.array([self.position.x, self.position.y, self.position.z])
        message.timestamp_seconds = self.time_seconds
        return message
    
    def write_to_obj(self, message: UAVFPose):
        message.orientation = self.rotation.as_quat()
        message.position = np.array([self.position.x, self.position.y, self.position.z])
        message.timestamp_seconds = self.time_seconds
    
    @staticmethod
    def from_ros(message: UAVFPose):
        quaternion = message.orientation
        position = Point()
        position.x = message.position[0]
        position.y = message.position[1]
        position.z = message.position[2]
        
        return PoseDatum(
            position = position,
            rotation = Rotation.from_quat(orientation),
            time_seconds = message.timestamp_seconds
        )


class OnceCallable():
    """
    Runs the given function only the first time it's called and do nothing for subsequent invocations.
    """
    def __init__(self, func: Callable):
        self._func = func
        self._called = False
        
    def __call__(self, *args, **kwargs) -> Any | None:
        if self._called:
            return 
        
        self._called = True
        return self._func(*args, **kwargs)


BufItemT = TypeVar("BufItemT")
class AsyncBuffer(Generic[BufItemT]):
    def __init__(self, capacity: int = 4):
        if capacity < 1:
            raise ValueError(f"Buffer capacity cannot be less than 1. Got {capacity}")
        self._capacity = capacity
        self._queue: deque[BufItemT] = deque(maxlen=self.capacity)
        
        # Lock for the whole queue. 
        self.lock = threading.Lock()
        
    @property
    def capacity(self):
        return self._capacity
    
    @property
    def count(self):
        with self.lock:
            return len(self._queue)
        
    def __bool__(self):
        return bool(self.count)
    
    def put(self, datum: BufItemT):
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
    
    def get_all(self) -> list[BufItemT]:
        """
        Returns all items in the buffer in the order of freshest first.
        
        Can be useful if we want a more refined search.
        """
        with self.lock:
            return list(reversed(self._queue))
        
    def get_all_reversed(self) -> list[BufItemT]:
        """
        Returns all items in the buffer in the order of oldest first.
        """
        with self.lock:
            return list(self._queue)


InputT = TypeVar("InputT")
class Subscriptions(Generic[InputT]):
    """
    Manages subscriptions in a thread-safe way.
    
    This class can be used in the future to subsume ROS' subscription
    functionality when we stay within Python.
    """
    def __init__(self):
        self._callbacks: dict[float, Callable[[InputT], Any]] = {}
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger()
    
    def add(self, callback: Callable[[InputT], Any]) -> Callable[[], None]:
        """
        Adds the callback to the collection of subscriptions to be called
        when there is a notification.
        
        Returns a function to unsubscribe.
        """
        subscription_id = random.random()
        
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
            for index, callback in enumerate(self._callbacks.values()):
                try:
                    self.logger.log(logging.INFO, f"Calling callback {index}")
                    callback(new_value)
                    self.logger.log(logging.INFO, "Called callback")
                except Exception as e:
                    self.logger.log(logging.INFO, f"Callback failed: {e}")


MessageT = TypeVar("MessageT")
LoggingBufferT = TypeVar("LoggingBufferT")
class RosLoggingProvider(Generic[MessageT, LoggingBufferT]):
    """
    Provider wrapping a ROS topic subscription to automatically log files as well as providing
    a getter and subscription mechanism.
    
    The only method that need to be overidden are 
    """
    
    LOGGER_INDEX = 0
    
    def __init__(
        self, 
        node_context: Node,
        logs_dir: str | os.PathLike | Path | None = None, 
        buffer_size = 64,
        logger_name: str | None = None
    ):
        """
        Parameters:
            node_context: The Node used to create subscription with ROS.
            logs_path: The parent directory to which to log.
            buffer_size: The number of world positions to keep in the buffer
                for offsetted access.
        """
        self.node = node_context
        
        # Initialize logger
        if logger_name:
            self.logger = logging.getLogger()
        else:
            self.logger = logging.getLogger("RosLoggingProvider" + str(__class__.LOGGER_INDEX))
            __class__.LOGGER_INDEX += 1
        
        self._first_value: LoggingBufferT | None = None
        
        self._subscribers: Subscriptions[LoggingBufferT] = Subscriptions()
        
        self._logs_dir = Path(logs_dir) if logs_dir else None
        if self._logs_dir:
            if not self._logs_dir.exists():
                self._logs_dir.mkdir(parents=True)
            elif not self._logs_dir.is_dir():
                raise FileExistsError(f"{self._logs_dir} exists but is not a directory")
        
        self._buffer: AsyncBuffer[LoggingBufferT] = AsyncBuffer(buffer_size)
        
        self._subscribe_to_topic(self._handle_update)
        
        self.log(f"Finished intializing RosLoggingProvider. Logging to {self._logs_dir}")
        
    def log(self, message, level = logging.INFO):
        self.logger.log(level, message)

    @abstractmethod
    def _subscribe_to_topic(self, action: Callable[[MessageT], Any]) -> None:
        """
        Abstract method for inherited class to implement to allow subscription to the topic.
        
        Do not call manually
        
        Example: 
        ```
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )
    
        self.node.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            pose_callback,
            qos_profile
        )
        ```
        """
        ...
        
    @abstractmethod
    def log_to_file(self, item: LoggingBufferT):
        """
        Abstract method to write the data to a file.
        """
        ...
        
    @abstractmethod
    def format_data(self, message: MessageT) -> LoggingBufferT:
        """
        Method converting the mssage type to the internal data representation.
        
        Can just return the item if processing is not necessary.
        """
        ...
        
    def get_logs_dir(self):
        return self._logs_dir
        
    def _handle_update(self, item: MessageT):
        self.log("Got pose in provider")
        formatted: LoggingBufferT = self.format_data(item)
        
        if not self._first_value:
            self.log("Provider got first pose!")
            self._first_value = formatted
            
        self._buffer.put(formatted)
        self.log("Put pose in buffer")
        
        self._subscribers.notify(formatted)
        self.log("Notified subscribers")
        
        
        if self._logs_dir is not None:
            self.log_to_file(formatted)
        
    def get_first_datum(self):
        """
        Get the first formatted datum received by this provider.
        """
        return self._first_value
        
    def get(self, offset: int = 0):
        """
        Gets the item at the freshness offset specified (if specified).
        Otherwise, get the freshest datum
        """
        return self._buffer.get_fresh(offset)
    
    def subscribe(self, callback: Callable[[LoggingBufferT], Any]):
        self._subscribers.add(callback)
        
    def wait_for_data(self, timeout_seconds: float = float('inf')):
        """
        Waits until the first datum is added to the buffer.
        """
        start = time.time()
        
        while self._buffer.count == 0:
            diff = time.time() - start
            self.log(f"Waiting. diff: {diff}")
            if diff >= timeout_seconds:
                raise TimeoutError("Timed out waiting for datum")
            
            time.sleep(0.1)
