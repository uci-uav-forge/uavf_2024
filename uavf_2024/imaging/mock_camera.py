# copy-paste of camera_control.py but replacing the RTSP stream with a random image and making the sdk control functions do nothing
from pathlib import Path
import threading
import time
from siyi_sdk import SIYISTREAM,SIYISDK
from uavf_2024.imaging.imaging_types import Image, HWC
from scipy.spatial.transform import Rotation
import numpy as np
import json
from collections import deque
from bisect import bisect_left


class LogType:

    def __init__(self, image : Image, metadata : dict, timestamp : float):
        self.image = image
        self.metadata = metadata
        self.timestamp = timestamp

    def get_data(self):
        return self.image, self.metadata, self.timestamp

class LogBuffer:
    """
    Buffer for logging data implementing a Lock for multithreading. 
    """
    def __init__(self):
        self.log_data = deque(maxlen=4)
        self.lock = threading.Lock()
        
    def append(self, datum : LogType):
        with self.lock:
            self.log_data.append(datum)
        
    def get_latest(self) -> LogType | None:
        with self.lock:
            return self.log_data[-1]
        
    def pop_data(self) -> LogType | None:
        with self.lock:
            return self.log_data.popleft() if len(self.log_data) > 0 else None

class ImageBuffer:
    """
    Buffer for one Image implementing a Lock for multithreading. 
    """
    def __init__(self):
        self.image = None
        self.lock = threading.Lock()
        
    def put(self, image: Image[np.ndarray]):
        with self.lock:
            self.image = image
            
    def get_latest(self) -> Image[np.ndarray] | None:
        with self.lock:
            return self.image
    
class MetadataBuffer:
    def __init__(self):
        self._queue = deque(maxlen=128)
        self.lock = threading.Lock()
    
    def append(self, datum):
        with self.lock:
            self._queue.append(datum)
        
    def get_interpolated(self, timestamp: float):
        with self.lock:
            if self._queue[-1]['time_seconds'] < timestamp:
                return self._queue[-1]
            if self._queue[0]['time_seconds'] > timestamp:
                return self._queue[0]
            idx = bisect_left([d['time_seconds'] for d in self._queue], timestamp)
            before = self._queue[idx-1]
            after = self._queue[idx]
            proportion = (timestamp-before['time_seconds']) / (after['time_seconds']-before['time_seconds'])
            att_before = before['attitude']
            att_after = after['attitude']
            attitude = tuple([
                att_before[i] + proportion * (att_after[i] - att_before[i])
                for i in range(3)
            ])
            zoom = before['zoom'] + proportion * (after['zoom']-before['zoom'])
            return {
                'time_seconds': timestamp,
                'attitude': attitude,
                'zoom': zoom
            }

class Camera:
    def __init__(self, log_dir: str | Path | None = None):
        """
        Currently starts recording and logging as soon as constructed.
        This should be changed to after takeoff.
        """
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir:
            self._prep_log_dir(self.log_dir)
        
        
        # Buffer for taking the latest image, logging it, and returning it in get_latest_image
        self.buffer = ImageBuffer()
        self.log_buffer = LogBuffer()
        self.logging_thread: threading.Thread | None = None
        self.logging = False
        self.recording_thread: threading.Thread | None = None
        self.recording = False
        self.metadata_buffer = MetadataBuffer()
    
    @staticmethod
    def _prep_log_dir(log_dir: Path):
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        
    def set_log_dir(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
    
    def _logging_worker(self):
        while self.logging and self.log_dir:
            log_data = self.log_buffer.pop_data()
            if log_data is None:
                if not self.recording: # finish logging if recording is done
                    self.logging = False
                    break
                time.sleep(0.1)
                continue
            image, metadata, timestamp = log_data.get_data()
            image.save(self.log_dir / f"{timestamp}.jpg")
            json.dump(
                metadata, 
                open(self.log_dir / f"{timestamp}.json", 'w')
            )
            time.sleep(0.05)
        
    def _get_image(self):
        return np.random.randint(0, 255, (1080, 1920, 3)).astype(np.float32)

    def _recording_worker(self):
        """
        Worker function that continuously gets frames from the stream and puts them in the buffer as well as logging them.
        """
        while self.recording:
            try:
                img_arr = self._get_image()
                img_stamp = time.time()
                attitude_position = self.getAttitude()
                zoom = self.getZoomLevel()
                attitude_stamp = time.time()
                
                if img_arr is None:
                    time.sleep(0.1)
                    continue
            except Exception as e:
                # Waits 100ms before trying again
                print(f"Error getting frame: {e}")
                time.sleep(0.1)
                continue
            
            image = Image(img_arr, HWC)
            self.buffer.put(image)
            metadata = {
                    "attitude": attitude_position,
                    "zoom": zoom,
                    "time_seconds": attitude_stamp
                }
            self.metadata_buffer.append(metadata) # allows for interpolation of attitude and zoom
            log_data = LogType(image, metadata, img_stamp)
            self.log_buffer.append(log_data)
            time.sleep(0.1)
                
    def start_recording(self):
        """
        Currently called in __init__, but this should be changed to being called when we're in the air.
        """
        if self.recording:
            return
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording = True
        self.recording_thread.start()
        self.start_logging()
    
    def start_logging(self):
        if self.logging or not self.log_dir or not self.recording:
            return
        self.logging_thread = threading.Thread(target=self._logging_worker)
        self.logging = True
        self.logging_thread.start()

    def stop_logging(self):
        if self.logging_thread:
            self.logging = False
            self.logging_thread.join()
            self.logging_thread = None

    def stop_recording(self):
        if self.recording_thread:
            self.recording = False
            self.recording_thread.join()
            self.recording_thread = None
        
    def get_latest_image(self) -> Image[np.ndarray] | None:
        """
        Returns the latest Image (HWC) from the buffer.
        """
        # return self.buffer.get_latest()
        return Image(self._get_image(), HWC)
    
    def requestAbsolutePosition(self, yaw: float, pitch: float):
        return True
    
    def requestGimbalSpeed(self, yaw_speed: int, pitch_speed: int):
        return True

    def request_center(self):
        return True
    
    def request_down(self):
        # Points up if the camera base is already pointed up
        return True
    
    def request_autofocus(self):
        return True
    
    def setAbsoluteZoom(self, zoom_level: float):
        return True
    
    def getAttitude(self):
        ''' Returns (yaw, pitch, roll) '''
        return (0,-90,0)
    
    def getAttitudeInterpolated(self, timestamp: float, offset: float=1):
        # return self.metadata_buffer.get_interpolated(timestamp-offset)['attitude'] # offset for camera lag
        return self.getAttitude()
    
    def getAttitudeSpeed(self):
        # Returns (yaw_speed, pitch_speed, roll_speed)
        return (0,0,0)

    @staticmethod
    def focalLengthFromZoomLevel(level: int):
        assert 1<= level <=10
        return 90.9 + 1597.2 * level

    def getFocalLength(self):
        '''
            calculates focal length using linear regression based on 
            data from calibration at zoom levels 1-7 and 10, and using the 
            x focal length values, then rounding to the nearest tenth 
        '''
        zoom = 1
        return Camera.focalLengthFromZoomLevel(zoom)

    def getZoomLevel(self):
        return 1
    
    def __del__(self):
        self.disconnect()
    
    def disconnect(self):
        self.stop_recording()

    @staticmethod
    def orientation_in_world_frame(drone_rot: Rotation, cam_attitude: np.ndarray) -> Rotation:
        '''
        Returns the rotation of the camera in the world frame.
        
        `cam_attitude` needs to be (yaw, pitch, roll) in degrees, where
        yaw is rotation around the z-axis, pitch is rotation around the negative y-axis, and roll is rotation around the x-axis.

        roll might be bugged because we aren't using it nor testing it very much.
        '''
        drone_euler = drone_rot.as_euler("ZYX", degrees=True)
        drone_heading = drone_euler[0]
        gimbal_heading = cam_attitude[0]
        gimbal_pitch = cam_attitude[1]
        orientation = Rotation.from_euler("ZY", [gimbal_heading+drone_heading, -gimbal_pitch], degrees=True)
        return orientation

if __name__ == "__main__":
    cam = Camera()
    out = cam.get_latest_image()
    # matplotlib.image.imsave("sample_frame.png",out.get_array().transpose(2,1,0))
    if out != None: out.save("sample_frame.png")
    cam.disconnect()