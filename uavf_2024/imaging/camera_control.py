from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from siyi_sdk import SIYISTREAM,SIYISDK
from uavf_2024.imaging.imaging_types import Image, HWC
import matplotlib.image 
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
        self.log_data = deque(maxlen=128)
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
        
    def put(self, image: Image):
        with self.lock:
            self.image = image
            
    def get_latest(self) -> Image | None:
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
            

class CameraLogger:
    """
    Class encapsulating multithreaded logging of images and metadata.
    
    The thread pool will be automatically destroyed when the CameraLogger object is destroyed.
    """
    def __init__(self, log_dir: str | Path, max_threads: int = 2):
        self.log_dir = Path(log_dir)
        self._prep_log_dir(self.log_dir)
        
        self.pool = ThreadPoolExecutor(max_workers=max_threads)
        
    @staticmethod
    def _prep_log_dir(log_dir: Path):
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            
    def log_async(self, image: Image, metadata: dict, timestamp: float):
        """
        Asynchronously logs the image and metadata to the log directory.
        """
        self.pool.submit(self._log_to_file, image, metadata, timestamp)
        
    def _log_to_file(self, image: Image, metadata: dict, timestamp: float):
        image.save(self.log_dir / f"{timestamp}.jpg")
        json.dump(
            metadata, 
            open(self.log_dir / f"{timestamp}.json", 'w')
        )
    

class Camera:
    def __init__(self, log_dir: str | Path | None = None):
        """
        Currently starts recording and logging as soon as constructed.
        This should be changed to after takeoff.
        """
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir:
            self._prep_log_dir(self.log_dir)
            self.threaded_logger = CameraLogger(self.log_dir)
        
        self.cam = SIYISDK(server_ip = "192.168.144.25", port= 37260,debug=False)
        self.stream = SIYISTREAM(server_ip = "192.168.144.25", port = 8554,debug=False)
        self.stream.connect()
        self.cam.connect()
        #self.cam.requestLockMode()
        
        # Buffer for taking the latest image, logging it, and returning it in get_latest_image
        self.buffer = ImageBuffer()

        self.recording_thread: threading.Thread | None = None
        self.recording = False
        
        # Controls whether images and data are submitted to the `threaded_logger`
        self.logging = False
        
        self.metadata_buffer = MetadataBuffer()
    
    @staticmethod
    def _prep_log_dir(log_dir: Path):
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        
    def set_log_dir(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)

    def _recording_worker(self):
        """
        Worker function that continuously gets frames from the stream and puts them in the buffer as well as logging them.
        """
        while self.recording:
            try:
                img_arr = self.stream.get_frame()
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
            
            if self.logging:
                self.threaded_logger.log_async(image, metadata, img_stamp)
                
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
        self.logging = True

    def stop_logging(self):
        self.logging = False

    def stop_recording(self):
        if self.recording_thread:
            self.recording = False
            self.recording_thread.join()
            self.recording_thread = None
        
    def get_latest_image(self) -> Image | None:
        """
        Returns the latest Image (HWC) from the buffer.
        """
        return self.buffer.get_latest()
    
    def requestAbsolutePosition(self, yaw: float, pitch: float):
        return self.cam.requestAbsolutePosition(yaw, pitch)
    
    def requestGimbalSpeed(self, yaw_speed: int, pitch_speed: int):
        return self.cam.requestGimbalSpeed(yaw_speed, pitch_speed)

    def request_center(self):
        return self.cam.requestAbsolutePosition(0, 0)
    
    def request_down(self):
        # Points up if the camera base is already pointed up
        return self.cam.requestAbsolutePosition(0,-90)
    
    def request_autofocus(self):
        return self.cam.requestAutoFocus()
    
    def setAbsoluteZoom(self, zoom_level: float):
        return self.cam.setAbsoluteZoom(zoom_level)
    
    def getAttitude(self):
        ''' Returns (yaw, pitch, roll) '''
        return self.cam.getAttitude()
    
    def getAttitudeInterpolated(self, timestamp: float, offset: float=1):
        return self.metadata_buffer.get_interpolated(timestamp-offset)['attitude'] # offset for camera lag
    
    def getAttitudeSpeed(self):
        # Returns (yaw_speed, pitch_speed, roll_speed)
        return self.cam.getAttitudeSpeed()

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
        zoom = self.cam.getZoomLevel()
        return Camera.focalLengthFromZoomLevel(zoom)

    def getZoomLevel(self):
        return self.cam.getZoomLevel()
    
    def __del__(self):
        self.disconnect()
    
    def disconnect(self):
        self.stop_recording()
        self.stream.disconnect()
        self.cam.disconnect()

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