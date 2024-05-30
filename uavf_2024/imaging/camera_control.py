from siyi_sdk import SIYISTREAM,SIYISDK
from uavf_2024.imaging.imaging_types import Image, HWC
import matplotlib.image 
from scipy.spatial.transform import Rotation
import numpy as np

class Camera:
    def __init__(self):
        self.cam = SIYISDK(server_ip = "192.168.144.25", port= 37260,debug=False)
        self.stream = SIYISTREAM(server_ip = "192.168.144.25", port = 8554,debug=False)
        self.stream.connect()
        self.cam.connect()
        #self.cam.requestLockMode()

        
    def take_picture(self) -> Image:
        '''
        Returns picture as ndarray with shape (height,width,3)
        '''
        pic = self.stream.get_frame()
        return Image(pic, HWC)
        # return np.random.rand(3, 3840, 2160)
    
    def requestAbsolutePosition(self, yaw: float, pitch: float):
        return self.cam.requestAbsolutePosition(yaw, pitch)
    
    def requestGimbalSpeed(self, yaw_speed: float, pitch_speed: float):
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
    out = cam.take_picture()
    matplotlib.image.imsave("sample_frame.png",out.get_array().transpose(2,1,0))
    cam.disconnect()