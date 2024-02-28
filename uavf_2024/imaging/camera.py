import numpy as np
from time import sleep
from siyi_sdk import SIYISTREAM,SIYISDK
from uavf_2024.imaging.imaging_types import Image, HWC
import matplotlib.image 

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

    def getFocalLength(self):
        '''
            calculates focal length linear regression from doing 
            calibration at zoom levels 1-7 and 10, and using the 
            x focal length values, then rounding to the nearest tenth 
        '''
        zoom = self.cam.getZoomLevel()
        return 90.9 + 1597.2 * zoom
    
    def disconnect(self):
        self.stream.disconnect()
        self.cam.disconnect()

if __name__ == "__main__":
    cam = Camera()
    out = cam.take_picture()
    matplotlib.image.imsave("sample_frame.png",out.get_array().transpose(2,1,0))
    cam.disconnect()