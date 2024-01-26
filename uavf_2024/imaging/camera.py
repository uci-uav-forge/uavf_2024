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

        
    def take_picture(self) -> Image:
        '''
        Returns picture as ndarray with shape (height,width,3)
        '''
        pic = self.stream.get_frame()
        return Image(pic, HWC)
        # return np.random.rand(3, 3840, 2160)

    def request_center(self):
        return self.cam.requestAbsolutePosition(0, 0)
    
    def request_autofocus(self):
        return self.cam.requestAutoFocus()
    
    def setAbsoluteZoom(self, zoom_level: float):
        return self.cam.setAbsoluteZoom(1)
    
    def disconnect(self):
        self.stream.disconnect()
        self.cam.disconnect()

if __name__ == "__main__":
    cam = Camera()
    out = cam.take_picture()
    matplotlib.image.imsave("sample_frame.png",out.get_array().transpose(2,1,0))
    cam.disconnect()