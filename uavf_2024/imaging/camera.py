import numpy as np
from time import sleep
from siyi_sdk import SIYISTREAM,SIYISDK
import matplotlib.image 

class Camera:
    def __init__(self):
        self.cam = SIYISDK(server_ip = "192.168.144.25", port= 37260,debug=False)
        self.stream = SIYISTREAM(server_ip = "192.168.144.25", port = 8554,debug=False)
        self.stream.connect()
        self.cam.connect()

        
    def take_picture(self) -> np.ndarray:
        '''
        Returns picture as ndarray with shape (3, width, height)
        '''
        pic = self.stream.get_frame()
        return pic
        # return np.random.rand(3, 3840, 2160)
    
    def disconnect(self):
        self.stream.disconnect()
        self.cam.disconnect()

if __name__ == "__main__":
    cam = Camera()
    out = cam.take_picture()
    matplotlib.image.imsave("sample_frame.png",out)
    cam.disconnect()