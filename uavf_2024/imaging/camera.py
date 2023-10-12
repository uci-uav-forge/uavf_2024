import numpy as np

class Camera:
    def __init__(self):
        pass
    def take_picture(self) -> np.ndarray:
        '''
        Returns picture as ndarray with shape (3, width, height)
        '''

        return np.random.rand(3, 3840, 2160)