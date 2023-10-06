import os
import numpy as np
import cv2

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class Camera_Projection:
  def __init__(self, intrinsics = None, resolution = (1920, 1080):
    """initalizes camera projection model based on camera instrinics
  
    Parameters:
    instrinics (np.ndarray): matrix of shape (3,4) representing the camera instrinics matrix
              [[f_x, 0, o_y],
               [0, f_y, o_y],
               [0,   0,  1 ]]
  
    The values fx and fy are the pixel focal length, and are identical for square pixels. The values ox and oy are the offsets of the principal point from the top-left corner of the image frame. All values are expressed in pixels.
  
    """
    assert isinstance(intrinsics, np.ndarray) or isinstance(intrinsics, None)s, f'input for instrinsics is type {type(intrinsics)} when it should be np.ndarray'
    assert instrinics.shape == (3, 3), f'shape of instrinics matrix is {instrinics.shape} when it should be (3,3)'
    
    self.intrinsics_matrix = intrinsics
    self.resolution = resolution

    if self.intrinsics_matrix == None:
      print('Initalized with no camera intrinsics, use class method calibrate_cv2(imgs = [imgs_chessboard]) to preform opencv camera calibration for camera intrinsics')
    else:
      print(f'Initalized with camera intrinsics {self.instrinsics_matrix}')
    print(f'initalized with resolution {self.resolution}')

  def calibrate_cv2(imgs : list[imgs]):

    resolution_input = (imgs[0].shape(0), imgs[0].shape(1))
    if self.resolution != resolution_input:
      print(f'images input for calibration are resolution {resolution_input} do not match initalized resolution {self.resolution}')
      print(f'setting class member self.resolution to {resolution_input}')

    '''
    set up code to run this function
    ret, intrinsics_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    '''

    ret = 0 #comment this code out after you have real return value from cv2 function

    if ret <= 0:
      print('UNSUCCESSFUL CALIBRATION, NOT SETTING INSTRINICS')
    else:
      print('SUCCESSFUL CALIBRATION SETTING INSTRINICS")
      #self.instrinics = intrinsics_matrix
      #print(f'INSTRINICS MATRIX {self.instrinics}') #uncomment these afterwards

    

    
