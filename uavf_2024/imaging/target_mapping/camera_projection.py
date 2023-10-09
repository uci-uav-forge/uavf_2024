import os
import numpy as np
import cv2

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class Camera_Projection:
  def __init__(self, intrinsics = None, resolution = (1920, 1080)):
    """initalizes camera projection model based on camera instrinics
  
    Parameters:
    instrinics (np.ndarray): matrix of shape (3,4) representing the camera instrinics matrix
              [[f_x, 0, o_y],
               [0, f_y, o_y],
               [0,   0,  1 ]]
  
    The values fx and fy are the pixel focal length, and are identical for square pixels. The values ox and oy are the offsets of the principal point from the top-left corner of the image frame. All values are expressed in pixels.
  
    """
    assert isinstance(intrinsics, np.ndarray) or isinstance(intrinsics, type(None)), f'input for instrinsics is type {type(intrinsics)} when it should be np.ndarray'
    
    
    self.intrinsics_matrix = intrinsics
    self.resolution = resolution

    if self.intrinsics_matrix == None:
      print('Initalized with no camera intrinsics, use class method calibrate_cv2(imgs = [imgs_chessboard]) to preform opencv camera calibration for camera intrinsics')
    
      '''uploading camera photos from img folder'''
      img_directory = os.path.join(CURRENT_FILE_PATH, "img")
      img_directory_ls = os.listdir(img_directory)
      img_list = []
      img_height, img_width = 2291, 1718  

      for f_name in img_directory_ls:

        img = cv2.imread( os.path.join(img_directory, f_name))
        imgS = cv2.resize(img, (img_width, img_height)) 
        ''' the photos were resized to the smallest resolution because the camera images saved had inconsistent resolution, '''
        img_list.append(imgS)
        
      self.calibrate_cv2( img_list )


    else:
      print(f'Initalized with camera intrinsics {self.instrinsics_matrix}')
    print(f'initalized with resolution {self.resolution}')

  
  def calibrate_cv2( self, imgs : list):

    chess_board_col_points, chess_board_row_points, real_coord_axis = 7, 9, 3

    '''termination criteria'''
    critera = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros(( chess_board_col_points * chess_board_row_points, real_coord_axis), np.float32)
    objp[ :, :2] = np.mgrid[ 0: chess_board_row_points, 0: chess_board_col_points].T.reshape( -1, 2) 
    '''Adjust square size to 20 mm for real world coordinates'''

    '''objpoints is storing points in the real world coordinate system while imgpoints is storing points in the 2d image coordinate system'''
    objpoints = []
    imgpoints = []

    resolution_input = (imgs[0].shape[0], imgs[0].shape[1])
    if self.resolution != resolution_input:
      print(f'images input for calibration are resolution {resolution_input} do not match initalized resolution {self.resolution}')
      print(f'setting class member self.resolution to {resolution_input}')
      self.resolution = resolution_input

    
    for camera_photo in imgs:
      gray = cv2.cvtColor(camera_photo, cv2.COLOR_BGR2GRAY)
      ret, corners = cv2.findChessboardCorners( gray, (chess_board_row_points, chess_board_col_points), None)

      '''corner of chessboard has been found, adding point to the two lists'''
      if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix( gray, corners, ( 11, 11), (-1, -1), critera)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners( gray, (chess_board_row_points, chess_board_col_points), corners2, ret)
        cv2.imshow( "img", gray)
        cv2.waitKey(500)

    cv2.destroyAllWindows()


    '''set up code to run this function '''
    ret, intrinsics_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)



    ''' reprojection error'''
    mean_error = 0
    for i in range(len(objpoints)):
      imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], intrinsics_matrix, dist)
      error = cv2.norm( imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
      mean_error += error
    
    print(f"total reprojection error: { mean_error/ len(objpoints)}")

    if ret <= 0:
      print('UNSUCCESSFUL CALIBRATION, NOT SETTING INSTRINICS')
    else:
      print("SUCCESSFUL CALIBRATION SETTING INSTRINICS")
      self.instrinics = intrinsics_matrix
      print(f'INSTRINICS MATRIX {self.instrinics}') #uncomment these afterwards




if __name__ == "__main__":
  '''Testing class instance'''
  camera_initialize = Camera_Projection()
    

    
