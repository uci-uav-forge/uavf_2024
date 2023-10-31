import os
import numpy as np
import cv2

#from PIL import Image

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class Camera_Projection:
  def __init__(self, intrinsics = None, resolution = (1920, 1080), img_folder = "DNG", debug = False):
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

    if self.intrinsics_matrix is None:
      '''Load the camera photos from the image folder'''
      img_directory = os.path.join(CURRENT_FILE_PATH, img_folder)
      img_directory_ls = os.listdir(img_directory)
     
      img_list = np.array([cv2.imread(os.path.join(img_directory, f_name)) for f_name in img_directory_ls])
      self.calibrate_cv2( img_list , debug)

    else:
      if debug:
        print(f'Initalized with camera intrinsics {self.intrinsics_matrix}')
        print(f'initalized with resolution {self.resolution}')


  
  def calibrate_cv2( self, imgs , debug):
    '''The unit test chessboard has the following properties:
       - 8 columns and 10 rows
       - 7 vertices where two columns meet per row
       - 9 vertices where two rows meet per column
       - 3 real coordinate axes: (x, y, z)

       the resolution is determined by the first image's width and height
       
       the exact criteria values are based on the criteria flags used in opencv documentation example
        code source: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

       The termination criteria format is a tuple of three elements:
         - Termination criteria flag: Specifies when to terminate the optimization
         - Max iterations: The maximum number of iterations allowed
         - Acceptable error (epsilon): The desired accuracy for termination '''
    
    chess_board_col_points, chess_board_row_points, real_coord_axis = 7, 9, 3
    resolution_input = (imgs[0].shape[0], imgs[0].shape[1])
    critera = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros(( chess_board_col_points * chess_board_row_points, real_coord_axis), np.float32)
    objp[ :, :2] = np.mgrid[ 0: chess_board_row_points, 0: chess_board_col_points].T.reshape( -1, 2) 
    

    '''Store real world coordinate points in objpoints and 2d image coordinate points in imgpoints '''
    objpoints = []
    imgpoints = []

    if self.resolution != resolution_input:
      if debug: 
        print(f'images input for calibration are resolution {resolution_input} do not match initalized resolution {self.resolution}')
        print(f'setting class member self.resolution to {resolution_input}')
      self.resolution = resolution_input

    for camera_photo in imgs:
      assert( self.resolution == (camera_photo.shape[0], camera_photo.shape[1]))
      gray = cv2.cvtColor(camera_photo, cv2.COLOR_BGR2GRAY)
      ret, corners = cv2.findChessboardCorners( gray, (chess_board_row_points, chess_board_col_points), None)

      '''Add the found corner of chessboard to the two lists'''
      if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix( gray, corners, ( 11, 11), (-1, -1), critera)
        imgpoints.append(corners2)

        if debug:
          cv2.drawChessboardCorners( gray, (chess_board_row_points, chess_board_col_points), corners2, ret)
          cv2.imshow( "img", gray)
          cv2.waitKey(500)

    if debug:
      cv2.destroyAllWindows()

    ret, intrinsics_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    '''Calculate the reprojection error using the returned calibration values'''
    mean_error = 0
    for i in range(len(objpoints)):
      imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], intrinsics_matrix, dist)
      error = cv2.norm( imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
      mean_error += error

    self.reprojection_error = mean_error/ len(objpoints)
    
    if debug: 
      print(f"total reprojection error: { self.reprojection_error}")
    
    assert ret > 0, f'UNSUCCESSFUL CALIBRATION, NOT SETTING INSTRINCS'

    self.intrinsics_matrix = np.around(intrinsics_matrix, decimals = 4)
    np.savetxt(os.path.join(CURRENT_FILE_PATH, "intrinsics_matrix.txt"), self.intrinsics_matrix, delimiter = ",")
    
    if debug:
      print("SUCCESSFUL CALIBRATION SETTING INSTRINICS")
      print(f'INSTRINICS MATRIX {self.intrinsics_matrix}')
    
    




if __name__ == "__main__":
  '''Testing class instance'''
  camera_initialize = Camera_Projection(debug= True)