import unittest
from uavf_2024.imaging.target_mapping.camera_projection import Camera_Projection, CURRENT_FILE_PATH
import numpy as np
import cv2, os
import math

class TestCameraCalibration(unittest.TestCase):
    def test_low_reprojection_error(self):

        projection_results = Camera_Projection(img_folder = 'DNG')
        assert ( projection_results.reprojection_error < 0.1)

    
    def test_rerun_single_img_folder(self):
        '''test verifies order of photos does not matter when calibrating'''
        
        original_img_results = Camera_Projection(img_folder = "DNG")
        rearranged_img_results = Camera_Projection(img_folder = "DNG_1")

        assert( original_img_results.reprojection_error < 0.1 and rearranged_img_results.reprojection_error < 0.1)
        assert( np.all(original_img_results.intrinsics_matrix == rearranged_img_results.intrinsics_matrix ) )
        


    def test_with_diff_img_folderes(self):
        '''test verifies a small mean square error and small root mean square error between two different image sets'''

        img_set_1_results = Camera_Projection(img_folder = "DNG")
        img_set_2_results = Camera_Projection(img_folder = "PNG")

        result_array_mse = np.mean( (img_set_1_results.intrinsics_matrix - img_set_2_results.intrinsics_matrix )**2 )
        result_array_rmse = np.sqrt(result_array_mse)

        assert ( result_array_mse < 10 and result_array_rmse < 5)
        assert ( img_set_1_results.reprojection_error < 0.1 and img_set_2_results.reprojection_error < 0.1) 


    def test_with_loading_intrinsic(self):
         '''test loads up stored instrinsics matrix'''

         loaded_intrinsic_matrix = np.loadtxt(os.path.join(CURRENT_FILE_PATH, "intrinsics_matrix.txt"), delimiter=",")
         img_folder_result = Camera_Projection(intrinsics=loaded_intrinsic_matrix)

         assert( np.all(img_folder_result.intrinsics_matrix == loaded_intrinsic_matrix))


