import unittest
from uavf_2024.imaging.target_mapping.camera_projection import Camera_Projection, CURRENT_FILE_PATH
import numpy as np
import cv2, os
import math

def resizing_photos(batch_name, mix = False, re_shape_resolution = False):
        # this is to make sure all photos are the same size within and between photo sets and to mix up the order
        batch_dir_path = os.path.join(CURRENT_FILE_PATH, batch_name)
        batch_img_lst = os.listdir(batch_dir_path)

        if re_shape_resolution is False:
            resolution = cv2.imread(os.path.join(batch_dir_path, batch_img_lst[0])).shape
            re_shape_resolution = (resolution[1], resolution[0])
        
        if mix is True:
             new_order = np.arange(len(batch_img_lst))
             new_order = np.random.permutation(new_order)

        for index, f_name in enumerate(batch_img_lst):
            assert ".png" in f_name, "The filename does not contain '.png'"

            img = cv2.imread( os.path.join(batch_dir_path, f_name))
            img_r = cv2.resize( img, re_shape_resolution)
            os.remove(os.path.join(batch_dir_path, f_name))

            if mix is True:
                f_name = str(new_order[index]) + f_name.replace(".png", "") + ".png"
            cv2.imwrite( os.path.join(batch_dir_path, f_name),img_r)

        return re_shape_resolution


class TestTargetTracker(unittest.TestCase):
    def test_with_one_batch(self):

        resolution = resizing_photos(batch_name= "DNG")
        projection_results = Camera_Projection(batch = 'DNG')

        assert ( projection_results.reprojection_error < 0.1)

    
    def test_rerun_single_batch(self):
        #Testing if matrix will be the same with the same batch mixed
        resolution = resizing_photos(batch_name= "DNG", mix = False)
        batch_1_results = Camera_Projection(batch = "DNG")
        same_resolution = resizing_photos(batch_name= "DNG", mix = True, re_shape_resolution= resolution)
        batch_2_results = Camera_Projection(batch = "DNG")

        assert( resolution == same_resolution)
        assert( batch_1_results.reprojection_error < 0.1 and batch_2_results.reprojection_error < 0.1)
        assert( np.all(batch_1_results.intrinsics_matrix == batch_2_results.intrinsics_matrix ) )
        


    def test_with_diff_batches(self):
        # Testing on camera photos with same pose but different lighting/details
        resolution = resizing_photos(batch_name= "DNG")
        return_resolution = resizing_photos(batch_name= "PNG", re_shape_resolution= resolution)
        batch_1_results = Camera_Projection(batch = "DNG")
        batch_2_results = Camera_Projection(batch = "PNG")

        result_array_mse = np.mean( (batch_1_results.intrinsics_matrix - batch_2_results.intrinsics_matrix )**2 )
        result_array_rmse = np.sqrt(result_array_mse)

        assert ( result_array_mse < 10 and result_array_rmse < 5)
        assert (resolution == return_resolution)
        assert ( batch_1_results.reprojection_error < 0.1 and batch_2_results.reprojection_error < 0.1) 


    def test_with_loading_intrinsic(self):
         #test case loading matrix
         loaded_intrinsic_matrix = np.loadtxt(os.path.join(CURRENT_FILE_PATH, "intrinsics_matrix.txt"), delimiter=",")
         batch_result = Camera_Projection(intrinsics=loaded_intrinsic_matrix)
         assert( np.all(batch_result.intrinsics_matrix == loaded_intrinsic_matrix))


