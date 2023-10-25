import unittest
from uavf_2024.imaging.target_mapping.camera_projection import Camera_Projection, CURRENT_FILE_PATH
import numpy as np
import cv2, os
import math

def resizing_photos(batch_name, re_shape_resolution:tuple):

        batch_dir_path = os.path.join(CURRENT_FILE_PATH, batch_name)
        batch_img_lst = os.listdir(batch_dir_path)

        resolution = cv2.imread(os.path.join(batch_dir_path, batch_img_lst[0])).shape
        

        for f_name in batch_img_lst:
            assert ".png" in f_name, "The filename does not contain '.png'"
            img = cv2.imread( os.path.join(batch_dir_path, f_name))
            img_r = cv2.resize( img, re_shape_resolution)
            cv2.imwrite( os.path.join(batch_dir_path, f_name),img_r)
class TestTargetTracker(unittest.TestCase):

    def test_with_one_batch(self):
        #Testing on DNG photos saved as PNG

        img_directory = os.path.join(CURRENT_FILE_PATH, 'DNG_1')
        img_directory_ls = os.listdir(img_directory)
        resolution = cv2.imread(os.path.join(img_directory, img_directory_ls[0])).shape

        re_shape_resolution = (resolution[1], resolution[0])

        for f_name in img_directory_ls:
            assert ".png" in f_name, "The filename does not contain '.png'"

            img = cv2.imread( os.path.join(img_directory, f_name))
            img_r = cv2.resize( img, re_shape_resolution)
            cv2.imwrite(os.path.join(img_directory, f_name),img_r)


        projection_results = Camera_Projection(batch = 'DNG_1')

        assert ( projection_results.reprojection_error < 0.1)


    def test_with_identical_batches(self):
        #Testing on two directories with same photo files
        batch_names = ["DNG", "PNG"]
        
        batch_1_dir_path = os.path.join(CURRENT_FILE_PATH, batch_names[0])
        batch_1_img_lst = os.listdir(batch_1_dir_path)

        resolution = cv2.imread(os.path.join(batch_1_dir_path, batch_1_img_lst[0])).shape

        re_shape_resolution = (resolution[1], resolution[0])

        [resizing_photos(batch_label, re_shape_resolution) for batch_label in batch_names]

        batch_1_results = Camera_Projection(batch = batch_names[0])
        batch_2_results = Camera_Projection(batch = batch_names[1])

        assert( np.all(batch_1_results.intrinsics_matrix == batch_2_results.intrinsics_matrix ) )
        assert( batch_1_results.reprojection_error < 0.1 and batch_2_results.reprojection_error < 0.1)


    def test_with_diff_ext_batches(self):
        # Testing on camera photos that were created as DNG and JPG: Difference in lighting can be seen
        batch_names = ['DNG_1', 'PNG_1']

        batch_1_dir_path = os.path.join(CURRENT_FILE_PATH, batch_names[0])
        batch_1_img_lst = os.listdir(batch_1_dir_path)

        resolution = cv2.imread(os.path.join(batch_1_dir_path, batch_1_img_lst[0])).shape

        re_shape_resolution = (resolution[1], resolution[0])

        [resizing_photos(batch_label, re_shape_resolution) for batch_label in batch_names]

        batch_1_results = Camera_Projection(batch = batch_names[0])
        batch_2_results = Camera_Projection(batch = batch_names[1])
        # DNG_1 and PNG_1 both contain exact same photos, but they were saved in two different formats: (raw) DNG and JPG -> PNG
        # There is difference in the lighting of the photos between the two formats due to this

        batch_1_results = Camera_Projection(batch = batch_names[0])
        batch_2_results = Camera_Projection(batch = batch_names[1])

        result_array_mse = np.mean( (batch_1_results.intrinsics_matrix - batch_2_results.intrinsics_matrix )**2 )
        result_array_rmse = np.sqrt(result_array_mse)
        # the variance of the pixel origin coordinate is less than 1, the focal lengths however are 6-7;
        
        print ( result_array_rmse, result_array_mse )
        assert ( batch_1_results.reprojection_error < 0.1 and batch_2_results.reprojection_error < 0.1)


