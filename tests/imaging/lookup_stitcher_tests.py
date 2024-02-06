import unittest
import cv2 as cv
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from uavf_2024.imaging import Localizer
from uavf_2024.imaging.imaging_types import FullPrediction
import shapely.geometry

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def csv_to_np(csv_str: str, delim: str = ",", dtype: type = int):
    '''
    Parses strings like "1,2,3" or "1:2:3" into numpy array [1,2,3]
    '''
    return np.array(
        [
            dtype(x) for x in
            csv_str.split(delim)
        ]
    )

class ImageFinder:
    def __init__(self, image_folder: str):
        self.imgs = []
        self.transforms = []

        for file in os.listdir(image_folder):
            if file.endswith(".png"):
                transform_no = file.split(".")[0][5:]
                with open(f"{image_folder}/transform{transform_no}.txt", "r") as f:
                    self.transforms.append(csv_to_np(f.read(), delim=' ',dtype=float))
                img = cv.imread(f"{image_folder}/{file}")
                x_scale, y_scale ,*_ = self.transforms[-1]
                # resize image to be same aspect ratio as the x and y scales
                aspect = y_scale/x_scale
                new_size = 1000
                img = cv.resize(img, (new_size, int(new_size*aspect)))
                self.imgs.append(img)

        self.sift = cv.SIFT_create()
        self.matcher = cv.BFMatcher()

    def find_image(self, newImg: cv.Mat, cam_pose: np.ndarray) -> np.ndarray:
        CAM_FOV = 50.94
        CAM_RES = (1920, 1080)
        # find corners of newImg in world coordinates using cam_pose
        localizer = Localizer(CAM_FOV, CAM_RES)

        top_left = localizer.prediction_to_coords(
            FullPrediction(0, 0, None, None, None),
            cam_pose
        ).position

        top_right = localizer.prediction_to_coords(
            FullPrediction(CAM_RES[0], 0, None, None, None),
            cam_pose
        ).position

        bottom_left = localizer.prediction_to_coords(
            FullPrediction(0, CAM_RES[1], None, None, None),
            cam_pose
        ).position

        bottom_right = localizer.prediction_to_coords(
            FullPrediction(CAM_RES[0], CAM_RES[1], None, None, None),
            cam_pose
        ).position
        # create shapely polygon
        newImg_polygon = shapely.geometry.Polygon([
            (top_left[0], top_left[2]),
            (top_right[0], top_right[2]),
            (bottom_right[0], bottom_right[2]),
            (bottom_left[0], bottom_left[2])
        ])

        # figure out which image this overlaps with
        for img, transform in zip(self.imgs, self.transforms):
            x_scale, y_scale, y_rot, *offset = transform
            # find corners of img in world coordinates
            img_top_left = np.array([-x_scale/2,0,-y_scale/2])
            img_top_right = np.array([x_scale/2,0,-y_scale/2])
            img_bottom_left = np.array([-x_scale/2,0,y_scale/2])
            img_bottom_right = np.array([x_scale/2,0,y_scale/2])

            # find the transform from img to world
            img_rot = R.from_euler('y', y_rot, degrees=True)

            img_top_left = img_rot.apply(img_top_left) + offset
            img_top_right = img_rot.apply(img_top_right) + offset
            img_bottom_left = img_rot.apply(img_bottom_left) + offset
            img_bottom_right = img_rot.apply(img_bottom_right) + offset

            # check how much of the newImg is in the img
            img_polygon = shapely.geometry.Polygon([
                (img_top_left[0], img_top_left[2]),
                (img_top_right[0], img_top_right[2]),
                (img_bottom_right[0], img_bottom_right[2]),
                (img_bottom_left[0], img_bottom_left[2])
            ])

            overlap = img_polygon.intersection(newImg_polygon).area
            percentage = overlap/newImg_polygon.area
            print(percentage)
            
            



        # compute sfit features for newImg
        target_keypoints, target_descriptors = self.sift.detectAndCompute(newImg, None)
        for img in self.imgs:
            reference_keypoints, reference_descriptors = self.sift.detectAndCompute(img, None)
            matching = self.matcher.knnMatch(target_descriptors, reference_descriptors, k=2)
            good_matches = []
            for m,n in matching:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            # visualize matching
            img_matches = cv.drawMatches(newImg, target_keypoints, img, reference_keypoints, good_matches, None)
            cv.imwrite(f'{CURRENT_FILE_PATH}/matches.png', img_matches)

            # good_target_keypoints = [target_keypoints[m.queryIdx] for m in good_matches]
            # good_reference_keypoints = [reference_keypoints[m.trainIdx] for m in good_matches]

            
        

class LookupTests(unittest.TestCase):
    def test_lookup(self):
        finder = ImageFinder(f'{CURRENT_FILE_PATH}/imaging_data/image_lookup') 

        img_fname = f'{CURRENT_FILE_PATH}/imaging_data/3d_dataset/images/image10_-105,76,4.png'
        img = cv.imread(img_fname)
        cam_position = csv_to_np(img_fname.split(".")[0].split("_")[-1])
        rot_file = f'{CURRENT_FILE_PATH}/imaging_data/3d_dataset/images/rotation10.txt'
        with open(rot_file, 'r') as f:
            rot = csv_to_np(f.read(), dtype=float) 

        transform = finder.find_image(img, np.concatenate([cam_position, rot]))

if __name__ == "__main__":
    unittest.main()