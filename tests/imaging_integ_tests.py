import unittest
from uavf_2024.imaging.target_tracker import TargetTracker
from uavf_2024.imaging.image_processor import ImageProcessor
from uavf_2024.imaging.color_classification import ColorClassifier
from uavf_2024.imaging.imaging_types import TargetDescription, Target3D
import os
import numpy as np
import cv2 as cv

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

SHAPES = [
 "circle",
 "cross",
 "heptagon",
 "hexagon",
 "octagon",
 "pentagon",
 "quartercircle",
 "rectangle",
 "semicircle",
 "square",
 "star",
 "trapezoid",
 "triangle",
 "person"
]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"

def csv_to_np(csv_str: str, delim: str = ","):
    '''
    Parses strings like "1,2,3" or "1:2:3" into numpy array [1,2,3]
    '''
    return np.array(
        [
            int(x) for x in
            csv_str.split(delim)
        ]
    )

class TestPipeline(unittest.TestCase):
    def test_with_sim_dataset(self):
        def latlng_to_local(latlng):
            return np.array([latlng[0], 0, latlng[1]])
        def local_to_latlng(coords):
            return coords[0], coords[2]
        target_tracker = TargetTracker(
            latlng_to_local,
            local_to_latlng,
            67,
            5312
        )
        color_classifier = ColorClassifier()
        image_processor = ImageProcessor()
        ground_truth: list[Target3D] = []

        with open(f"{CURRENT_FILE_PATH}/imaging_data/sim_dataset/labels.txt", "r") as f:
            for line in f.readlines():
                label, location_str = line.split(" ")
                location = csv_to_np(location_str)

                shape_name, alphanumeric, shape_col_rgb, letter_col_rgb = label.split(",")
                shape_probs = np.eye(13)[SHAPES.index(shape_name)]
                letter_probs = np.eye(35)[LETTERS.index(alphanumeric)]

                shape_col_rgb = csv_to_np(shape_col_rgb, ":")
                letter_col_rgb = csv_to_np(letter_col_rgb, ":")
                shape_col_probs = color_classifier.predict(shape_col_rgb)                
                letter_color_probs = color_classifier.predict(letter_col_rgb)                

                ground_truth.append(
                    Target3D(
                        location[0],
                        location[2],
                        TargetDescription(
                            shape_probs,
                            letter_probs,
                            shape_col_probs,
                            letter_color_probs
                        )
                    )
                )
        
        images_dirname = f"{CURRENT_FILE_PATH}/sim_dataset/images"
        for file_name in os.listdir(images_dirname):
            img = cv.imread(f"{images_dirname}/{file_name}")
            pose_strs = file_name.split(".")[0].split("_")[1:]
            cam_position = csv_to_np(pose_strs[0])
            cam_angles = csv_to_np(pose_strs[1])

            predictions = image_processor.process_image(img)
            for pred in predictions:
               target_tracker.update_with_new_data(pred, np.concatenate(cam_position, cam_angles)) 
        
        EPSILON = 1e-6
        scores = []
        for gt_target in ground_truth:
            closest_match = target_tracker.closest_match(gt_target.description)
            if abs(closest_match.lat-gt_target.lat)<EPSILON and abs(closest_match.lng-gt_target.lng)<EPSILON:
                scores.append(1)
            else:
                scores.append(0)

        print(f"Imaging Sim Score: {np.mean(scores)}") 






