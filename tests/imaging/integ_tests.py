import unittest
from uavf_2024.imaging.localizer import Localizer
from uavf_2024.imaging.image_processor import ImageProcessor
from uavf_2024.imaging.color_classification import ColorClassifier
from uavf_2024.imaging.imaging_types import HWC, Image, TargetDescription, Target3D, COLORS, SHAPES, LETTERS
from uavf_2024.imaging.utils import calc_match_score
import os
import numpy as np
import shutil

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def stringify_target_description(desc: TargetDescription):
    return f"{COLORS[np.argmax(desc.shape_col_probs)]} {SHAPES[np.argmax(desc.shape_probs)]}, {COLORS[np.argmax(desc.letter_col_probs)]} {LETTERS[np.argmax(desc.letter_probs)]}"

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
    def test_with_sim_dataset(self, verbose: bool = False):
        # VFOV = 67.6 degrees
        # HFOV = 2*arctan(16/9*tan(67.6/2)) = 99.9 degrees
        target_localizer = Localizer(
            99.9,
            (5312, 2988)
        )
        color_classifier = ColorClassifier()
        debug_output_folder = f"{CURRENT_FILE_PATH}/imaging_data/visualizations/integ_test"
        if os.path.exists(debug_output_folder):
            shutil.rmtree(debug_output_folder)
        image_processor = ImageProcessor(debug_output_folder)
        ground_truth: list[Target3D] = []

        with open(f"{CURRENT_FILE_PATH}/imaging_data/sim_dataset/labels.txt", "r") as f:
            for line in f.readlines():
                label, location_str = line.split(" ")
                location = csv_to_np(location_str)

                shape_name, alphanumeric, shape_col_rgb, letter_col_rgb = label.split(",")
                shape_probs = np.eye(13)[SHAPES.index(shape_name)]
                letter_probs = np.eye(36)[LETTERS.index(alphanumeric)]

                shape_col_rgb = csv_to_np(shape_col_rgb, ":")
                letter_col_rgb = csv_to_np(letter_col_rgb, ":")
                shape_col_probs = color_classifier.predict(shape_col_rgb)                
                letter_color_probs = color_classifier.predict(letter_col_rgb)                

                ground_truth.append(
                    Target3D(
                        location,
                        TargetDescription(
                            shape_probs,
                            letter_probs,
                            shape_col_probs,
                            letter_color_probs
                        )
                    )
                )
        
        images_dirname = f"{CURRENT_FILE_PATH}/imaging_data/sim_dataset/images"
        predictions_3d: list[Target3D] =  []
        for file_name in os.listdir(images_dirname):
            img = Image.from_file(f"{images_dirname}/{file_name}")
            pose_strs = file_name.split(".")[0].split("_")[1:]
            cam_position = csv_to_np(pose_strs[0])
            cam_angles = csv_to_np(pose_strs[1])

            predictions = image_processor.process_image(img)
            for pred in predictions:
               predictions_3d.append(target_localizer.prediction_to_coords(pred, np.concatenate([cam_position, cam_angles])))
        
        EPSILON = 1 
        scores = []
        for gt_target in ground_truth:
            physically_closest_match = min(predictions_3d, key=lambda pred: np.linalg.norm(pred.position-gt_target.position))
            closest_match = max(predictions_3d, key=lambda pred: calc_match_score(pred.description, gt_target.description))
            is_close_enough = np.linalg.norm(closest_match.position-gt_target.position) < EPSILON
            if verbose:
                print(f"Closest Match for {stringify_target_description(gt_target.description)}:")
                print(f"\tClosest detection distance: {np.linalg.norm(physically_closest_match.position-gt_target.position)}")
                print(f"\tClosest detection descriptor score: {calc_match_score(physically_closest_match.description, gt_target.description)}")
                print(f"\tClosest detection descriptor: {physically_closest_match.description}")
                print(f"\tHighest descriptor score: {calc_match_score(closest_match.description, gt_target.description)}")
                print(f"\tHighest match descriptor: {closest_match.description}")
                print(f"\tHigh score match distance: {np.linalg.norm(closest_match.position-gt_target.position)}")
                print(f"\tClose enough? {is_close_enough}")
            scores.append(int(is_close_enough))

        print(f"Imaging Sim Score: {np.sum(scores)}/{len(scores)}") 

if __name__ == "__main__":
    tests = TestPipeline()
    tests.test_with_sim_dataset()