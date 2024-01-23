import unittest
from uavf_2024.imaging.localizer import Localizer
from uavf_2024.imaging.area_coverage import AreaCoverageTracker
from uavf_2024.imaging.image_processor import ImageProcessor
from uavf_2024.imaging.tracker import TargetTracker
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
    def test_with_sim_dataset(self, verbose: bool = True):
        # VFOV = 67.6 degrees
        # HFOV = 2*arctan(16/9*tan(67.6/2)) = 99.9 degrees
        target_localizer = Localizer(
            99.9,
            (5312, 2988)
        )
        area_tracker = AreaCoverageTracker(
            99.9,
            (5312, 2988)
        )
        color_classifier = ColorClassifier()
        if verbose:
            debug_output_folder = f"{CURRENT_FILE_PATH}/imaging_data/visualizations/integ_test"
            if os.path.exists(debug_output_folder):
                shutil.rmtree(debug_output_folder)
        else:
            debug_output_folder = None
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

        tracker = TargetTracker([t.description for t in ground_truth])
        
        images_dirname = f"{CURRENT_FILE_PATH}/imaging_data/sim_dataset/images"
        predictions_3d: list[Target3D] =  []
        for file_name in sorted(os.listdir(images_dirname)):
            img = Image.from_file(f"{images_dirname}/{file_name}")
            pose_strs = file_name.split(".")[0].split("_")[1:]
            cam_position = csv_to_np(pose_strs[0])
            cam_angles = csv_to_np(pose_strs[1])

            predictions = image_processor.process_image(img)
            area_tracker.update(np.concatenate([cam_position, cam_angles]), label=file_name.split("_")[0])
            for pred in predictions:
               predictions_3d.append(target_localizer.prediction_to_coords(pred, np.concatenate([cam_position, cam_angles])))

        tracker.update(predictions_3d)

        closest_tracks = tracker.estimate_positions()

        EPSILON = 2 
        scores = []
        for gt_target, pred_track in zip(ground_truth, closest_tracks):
            is_close_enough = np.linalg.norm(pred_track.position-gt_target.position) < EPSILON
            scores.append(int(is_close_enough))
            if verbose:
                physically_closest_match = min(predictions_3d, key=lambda pred: np.linalg.norm(pred.position-gt_target.position))
                closest_match = max(predictions_3d, key=lambda pred: calc_match_score(pred.description, gt_target.description))
                print(stringify_target_description(gt_target.description))

                print(f"\tTrack distance: {np.linalg.norm(pred_track.position-gt_target.position):.3f}")
                print(f"\tDetections used in track:")
                print(f"\t\t{[detection.id for detection in pred_track.get_measurements()]}") 

                print(f"\tClose tracks (each line is one track):")
                for track in tracker.tracks:
                    if np.linalg.norm(track.position - gt_target.position) < EPSILON:
                        print(f"\t\t{[detection.id for detection in track.get_measurements()]}")

                print(f"\tClose detections:")
                print(f"\t\t{[p.id for p in filter(lambda pred: np.linalg.norm(pred.position-gt_target.position) < EPSILON, predictions_3d)]}")
                print(f"\tPhysically closest detection distance: {np.linalg.norm(physically_closest_match.position-gt_target.position):.3f}")
                print(f"\tPhysically closest detection descriptor score: {calc_match_score(physically_closest_match.description, gt_target.description)}")
                print(f"\tPhysically closest detection id: {physically_closest_match.id}")
                print(f"\tHighest descriptor match score: {calc_match_score(closest_match.description, gt_target.description)}")
                print(f"\tHighest descriptor match id: {closest_match.id}")
                print(f"\tHigh descriptor match distance: {np.linalg.norm(closest_match.position-gt_target.position):.3f}")
                print(f"\tClose enough? {is_close_enough}")

        print(f"Imaging Sim Score: {np.sum(scores)}/{len(scores)}") 
        area_tracker.visualize(f"{debug_output_folder}/coverage.png", 5000)


if __name__ == "__main__":
    tests = TestPipeline()
    tests.test_with_sim_dataset()