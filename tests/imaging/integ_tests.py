import unittest
from uavf_2024.imaging.localizer import Localizer
from uavf_2024.imaging.area_coverage import AreaCoverageTracker
from uavf_2024.imaging.image_processor import ImageProcessor
from uavf_2024.imaging.tracker import TargetTracker
from uavf_2024.imaging.imaging_types import FullBBoxPrediction, Image, ProbabilisticTargetDescriptor, Target3D, COLORS, SHAPES, LETTERS, CertainTargetDescriptor
from uavf_2024.imaging.utils import calc_match_score
import os
import numpy as np
import shutil
import cv2 as cv
import random
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R

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

class TestPipeline(unittest.TestCase):
    def test_with_sim_dataset(self, verbose: bool = False):
        '''
        Runs the entire pipeline on the simulated dataset that includes multiple
        images annotated with the 3D position of the targets, and selects 100
        random subsets of 5 objects to choose as the drop targets,
        and compares the estimated positions of the targets to the ground truth
        to see how many are within a certain distance of the ground truth.
        '''
        FOV = 50.94 # in degrees, 
        # FOV IS NOT THE SAME AS THE CAMERA IN GODOT. 
        # This is the horizontal FOV, the camera in godot has a vertical FOV of 30
        # conversion formula is 2*arctan(16/9*tan(h_fov/2))

        # The resolution of the camera in godot is 1920x1080
        RES = (1920, 1080)
        target_localizer = Localizer(
            FOV,
            RES
        )
        area_tracker = AreaCoverageTracker(
            FOV,
            RES
        )
        if verbose:
            debug_output_folder = f"{CURRENT_FILE_PATH}/imaging_data/visualizations/integ_test"
            if os.path.exists(debug_output_folder):
                shutil.rmtree(debug_output_folder)
        else:
            debug_output_folder = None
        image_processor = ImageProcessor(debug_output_folder)


        tracker = TargetTracker()

        DATASET_FOLDER = f'{CURRENT_FILE_PATH}/imaging_data/3d_dataset'
        
        all_ground_truth: list[Target3D] = []
        with open(f"{DATASET_FOLDER}/labels.txt", "r") as f:
            for line in f.readlines():
                label, location_str = line.split(" ")
                shape_name, alphanumeric, shape_col, letter_col = label.split(",")

                all_ground_truth.append(
                    Target3D(
                        csv_to_np(location_str),
                        CertainTargetDescriptor(
                            shape_col,
                            shape_name,
                            letter_col,
                            alphanumeric
                        ).as_probabilistic()
                    )
                )

        images_dirname = f"{DATASET_FOLDER}/images"
        predictions_3d: list[Target3D] =  []
        # sort by image number (e.g. img_2 is before img_10 despite lexigraphical ordering)
        def sort_key(file_name: str):
            return int(file_name.split("_")[0][5:])
        
        img_files = filter(lambda f: f.endswith(".png"), os.listdir(images_dirname))
        for file_name in tqdm(sorted(img_files, key=sort_key)):
            img = Image.from_file(f"{images_dirname}/{file_name}")
            img_no = sort_key(file_name)
            pose_str = file_name.split(".")[0].split("_")[1:]
            cam_position = csv_to_np(pose_str[0])
            
            with open(f"{images_dirname}/rotation{img_no}.txt", "r") as f:
                raw_quaternion = csv_to_np(f.read(), dtype=float)
                cam_rot = R.from_quat(raw_quaternion)


            predictions = image_processor.process_image(img)
            area_tracker.update(([cam_position, cam_rot]), label=file_name.split("_")[0])

            if verbose:
                bounding_boxes_image_path = f"{debug_output_folder}/img_{img_no}/bounding_boxes.png"
                boxes_img = cv.imread(bounding_boxes_image_path)

            # calculate 3d positions for all detections, and draw them on the debug image
            for pred in predictions:
                pred_3d = target_localizer.prediction_to_coords(pred, [cam_position, cam_rot])
                predictions_3d.append(pred_3d)

                if not verbose: continue
                x,y,w,h, = pred.x, pred.y, pred.width, pred.height
                x3, y3, z3 = pred_3d.position
                cv.putText(boxes_img, f"{x3:.01f}, {y3:.01f}, {z3:.01f}", (x,y+h+20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if verbose:
                # draw on ground truth positions
                for gt in all_ground_truth:
                    x_reproj, y_reproj = target_localizer.coords_to_2d(gt.position, [cam_position, cam_rot])
                    re_estimated_3d = target_localizer.prediction_to_coords(FullBBoxPrediction(x_reproj, y_reproj, None, None, None), [cam_position, cam_rot])
                    x2_reproj, y2_reproj = target_localizer.coords_to_2d(re_estimated_3d.position, [cam_position, cam_rot])
                    if 0<=x_reproj<RES[0] and 0<=y_reproj<RES[1]:
                        cv.circle(boxes_img, (int(x_reproj), int(y_reproj)), 7, (255,0,0), -1)
                        cv.putText(boxes_img, str(gt.descriptor.collapse_to_certain()), (int(x_reproj), int(y_reproj)), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                        cv.circle(boxes_img, (int(x2_reproj), int(y2_reproj)), 5, (255,255,0), -1)

                cv.imwrite(bounding_boxes_image_path, boxes_img)



        area_tracker.visualize(f"{debug_output_folder}/coverage.png", 5000)
        tracker.update(predictions_3d)


        POSITION_ERROR_ACCEPTABLE_BOUND = 5 

        NUM_TARGET_SUBSETS = 1000

        scores_across_subsets = []
        hist = np.zeros(6)
        distances_across_subsets = []

        for i in tqdm(range(NUM_TARGET_SUBSETS)):
            ground_truth: list[Target3D] = random.sample(all_ground_truth, 5)

            closest_tracks = tracker.estimate_positions([t.descriptor.collapse_to_certain() for t in ground_truth])
            scores = []
            distances = []
            for gt_target, pred_track in zip(ground_truth, closest_tracks):
                is_close_enough = np.linalg.norm(pred_track.position-gt_target.position) < POSITION_ERROR_ACCEPTABLE_BOUND
                scores.append(int(is_close_enough))
                if is_close_enough:
                    distances.append(np.linalg.norm(pred_track.position-gt_target.position))
                if i==0 and verbose: # we only want to print this extra info for the first one to not clog up the output
                    print(f"Closest Match for {str(gt_target.descriptor.collapse_to_certain())}:")
                    physically_closest_match = min(predictions_3d, key=lambda pred: np.linalg.norm(pred.position-gt_target.position))
                    closest_match = max(predictions_3d, key=lambda pred: calc_match_score(pred.descriptor, gt_target.descriptor))

                    print(f"\tTrack distance: {np.linalg.norm(pred_track.position-gt_target.position):.3f}")
                    print(f"\tDetections used in track:")
                    print(f"\t\t{[detection.id for detection in pred_track.get_measurements()]}") 

                    print(f"\tClose tracks (each line is one track):")
                    for track in tracker.tracks:
                        if np.linalg.norm(track.position - gt_target.position) < POSITION_ERROR_ACCEPTABLE_BOUND:
                            print(f"\t\t{[detection.id for detection in track.get_measurements()]}")

                    print(f"\tClose detections:")
                    print(f"\t\t{[p.id for p in filter(lambda pred: np.linalg.norm(pred.position-gt_target.position) < POSITION_ERROR_ACCEPTABLE_BOUND, predictions_3d)]}")
                    print(f"\tPhysically closest detection distance: {np.linalg.norm(physically_closest_match.position-gt_target.position):.3f}")
                    print(f"\tPhysically closest detection descriptor score: {calc_match_score(physically_closest_match.descriptor, gt_target.descriptor)}")
                    print(f"\tPhysically closest detection id: {physically_closest_match.id}")
                    print(f"\tHighest descriptor match score: {calc_match_score(closest_match.descriptor, gt_target.descriptor)}")
                    print(f"\tHighest descriptor match id: {closest_match.id}")
                    print(f"\tHigh descriptor match distance: {np.linalg.norm(closest_match.position-gt_target.position):.3f}")
                    print(f"\tClose enough? {is_close_enough}")
            if i==0: 
                print(f"Imaging Sim Score: {np.sum(scores)}/{len(scores)}") 
            scores_across_subsets.append(np.sum(scores))
            hist[np.sum(scores)] += 1
            distances_across_subsets.extend(distances)
        
        avg_score = np.mean(scores_across_subsets)
        avg_distances = np.mean(distances_across_subsets)
        distances_std = np.std(distances_across_subsets)
        print(f"Imaging Sim Average Score: {avg_score}/{len(scores)}")
        print(f"Distribution of scores: {dict(zip(range(6), hist.astype(int)))}")
        print(f"Localization error for correct detections: {avg_distances:.3f} +/- {distances_std:.3f}")


if __name__ == "__main__":
    tests = TestPipeline()
    tests.test_with_sim_dataset(verbose=True)