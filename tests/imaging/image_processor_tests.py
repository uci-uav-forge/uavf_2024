from __future__ import annotations
import shutil
import torch
from torchvision.ops import box_iou
import unittest
from uavf_2024.imaging.image_processor import ImageProcessor
from uavf_2024.imaging.imaging_types import HWC, FullPrediction, Image, TargetDescription
from uavf_2024.imaging import profiler
import numpy as np
import os
from time import time
from tqdm import tqdm
import line_profiler
from memory_profiler import profile as mem_profile

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def calc_metrics(predictions: list[FullPrediction], ground_truth: list[FullPrediction]):
    true_positives = 0 # how many predictions were on top of a ground-truth box
    targets_detected = 0 # how many ground-truth boxes had at least 1 prediction on top of them
    shape_top_1_accuracies = []
    letter_top_1_accuracies = []
    shape_color_top_1_accuracies = []
    letter_color_top_1_accuracies = []

    # letter_dict is from the letter model's raw_output[0].names
    # it is basically 0-25 in alphabetical order and maps the predicton results from the model to 
    # the new letter labels indicies
    letter_dict = {0: '0', 1: '1', 2: '10', 3: '11', 4: '12', 5: '13', 6: '14', 7: '15', 8: '16', 9: '17', 10: '18', 11: '19', 12: '2', 13: '20', 14: '21', 15: '22', 16: '23', 17: '24', 18: '25', 19: '26', 20: '27', 21: '28', 22: '29', 23: '3', 24: '30', 25: '31', 26: '32', 27: '33', 28: '34', 29: '35', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}
    # old letter labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
    # new letter labels = "0123456789ABCDEFGHIKLMNOPQRSTUVWXYZ"
    # old to new:
    #   A - Z (0-25): + 10
    #   1 - 9 (26-34): -25

    for truth in ground_truth:
        x,y = truth.x, truth.y
        w,h = truth.width, truth.height 
        true_box = np.array([[
            x,y,x+w,y+h
        ]])

        shape = np.argmax(truth.description.shape_probs)
        letter = np.argmax(truth.description.letter_probs)
        # convert from old letter labels to new letter labels
        if letter <= 25:
            letter += 10
        else:
            letter -= 25
        shape_col = np.argmax(truth.description.shape_col_probs)
        letter_col = np.argmax(truth.description.letter_col_probs)

        this_target_was_detected = False
        for pred in predictions:
            pred_box = np.array([[
                pred.x,pred.y,pred.x+pred.width,pred.y+pred.height
            ]])

            iou = box_iou(torch.Tensor(true_box), torch.Tensor(pred_box))
            if iou>0.1:
                true_positives+=1
                this_target_was_detected = True
                shape_top_1_accuracies.append(int(shape == np.argmax(pred.description.shape_probs)))
                letter_top_1_accuracies.append(int(letter == int(letter_dict[np.argmax(pred.description.letter_probs)])))
                shape_color_top_1_accuracies.append(int(shape_col == np.argmax(pred.description.shape_col_probs)))
                letter_color_top_1_accuracies.append(int(letter_col == np.argmax(pred.description.letter_col_probs)))

        if this_target_was_detected:
            targets_detected+=1

    recall = targets_detected / len(ground_truth) if len(ground_truth)>0 else None
    precision = true_positives / len(predictions) if len(predictions)>0 else None
    shape_top1 = np.mean(shape_top_1_accuracies) if len(shape_top_1_accuracies)>0 else None
    letter_top1 = np.mean(letter_top_1_accuracies) if len(letter_top_1_accuracies)>0 else None
    shape_color_top1 = np.mean(shape_color_top_1_accuracies) if len(shape_color_top_1_accuracies)>0 else None
    letter_color_top1 = np.mean(letter_color_top_1_accuracies) if len(letter_color_top_1_accuracies)>0 else None

    return (
        recall,
        precision,
        shape_top1,
        letter_top1,
        shape_color_top1,
        letter_color_top1
    )

def parse_dataset(imgs_path, labels_path) -> tuple[list[Image], list[list[FullPrediction]]]:
    '''
    ret_value[i] is the list of predictions for the ith image
    ret_value[i][j] is the jth prediction for the ith image
    '''
    imgs: list[Image] = []
    labels = []
    for img_file_name in os.listdir(imgs_path):
        img = Image.from_file(f"{imgs_path}/{img_file_name}")
        ground_truth: list[FullPrediction] = []
        with open(f"{labels_path}/{img_file_name.split('.')[0]}.txt") as f:
            for line in f.readlines():
                label = line.split(' ')
                shape, letter, shape_col, letter_col = map(int, label[:4])
                box = np.array([float(v) for v in label[4:] if v != ""])
                box[[0,2]]*=img.shape[1]
                box[[1,3]]*=img.shape[0]
                box[[0,1]] -= box[[2,3]]/2 # adjust xy to be top-left
                x,y,w,h = box.astype(int)

                ground_truth.append(FullPrediction(
                    x,y,w,h,
                    TargetDescription(
                        np.eye(9)[shape], np.eye(36)[letter], np.eye(8)[shape_col], np.eye(8)[letter_col]
                    )
                ))
        imgs.append(img)
        labels.append(ground_truth)
    return (imgs, labels)

class TestImagingFrontend(unittest.TestCase):

    @mem_profile
    def test_runs_without_crashing(self):
        image_processor = ImageProcessor()
        sample_input = Image.from_file(f"{CURRENT_FILE_PATH}/imaging_data/fullsize_dataset/images/image0.png")
        res = image_processor.process_image(sample_input)

    @profiler
    def test_benchmark_fullsize_images(self):
        image_processor = ImageProcessor()
        sample_input = Image.from_file(f"{CURRENT_FILE_PATH}/imaging_data/fullsize_dataset/images/image0.png")
        times = []
        N_runs = 10
        for i in tqdm(range(N_runs)):
            start = time()
            res = image_processor.process_image(sample_input)
            elapsed = time()-start
            times.append(elapsed)
        print(f"Fullsize image benchmarks (average of {N_runs} runs):")
        print(f"Avg: {np.mean(times)}, StdDev: {np.std(times)}")
        lstats = profiler.get_stats()
        line_profiler.show_text(lstats.timings, lstats.unit)
    
    def test_no_duplicates(self):
        # Given 5 identified bounding boxes, removes duplicate bounding box using nms such that there are 4 bounding boxes left
        debug_output_folder = f"{CURRENT_FILE_PATH}/imaging_data/visualizations/test_duplicates"
        sample_input = Image.from_file(f"{CURRENT_FILE_PATH}/imaging_data/fullsize_dataset/images/image0.png")
        image_processor = ImageProcessor(debug_output_folder)
        res = image_processor.process_image(sample_input)
        assert len(res)==4

    def test_metrics(self):
        debug_output_folder = f"{CURRENT_FILE_PATH}/imaging_data/visualizations/test_metrics"
        if os.path.exists(debug_output_folder):
            shutil.rmtree(debug_output_folder)
        image_processor = ImageProcessor(debug_output_folder)
        #change the tile_dataset
        imgs, labels = parse_dataset(f"{CURRENT_FILE_PATH}/imaging_data/tile_dataset/images", f"{CURRENT_FILE_PATH}/imaging_data/tile_dataset/labels")
        
        recalls = []
        precisions = []
        shape_top1s = []
        letter_top1s = []
        shape_color_top1s = []
        letter_color_top1s = []

        for img, ground_truth in zip(imgs, labels):
            predictions = image_processor.process_image(img)

            (
                recall,
                precision,
                shape_top1,
                letter_top1,
                shape_color_top1,
                letter_color_top1
            ) = calc_metrics(predictions, ground_truth) 
            
            for metric, aggregate in zip(
                [recall, precision, shape_top1, letter_top1, shape_color_top1, letter_color_top1],
                [recalls, precisions, shape_top1s, letter_top1s, shape_color_top1s, letter_color_top1s]
            ):
                if not metric is None:
                    aggregate.append(metric)

        print(f"Recall: {np.mean(recalls)}")
        print(f"Precision: {np.mean(precisions)}")
        print(f"Shape top 1 acc: {np.mean(shape_top1s)}")
        print(f"Letter top 1 acc: {np.mean(letter_top1s)}")
        print(f"Shape color top 1 acc: {np.mean(shape_color_top1s)}")
        print(f"Letter color top 1 acc: {np.mean(letter_color_top1s)}")

if __name__ == "__main__":
    unittest.main()