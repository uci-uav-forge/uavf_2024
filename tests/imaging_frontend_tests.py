import torch
from torchvision.ops import box_iou
import unittest
from uavf_2024.imaging.image_processor import ImageProcessor
from uavf_2024.imaging.imaging_types import FullPrediction
from uavf_2024.imaging.visualizations import visualize_predictions
import numpy as np
import cv2 as cv
import os

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def calc_metrics(predictions: list[FullPrediction], ground_truth: list[FullPrediction]):
    true_positives = 0 # how many predictions were on top of a ground-truth box
    targets_detected = 0 # how many ground-truth boxes had at least 1 prediction on top of them
    shape_top_1_accuracies = []
    letter_top_1_accuracies = []
    shape_color_top_1_accuracies = []
    letter_color_top_1_accuracies = []


    for truth in ground_truth:
        x,y = truth.x, truth.y
        w,h = truth.width, truth.height 
        true_box = np.array([[
            x,y,x+w,y+h
        ]])

        shape = np.argmax(truth.shape_confidences)
        letter = np.argmax(truth.letter_confidences)
        shape_col = np.argmax(truth.shape_color_confidences)
        letter_col = np.argmax(truth.letter_color_confidences)

        this_target_was_detected = False
        for pred in predictions:
            pred_box = np.array([[
                pred.x,pred.y,pred.x+pred.width,pred.y+pred.height
            ]])

            iou = box_iou(torch.Tensor(true_box), torch.Tensor(pred_box))
            if iou>0.1:
                true_positives+=1
                this_target_was_detected = True
                shape_top_1_accuracies.append(int(shape == np.argmax(pred.shape_confidences)))
                letter_top_1_accuracies.append(int(letter == np.argmax(pred.letter_confidences)))
                shape_color_top_1_accuracies.append(int(shape_col == np.argmax(pred.shape_color_confidences)))
                letter_color_top_1_accuracies.append(int(letter_col == np.argmax(pred.letter_color_confidences)))

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

def parse_dataset(imgs_path, labels_path) -> tuple[list[np.ndarray], list[list[FullPrediction]]]:
    '''
    ret_value[i] is the list of predictions for the ith image
    ret_value[i][j] is the jth prediction for the ith image
    '''
    imgs = []
    labels = []
    for img_file_name in os.listdir(imgs_path):
        img = cv.imread(f"{imgs_path}/{img_file_name}")
        ground_truth: list[FullPrediction] = []
        with open(f"{labels_path}/{img_file_name.split('.')[0]}.txt") as f:
            for line in f.readlines():
                label = line.split(' ')
                shape, letter, shape_col, letter_col = map(int, label[:4])
                box = np.array([float(v) for v in label[4:]])
                box[[0,2]]*=img.shape[1]
                box[[1,3]]*=img.shape[0]
                box[[0,1]] -= box[[2,3]]/2 # adjust xy to be top-left
                x,y,w,h = box.astype(int)

                ground_truth.append(FullPrediction(
                    x,y,w,h,np.eye(13)[shape], np.eye(36)[letter], np.eye(8)[shape_col], np.eye(8)[letter_col]
                ))
        imgs.append(img)
        labels.append(ground_truth)
    return (imgs, labels)

class TestImagingFrontend(unittest.TestCase):
    def setUp(self) -> None:
        self.image_processor = ImageProcessor()

    def test_runs_without_crashing(self):
        sample_input = cv.imread(f"{CURRENT_FILE_PATH}/fullsize_dataset/images/image0.png")
        res = self.image_processor.process_image(sample_input)

    def test_metrics(self):
        imgs, labels = parse_dataset(f"{CURRENT_FILE_PATH}/tile_dataset/images", f"{CURRENT_FILE_PATH}/tile_dataset/labels")
        
        recalls = []
        precisions = []
        shape_top1s = []
        letter_top1s = []
        shape_color_top1s = []
        letter_color_top1s = []

        for img, ground_truth in zip(imgs, labels):
            predictions = self.image_processor.process_image(img)

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
