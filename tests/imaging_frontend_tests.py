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
    true_positives = 0
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

        for pred in predictions:
            pred_box = np.array([[
                pred.x,pred.y,pred.x+pred.width,pred.y+pred.height
            ]])

            iou = box_iou(torch.Tensor(true_box), torch.Tensor(pred_box))
            if iou>0.1:
                true_positives+=1
                shape_top_1_accuracies.append(int(shape == np.argmax(pred.shape_confidences)))
                letter_top_1_accuracies.append(int(letter == np.argmax(pred.letter_confidences)))
                shape_color_top_1_accuracies.append(int(shape_col == np.argmax(pred.shape_color_confidences)))
                letter_color_top_1_accuracies.append(int(letter_col == np.argmax(pred.letter_color_confidences)))

    recall = true_positives / len(ground_truth)
    precision = true_positives / len(predictions)
    shape_top1 = np.mean(shape_top_1_accuracies)
    letter_top1 = np.mean(letter_top_1_accuracies)
    shape_color_top1 = np.mean(shape_color_top_1_accuracies)
    letter_color_top1 = np.mean(letter_color_top_1_accuracies)

    return (
        recall,
        precision,
        shape_top1,
        letter_top1,
        shape_color_top1,
        letter_color_top1
    )



class TestImagingFrontend(unittest.TestCase):
    def setUp(self) -> None:
        self.image_processor = ImageProcessor()

    def test_runs_without_crashing(self):
        sample_input = cv.imread(f"{CURRENT_FILE_PATH}/test_dataset/images/image0.png")
        res = self.image_processor.process_image(sample_input)

    def test_metrics(self):
        sample_input = cv.imread(f"{CURRENT_FILE_PATH}/test_dataset/images/image0.png")
        predictions = self.image_processor.process_image(sample_input)
        visualize_predictions(sample_input, predictions, "preds")
        ground_truth: list[FullPrediction] = []
        with open(f"{CURRENT_FILE_PATH}/test_dataset/labels/image0.txt", "r") as f:
            for line in f.readlines():
                label = line.split(' ')
                shape, letter, shape_col, letter_col = map(int, label[:4])
                box = np.array([float(v) for v in label[4:]])
                box[[0,2]]*=sample_input.shape[1]
                box[[1,3]]*=sample_input.shape[0]
                box[[0,1]] -= box[[2,3]]/2 # adjust xy to be top-left
                x,y,w,h = box.astype(int)

                ground_truth.append(FullPrediction(
                    x,y,w,h,np.eye(13)[shape], np.eye(36)[letter], np.eye(8)[shape_col], np.eye(8)[letter_col]
                ))

        visualize_predictions(sample_input, ground_truth, "ground_truth")
        
        (
            recall,
            precision,
            shape_top1,
            letter_top1,
            shape_color_top1,
            letter_color_top1
        ) = calc_metrics(predictions, ground_truth)

        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"Shape top 1 acc: {shape_top1}")
        print(f"Letter top 1 acc: {letter_top1}")
        print(f"Shape color top 1 acc: {shape_color_top1}")
        print(f"Letter color top 1 acc: {letter_color_top1}")
