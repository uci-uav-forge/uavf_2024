import unittest
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from uavf_2024.imaging.letter_classification import LetterClassifier
from uavf_2024.imaging.auc_heuristic import roc_auc

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

# inspired by letter_test.py
class TestAUCLetterClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.letter_size = 128
        self.letter_classifier = LetterClassifier(self.letter_size)
        self.imgs_path = CURRENT_FILE_PATH + "/imaging_data/letter_dataset/images" # "/imaging_data/letter_dataset/12.06.23/images"
        self.labels_path = CURRENT_FILE_PATH + "/imaging_data/letter_dataset/labels" # "/imaging_data/letter_dataset/12.06.23/labels"

    
    def save_raw_letter_output(self,imgs_path, labels_path) -> (np.array, np.array):
        # Saves true prediction distribution of letter classification based input
        # y_true: array of true labels for each image (N, 1)
        # y_probas: array of class probability dist for each image (N, C)
        #   N = number of images in imgs_path
        #   C = number of classes 
        y_probas, y_true = [], []

        for img_file_name in os.listdir(imgs_path):
            img = cv.imread(f"{imgs_path}/{img_file_name}")
            raw_output = self.letter_classifier.model.predict(img)
            pred = raw_output[0].probs.data.numpy()
            with open(f"{labels_path}/{img_file_name.split('.')[0]}.txt") as f:
                truth = int(f.read(2))
            y_true.append(truth)
            y_probas.append(pred)
        return np.array(y_true), np.array(y_probas)


    def test_with_current_letter_classifier(self):
        y_true, y_probas = self.save_raw_letter_output(self.imgs_path, self.labels_path)
        temp = roc_auc(y_probas, y_true)
        print("Micro-average ROC curve area =",temp[0][-2])
        print("Macro-average ROC curve area =",temp[0][-1])
        # plt.show()
