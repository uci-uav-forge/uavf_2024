import unittest
import os
import numpy as np
import cv2 as cv

from uavf_2024.imaging.letter_classification import LetterClassifier

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class TestLetterClassification(unittest.TestCase):
    def setUp(self) -> None:
        self.letter_size = 128
        self.letter_classifier = LetterClassifier(self.letter_size)

    def test_letter_classification(self):
        imgs_path = CURRENT_FILE_PATH + "/imaging_data/letter_dataset/images"
        labels_path = CURRENT_FILE_PATH + "/imaging_data/letter_dataset/labels"
        total = 0
        correct = 0
        for img_file_name in os.listdir(imgs_path):
            img = cv.imread(f"{imgs_path}/{img_file_name}")
            raw_output = self.letter_classifier.model.predict(img)
            pred = np.argmax(raw_output[0].probs.data.numpy())
            with open(f"{labels_path}/{img_file_name.split('.')[0]}.txt") as f:
                truth = int(f.read(2))
            if truth == pred:
                correct += 1
            total += 1
        print(f"Letter only tests: {correct} out of {total}")