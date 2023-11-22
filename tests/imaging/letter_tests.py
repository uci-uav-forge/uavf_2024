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
        top_1 = 0
        top_5 = 0
        for img_file_name in os.listdir(imgs_path):
            img = cv.imread(f"{imgs_path}/{img_file_name}")
            raw_output = self.letter_classifier.model.predict(img)
            pred = np.argsort(raw_output[0].probs.data.numpy())[-5:]
            pred1 = []
            for p in pred:
                pred1.append(int(raw_output[0].names[p]))
            with open(f"{labels_path}/{img_file_name.split('.')[0]}.txt") as f:
                truth = int(f.read(2))
                # convert old labels (letters first) to new labels (digits first)
                if truth <= 24:
                    truth += 10
                else:
                    truth -= 25
            if truth == pred1[4]:
                top_1 += 1
            for p in pred1:
                if truth == p:
                    top_5 += 1
                    print(f"CORRECT: {img_file_name}, LABEL: {p}")
                    break
            total += 1
        print(f"Letter only tests:\nTop 1: {top_1} out of {total}\nTop 5: {top_5} out of {total}")