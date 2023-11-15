from __future__ import annotations
from ultralytics import YOLO
import numpy as np
from ultralytics.engine.results import Results
import torch
import os

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class LetterClassifier:
    def __init__(self, img_size):
        self.model = YOLO(f'{CURRENT_FILE_PATH}/weights/letter.pt')

        rand_input = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
        self.model.predict(list(rand_input), verbose=False)
    
    def predict(self, imgs : list[np.ndarray]) -> list[np.ndarray]:
        '''
        Returns MxN array of class probabilities
        M: number of images
        N: number of classes
        For example you can process 3 images at once and get back a 3x26 array where each image has its own array of class probabilities
        '''
        imgs = [np.repeat(img[...,np.newaxis],3,axis=2) for img in imgs]
        raw_output: list[Results] = self.model.predict(imgs)
        output = [data.probs.data.numpy() for data in raw_output]
        return output
