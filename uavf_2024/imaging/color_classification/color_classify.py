import numpy as np
from uavf_2024.imaging.imaging_types import COLORS_TO_RGB
import tensorflow as tf
import os
CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
class ColorClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model( CURRENT_FILE_PATH /'trained_model')

    def predict(self, image: np.ndarray) -> (int, int):
        ''' Returns probabilities for each color'''
        result = self.model.predict(image)
        shape_predict = result[0]
        letter_predict = result[1]
        return shape_predict, letter_predict
