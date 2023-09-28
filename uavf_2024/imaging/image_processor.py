import numpy as np
from dataclasses import dataclass
from imaging_types import Predictions

class ImageProcessor:
    def __init__(self):
        '''
        Initialize and warm-up all ML models here
        '''
        pass
    def process_image(img: np.ndarray) -> Predictions:
        '''
        img shape should be (channels, width, height)
        (that tuple order is a placeholder for now and we can change it later, but it should be consistent and we need to keep the docstring updated)

        TODO: replace this with actual ML models and tiling
        '''
        width, height = img.shape[1:]


        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        width = np.random.randint(30, 60)
        height = np.random.randint(30, 60)

        shape_confidences = np.random.rand(8)
        letter_confidences = np.random.rand(36)
        letter_color_confidences = np.random.rand(8)
        shape_color_confidences = np.random.rand(8)

        return Predictions(x, y, width, height, shape_confidences, letter_confidences, shape_color_confidences, letter_color_confidences)
