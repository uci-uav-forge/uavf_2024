from enum import Enum
import numpy as np
from dataclasses import dataclass
import torch.nn.functional as F

class Shape(Enum):
    CIRCLE = 0
    SEMICIRCLE = 1
    QUARTERCIRCLE = 2
    TRIANGLE = 3
    RECTANGLE = 4
    PENTAGON = 5
    STAR = 6
    CROSS = 7
    EMERGENT = 8

class Color(Enum):
    WHITE = 0
    BLACK = 1
    RED = 2
    BLUE = 3
    GREEN = 4
    PURPLE = 5
    BROWN = 6
    ORANGE = 7

@dataclass
class TargetDescription:
    shape: np.ndarray
    shape_color: np.ndarray
    letter: np.ndarray
    letter_color_one_hot: np.ndarray 

class TargetTracker:
    def __init__(self, target_descriptions: list[tuple[Shape, Color, int, Color]]):
        '''
        target_descriptions is list of 5 tuples:
        [
            (shape, shape_color, letter, letter_color)
        ]
        letter is a integer where [0,9] are their respective digits, and higher numbers are indices of the alphabet, where 10 is a and 35 is z
        '''
        self.target_descriptions = []
        for shape, shape_color, letter, letter_color in target_descriptions:
            shape_one_hot = F.one_hot(shape).numpy()
            shape_color_one_hot = F.one_hot(shape_color).numpy()
            letter_one_hot = F.one_hot(letter).numpy()
            letter_color_one_hot = F.one_hot(letter_color).numpy()
            self.target_descriptions

