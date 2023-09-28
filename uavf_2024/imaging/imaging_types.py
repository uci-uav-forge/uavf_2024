from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass
class Prediction:
    x: int
    y: int
    width: int
    height: int
    '''
    We can worry about typechecking these later, but the gist is that they're probability distributions over the possible classes.
    '''
    shape_confidences: np.ndarray
    letter_confidences: np.ndarray
    shape_color_confidences: np.ndarray
    letter_color_confidences: np.ndarray



    