from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass
class FullPrediction:
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



@dataclass
class InstanceSegmentationResult:
    '''
    `mask` and `img` should be (w,h,c) where c is 1 for mask and 3 for img
    '''
    x: int
    y: int
    width: int
    height: int
    confidences: np.ndarray
    mask: np.ndarray
    img: np.ndarray

@dataclass
class TargetDescription:
    shape: int
    letter: int
    shape_color: int
    letter_color: int

@dataclass
class Target3D:
    '''
    TODO: decision to use lat/lng is not final
    We might also want to incorporate information about the distance from which we've seen this target. Like, if we've only seen it from far away, and we get a new classification from a closer image, it should have more weight.
    '''
    lat: float
    lng: float
    shape_probs: np.ndarray
    letter_probs: np.ndarray
    shape_col_probs: np.ndarray
    letter_col_probs: np.ndarray