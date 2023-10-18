from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, TypeVar, Union
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

import torch

@dataclass
class TargetDescription:
    shape_probs: np.ndarray
    letter_probs: np.ndarray
    shape_col_probs: np.ndarray
    letter_col_probs: np.ndarray

@dataclass
class Tile:
    img: np.ndarray
    x: int
    y: int

@dataclass
class FullPrediction:
    x: int
    y: int
    width: int
    height: int
    '''
    We can worry about typechecking these later, but the gist is that they're probability distributions over the possible classes.
    '''
    description: TargetDescription

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
class Target3D:
    '''
    TODO: decision to use lat/lng is not final
    We might also want to incorporate information about the distance from which we've seen this target. Like, if we've only seen it from far away, and we get a new classification from a closer image, it should have more weight.
    '''
    lat: float
    lng: float
    description: TargetDescription

class ImageDimension(Enum):
    HEIGHT = 'h'
    WIDTH = 'w'
    CHANNELS = 'c'
    
HEIGHT = ImageDimension.HEIGHT
WIDTH = ImageDimension.WIDTH
CHANNELS = ImageDimension.CHANNELS
    
class ImageDimensionsOrder(NamedTuple):
    first_dim: ImageDimension
    second_dim: ImageDimension
    third_dim: ImageDimension

# Can support other dimension orders if necessary
HWC = ImageDimensionsOrder(HEIGHT, WIDTH, CHANNELS)
CHW = ImageDimensionsOrder(CHANNELS, HEIGHT, WIDTH)
_VALID_DIM_ORDERS = {HWC, CHW}

class Image:
    """
    Wraps a numpy array or torch tensor representing an image.
    Contains information about the dimension order of the underlying array, e.g., (height, width, channels) or (channels, height, width).
    
    Except for passing data to predictors, you should interface through it directly instead of accessing _array.
    NOTE: Add methods to interface with it if necessary.
    
    TODO: Refactor imaging modules to use this class
    
    Examples:
    
    image_hwc = Image(np.zeros((20, 20, 3)), HWC)
    
    image_chw = Image(np.zeros((3, 20, 20)), CHW)
    """    
    def __init__(
        self, 
        array: np.ndarray | torch.Tensor, 
        dim_order: ImageDimensionsOrder
    ):
        if not isinstance(array, np.ndarray) and not isinstance(array, torch.Tensor):
            raise TypeError("array must be a numpy array or torch tensor")
        
        if len(array.shape) != 3:
            raise ValueError("array must have 3 axes, got shape " + str(array.shape))
        
        if dim_order not in _VALID_DIM_ORDERS:
            raise ValueError("dim_order must be one of " + str(_VALID_DIM_ORDERS))

        self._dim_order = dim_order
        
        channels_index = self._dim_order.index(CHANNELS)
        if array.shape[channels_index] != 3:
            raise ValueError("Image array must have 3 channels, got " + str(array[channels_index]))
        
        self._array = array
        
    def __getitem__(self, key):
        return self._array[key]
    
    def __setitem__(self, key, value: Union[np.ndarray, torch.Tensor, 'Image', int, float, np.number, torch.NumberType]):
        if isinstance(value, Image):
            value = value._array
        
        # I'm not sure why this thrown a fit in VS Code, but it work. Trust.
        self._array[key] = value # type: ignore
    
    def __eq__(self, other: 'Image'):
        """
        Checks whether two images are equal, including whether they have the same dimension order.
        """
        return isinstance(other, Image) and self._dim_order == other._dim_order and (self._array == other._array).all()
    
    def get_array(self):
        return self._array
    
    @property
    def shape(self):
        return self._array.shape
    
    @property 
    def dim_order(self):
        return self._dim_order
    
    def change_dim_order(self, target_dim_order: ImageDimensionsOrder) -> None:
        """
        Use transpose to change the order of the dimensions in-place. This does NOT copy the underlying array.
        Changes the dim_order accordingly.
        """
        transposition_indices = (
            self._dim_order.index(target_dim_order.first_dim),
            self._dim_order.index(target_dim_order.second_dim),
            self._dim_order.index(target_dim_order.third_dim)
        )
        
        if isinstance(self._array, np.ndarray):
            self._array = self._array.transpose(transposition_indices)
        elif isinstance(self._array, torch.Tensor):
            self._array = self._array.permute(transposition_indices)
        else:
            TypeError("Inner array must be a numpy array or torch tensor")
        
        self._dim_order = target_dim_order
