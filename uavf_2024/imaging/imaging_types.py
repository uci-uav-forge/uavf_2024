from dataclasses import dataclass
from typing import Generator, NamedTuple, Union
import numpy as np
from enum import Enum

import torch

# TODO: Limit these to the types we actually use
integer = Union[int, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]
real = Union[float, np.float16, np.float32, np.float64, np.float128]
number = Union[integer, real]

img_coord_t = np.uint16

@dataclass
class TargetDescription:
    shape_probs: np.ndarray
    letter_probs: np.ndarray
    shape_col_probs: np.ndarray
    letter_col_probs: np.ndarray

@dataclass
class Tile:
    img: 'Image'
    x: img_coord_t
    y: img_coord_t

@dataclass
class FullPrediction:
    x: img_coord_t
    y: img_coord_t
    width: img_coord_t
    height: img_coord_t
    '''
    We can worry about typechecking these later, but the gist is that they're probability distributions over the possible classes.
    '''
    description: TargetDescription

@dataclass
class InstanceSegmentationResult:
    '''
    `mask` and `img` should be (w,h,c) where c is 1 for mask and 3 for img
    '''
    x: img_coord_t
    y: img_coord_t
    width: img_coord_t
    height: img_coord_t
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
    
    def __repr__(self):
        return f"Image({self._array}, {self._dim_order})"
    
    def __mul__(self, other: number | np.ndarray | torch.Tensor) -> 'Image':
        """
        Multiplies the underlying array by a scalar or another array.
        """
        return Image(self._array * other, self._dim_order)
    
    def get_array(self):
        return self._array
    
    def make_sub_image(self, x_coord, y_coord, width, height) -> 'Image':
        """
        Does not copy the underlying array.
        """
        return Image(self._array[x_coord:x_coord+width, y_coord:y_coord+height], self._dim_order)
    
    def make_tile(self, x_coord, y_coord, tile_size) -> Tile:
        """
        Does not copy the underlying array.
        """
        return Tile(self.make_sub_image(x_coord, y_coord, tile_size, tile_size), x_coord, y_coord)
    
    @property
    def shape(self):
        return self._array.shape
    
    @property 
    def dim_order(self):
        return self._dim_order
    
    @property
    def height(self):
        return self._array.shape[self._dim_order.index(HEIGHT)]
    
    @property
    def width(self):
        return self._array.shape[self._dim_order.index(WIDTH)]
    
    @property
    def channels(self):
        return self._array.shape[self._dim_order.index(CHANNELS)]
    
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
        
    def generate_tiles(self, tile_size: int, min_overlap: int = 0) -> Generator[Tile, None, None]:
        """
        Split high resolution input image into number of fixed-dimension, squares tiles, 
        ensuring that each tile overlaps with its neighbors by at least `min_overlap` pixels.

        @read: https://stackoverflow.com/questions/58383814/how-to-divide-an-image-into-evenly-sized-overlapping-if-needed-tiles

        Args:
            tile_size (int): Width/height of each tile
            min_overlap (int, optional): Number of pixels that each tile overlaps with its neighbors. Defaults to 0.

        Yields:
            Generator[Tile, None, None]: Generator that yields each tile
        """
    
        img_height, img_width = self.height, self.width

        if tile_size > img_width or tile_size > img_height:
            raise ValueError("tile dimensions cannot be larger than origin dimensions")

        # Number of tiles in each dimension
        x_count = np.uint8(np.ceil(
            (img_width - min_overlap) / (tile_size - min_overlap)
        ))
        y_count = np.uint8(np.ceil(
            (img_height - min_overlap) / (tile_size - min_overlap)
        ))

        # Total remainders
        overflow_x = tile_size + (x_count - 1) * (tile_size - min_overlap) - img_width
        overflow_y = tile_size + (y_count - 1) * (tile_size - min_overlap) - img_height

        # Temporarily suppress divide-by-zero warnings
        np.seterr(divide='ignore', invalid='ignore')

        # Set up remainders per tile
        remaindersX = np.ones((x_count-1,), dtype=np.uint8) * np.uint16(np.floor(overflow_x / (x_count-1)))
        remaindersY = np.ones((y_count-1,), dtype=np.uint8) * np.uint16(np.floor(overflow_y / (y_count-1)))
        remaindersX[0:np.remainder(overflow_x, np.uint16(x_count-1))] += 1
        remaindersY[0:np.remainder(overflow_y, np.uint16(y_count-1))] += 1

        np.seterr(divide='warn', invalid='warn')
            
        y = np.uint16(0)
        for vertical_index in range(y_count):
            x = np.uint16(0)
            for horizontal_index in range(x_count):
                # Converting back to int because its expected downstream
                # All dimensions should be refactored to use unit16
                yield self.make_tile(x, y, tile_size)
                
                if horizontal_index < (x_count-1):
                    next_horizontal_overlap = min_overlap + remaindersX[horizontal_index]
                    x += tile_size - next_horizontal_overlap
                    
            if vertical_index < (y_count-1):
                next_vertical_overlap = min_overlap + remaindersY[vertical_index]
                y += tile_size - next_vertical_overlap
