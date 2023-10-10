from typing import Generator

import numpy as np

from .imaging_types import TargetDescription, Target3D, Tile


def calc_match_score(target_desc: TargetDescription, target: Target3D):
    shape_score = target.shape_probs[target_desc.shape]
    letter_score = target.letter_probs[target_desc.letter]
    shape_color_score = target.shape_col_probs[target_desc.shape_color]
    letter_color_score = target.letter_col_probs[target_desc.letter_color]

    return shape_score * letter_score * shape_color_score * letter_color_score


def generate_tiles(img: np.ndarray, tile_size: int, min_overlap: int = 0) -> "Generator[Tile, None, None]":
    """
    Split high resolution input image into number of fixed-dimension, squares tiles, 
    ensuring that each tile overlaps with its neighbors by at least `min_overlap` pixels.
    
    @read: https://stackoverflow.com/questions/58383814/how-to-divide-an-image-into-evenly-sized-overlapping-if-needed-tiles

    IMO, it's not good practice to have a utils module, especially considering this function only has one caller.

    Args:
        img (np.ndarray): Color, full-resolution input image in the format (height, width, channels)
        tile_size (int): Width/height of each tile
        min_overlap (int, optional): Number of pixels that each tile overlaps with its neighbors. Defaults to 0.

    Yields:
        Generator[Tile, None, None]: Generator that yields each tile
    """
    
    img_height, img_width = img.shape[:2]
    
    if tile_size > img_width or tile_size > img_height:
        raise ValueError("tile dimensions cannot be larger than origin dimensions")

    # Number of tiles in each dimension
    x_count = np.uint8(np.ceil(img_width / tile_size))
    y_count = np.uint8(np.ceil(img_height / tile_size))

    # Total remainders
    overflow_x = x_count * tile_size - img_width
    overflow_y = y_count * tile_size - img_height

    # Set up remainders per tile
    remaindersX = np.ones((x_count-1,), dtype=np.uint8) * np.uint16(np.floor(overflow_x / (x_count-1)))
    remaindersY = np.ones((y_count-1,), dtype=np.uint8) * np.uint16(np.floor(overflow_y / (y_count-1)))
    remaindersX[0:np.remainder(overflow_x, np.uint16(x_count-1))] += 1
    remaindersY[0:np.remainder(overflow_y, np.uint16(y_count-1))] += 1
    
    x = np.uint16(0)
    for horizontal_index in range(x_count):
        y = np.uint16(0)
        for vertical_index in range(y_count):
            # Converting back to int because its expected downstream
            # All dimensions should be refactored to use unit16
            yield Tile(img[y:y+tile_size, x:x+tile_size], int(x), int(y))
            if vertical_index < (y_count-1):
                y = y + tile_size - remaindersY[vertical_index]
        if horizontal_index < (x_count-1):
            x = x + tile_size - remaindersX[horizontal_index]
    