import itertools
import math

import numpy as np

from uavf_2024.imaging.image_processor import Tile
from .imaging_types import TargetDescription, Target3D


def calc_match_score(target_desc: TargetDescription, target: Target3D):
    shape_score = target.shape_probs[target_desc.shape]
    letter_score = target.letter_probs[target_desc.letter]
    shape_color_score = target.shape_col_probs[target_desc.shape_color]
    letter_color_score = target.letter_col_probs[target_desc.letter_color]

    return shape_score * letter_score * shape_color_score * letter_color_score


def split_to_tiles(img: np.ndarray, tile_size: int) -> "list[Tile]":
    h, w = img.shape[:2]
    n_horizontal_tiles = math.ceil(w / tile_size)
    n_vertical_tiles = math.ceil(h / tile_size)
    all_tiles: list[Tile] = []
    v_indices = np.linspace(0, h - tile_size, n_vertical_tiles).astype(int)
    h_indices = np.linspace(0, w - tile_size, n_horizontal_tiles).astype(int)

    for v, h in itertools.product(v_indices, h_indices):
        tile = img[v : v + tile_size, h : h + tile_size]
        all_tiles.append(Tile(tile, h, v))

    return all_tiles
