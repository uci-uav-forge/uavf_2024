from .imaging_types import TargetDescription

from typing import Generator

import numpy as np

from .imaging_types import TargetDescription, Tile

from itertools import islice

def normalize_distribution( b: TargetDescription, offset):
    shape_norm_prob = (b.shape_probs + offset)/sum(b.shape_probs + offset)
    letter_norm_prob = (b.letter_probs + offset)/sum(b.letter_probs + offset)
    shape_col_norm_prob = (b.shape_col_probs + offset)/sum(b.shape_col_probs + offset)
    letter_col_norm_prob = (b.letter_col_probs + offset)/sum(b.letter_col_probs + offset)
    
    return TargetDescription(shape_norm_prob, letter_norm_prob, shape_col_norm_prob, letter_col_norm_prob)


def calc_match_score(a: TargetDescription, b: TargetDescription):
        '''
        Returns a number between 0 and 1 representing how likely the two descriptions are the same target
        
        '''
        b = normalize_distribution(b, 1e-4)
        shape_score = sum(a.shape_probs * b.shape_probs)
        letter_score = sum(a.letter_probs * b.letter_probs)
        shape_color_score = sum(a.shape_col_probs * b.shape_col_probs)
        letter_color_score = sum(a.letter_col_probs * b.letter_col_probs)
        return shape_score * letter_score * shape_color_score * letter_color_score
    
def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch