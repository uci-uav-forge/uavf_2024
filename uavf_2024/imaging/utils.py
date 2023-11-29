from typing import Generator, List

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
    

def sort_payload(list_payload_targets: List[TargetDescription], shape_confusion: np.ndarray , 
                 letter_confusion: np.ndarray, color_confusion = np.ndarray, penalty = True ):
    """
    Sorts a list of TargetDescription instances based on confidence scores calculated using confusion matrices.

    Args:
    - list_payload_targets (List[TargetDescription]): List of TargetDescription instances to be sorted.
    - shape_confusion (np.ndarray): Confusion matrix for shape classification.
    - letter_confusion (np.ndarray): Confusion matrix for letter classification.
    - color_confusion (np.ndarray): Confusion matrix for color classification.
    - penalty (bool, optional): If True, applies nonlinear penalty to confidence scores. Defaults to True.

    Returns:
    - List[TargetDescription]: Ordered version of the list_payload_targets based on confidence scores.
    """
    
    payload_targets_order = {}

    for target_position, payload_target in enumerate(list_payload_targets):
        payload_targets_order[target_position] = 1
        shape_descrp = np.where(payload_target.shape_probs == 1)[0][0]
        letter_descrp = np.where(payload_target.letter_probs == 1)[0][0]
        shape_col_descrp = np.where(payload_target.shape_col_probs == 1)[0][0]
        letter_col_descrp = np.where(payload_target.letter_col_probs == 1)[0][0]

        # Iterate through confusion matrices and corresponding description indices
        for confusion_matrix, description_vector in zip([shape_confusion, letter_confusion, color_confusion, color_confusion],
                                                        [shape_descrp, letter_descrp, shape_col_descrp, letter_col_descrp]):
            descrp_truth_row = confusion_matrix[description_vector]
            descrp_pos_truth = descrp_truth_row[description_vector]

            # Computes the penalty by summing the squares of each negative truth probability
            descrp_neg_truth_penalty = (descrp_truth_row[descrp_truth_row != descrp_pos_truth])**2
            descrp_rank = descrp_pos_truth - np.sum(descrp_neg_truth_penalty)
            descrp_rank = max(descrp_rank, 0.0001) # Ensure non-negative rank

            # Computes the target's confidence score by multiplying with the confidence values for the four descriptions
            if penalty:
                payload_targets_order[target_position] *= descrp_rank
            else:
                payload_targets_order[target_position] *= descrp_pos_truth

    #Reorder the payload target list based on the confidence values
    payload_targets_order =  dict(sorted(payload_targets_order.items(), key=lambda item: item[1], reverse=True))
    payload_ordered_list = [list_payload_targets[target_order] for target_order in payload_targets_order.keys()]

    return payload_ordered_list
 



def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


