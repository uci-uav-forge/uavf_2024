from typing import Generator, List

import numpy as np
import torch

from .imaging_types import ProbabilisticTargetDescriptor, Tile

from itertools import islice

def normalize_distribution( b: ProbabilisticTargetDescriptor, offset):
    shape_norm_prob = (b.shape_probs + offset)/sum(b.shape_probs + offset)
    letter_norm_prob = (b.letter_probs + offset)/sum(b.letter_probs + offset)
    shape_col_norm_prob = (b.shape_col_probs + offset)/sum(b.shape_col_probs + offset)
    letter_col_norm_prob = (b.letter_col_probs + offset)/sum(b.letter_col_probs + offset)
    
    return ProbabilisticTargetDescriptor(shape_norm_prob, letter_norm_prob, shape_col_norm_prob, letter_col_norm_prob)


def calc_match_score(a: ProbabilisticTargetDescriptor, b: ProbabilisticTargetDescriptor):
        '''
        Returns a number between 0 and 1 representing how likely the two descriptions are the same target
        
        '''
        b = normalize_distribution(b, 1e-4)
        shape_score = sum(a.shape_probs * b.shape_probs)
        letter_score = sum(a.letter_probs * b.letter_probs)
        shape_color_score = sum(a.shape_col_probs * b.shape_col_probs)
        letter_color_score = sum(a.letter_col_probs * b.letter_col_probs)
        return shape_score * letter_score * shape_color_score * letter_color_score
    

def sort_payload(list_payload_targets: List[ProbabilisticTargetDescriptor], shape_confusion: np.ndarray , 
                 letter_confusion: np.ndarray, color_confusion = np.ndarray, penalty = False ):
    """
    Sorts a list of TargetDescription instances based on confidence scores calculated using confusion matrices.
    Currently, the penalty algorithm evaluates the confusion between each payload target class and all other classes within the truth row.
    The algorithm will be revisited later.
    
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
    if penalty:
        print("Caution: The penalty algorithm is implemented, but its accuracy is not confirmed")

    for target_position, payload_target in enumerate(list_payload_targets):
        payload_targets_order[target_position] = 1
        shape_class_index = np.where(payload_target.shape_probs == 1)[0][0]
        arr = payload_target.letter_probs
        if np.all(arr.flatten() == arr.flatten()[0]):
            print("letter" )
            letter_class_index = -1
            shape_col_class_index = -1
            letter_col_class_index = -1

        else: 
            letter_class_index = np.where(payload_target.letter_probs == 1)[0][0]
            shape_col_class_index = np.where(payload_target.shape_col_probs == 1)[0][0]
            letter_col_class_index = np.where(payload_target.letter_col_probs == 1)[0][0]

        # Iterate through confusion matrices and corresponding description indices
        for confusion_matrix, class_index in zip([shape_confusion, letter_confusion, color_confusion, color_confusion],
                                                        [shape_class_index, letter_class_index, shape_col_class_index, letter_col_class_index]):
            
            if class_index == -1:
                print( " ") #place holder, i don't know how to handle person for confusion matrix brb
            else:
                descrp_truth_row = confusion_matrix[class_index]
                descrp_pos_truth = descrp_truth_row[class_index]
                # Computes the penalty by summing the squares of each negative truth probability
                descrp_neg_truth_penalty = (descrp_truth_row[descrp_truth_row != descrp_pos_truth])**2
                descrp_score = descrp_pos_truth - np.sum(descrp_neg_truth_penalty)
                descrp_score = max(descrp_score, 0.0001) # Ensure non-negative score after the penalty

            # Computes the target's confidence score by multiplying with the confidence values for the four descriptions
                if penalty:
                    payload_targets_order[target_position] *= descrp_score
                else:
                    payload_targets_order[target_position] *= descrp_pos_truth

    #Reorder the payload target list based on the confidence values
    payload_targets_order =  sorted(payload_targets_order.items(), key=lambda item: item[1], reverse=True)
    payload_ordered_list = [list_payload_targets[target_order] for target_order, target_score in payload_targets_order]

    return payload_ordered_list

def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def make_ortho_vectors(v: torch.Tensor, m: int):
    '''
    `v` is a (n,3) tensor
    make m unit vectors that are orthogonal to each v_i, and evenly spaced around v_i's radial symmetry
    
    to visualize: imagine each v_i is the vector coinciding 
    with a lion's face direction, and we wish to make m vectors for the lion's mane.

    it does this by making a "lion's mane" around the vector (0,0,1), which is easy with parameterizing
    with theta and using (cos(theta), sin(theta), 0). Then, it figures out the 2DOF R_x @ R_y rotation matrix
    that would rotate (0,0,1) into v_i, and applies it to those mane vectors.

    returns a tensor of shape (n,m,3)
    '''
    n = v.shape[0]
    thetas = torch.linspace(0, 2*torch.pi, m).to(v.device)

    phi_y = torch.atan2(v[:, 0], v[:, 2])
    square_sum = v[:,0]**2 + v[:,2]**2
    inverted = 1/torch.sqrt(square_sum)#fast_inv_sqrt(square_sum)
    phi_x = torch.atan(v[:, 1] * inverted) # This line is responsible for like 20-25% of the runtime of this function, so unironically if we implement fast inverse square root in pytorch we can get huge performance gains

    cos_y = torch.cos(phi_y)
    sin_y = torch.sin(phi_y)
    cos_x = torch.cos(phi_x)
    sin_x = torch.sin(phi_x)


    R = torch.stack(
            [cos_y, -sin_y*sin_x, sin_y*cos_x,
            torch.zeros_like(cos_x), cos_x, sin_x,
            -sin_y, -cos_y*sin_x, cos_y*cos_x]
    ).T.reshape(n,3,3)
    # (n,3,3)


    vectors = torch.stack(
        [
            torch.cos(thetas), 
            torch.sin(thetas), 
            torch.zeros_like(thetas)
        ],
    ) # (3,m)

    return torch.matmul(R, vectors).permute(0, 2, 1) # (n, m, 3)
