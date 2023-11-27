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
    

def sort_payload(a: list, b: dict, method = 2):
    """
    Given a list of TargetDescription instances (a) and a dictionary of three model's confusion matrices (b),
    along with the chosen method, this function returns an ordered version of 'a' based on confidence
    calculated using the specified method.
    """
    diag_search_targets = {}
    penalized_search_targets = {}
    attribute_names = ['shape_probs', 'letter_probs', 'shape_col_probs', 'letter_col_probs']
    conf_matrix = ['shape', 'letter', 'shape_col', 'letter_col']

    for attribute_name, confusion_matrix_key in zip(attribute_names, conf_matrix):
         for target_index, search_target in enumerate(a):
            # Determine the index where the current attribute is true in the target description
            search_targ_attr_value = np.where( getattr(search_target, attribute_name) == 1 )[0][0]

            # Use the determined index to look up the corresponding truth row in the confusion matrix
            conf_truth_row = b[confusion_matrix_key][search_targ_attr_value]

            # Calculate diagonal confidence and individual penalties for each class error
            diag_confid = conf_truth_row[ search_targ_attr_value]
            error_values = conf_truth_row[conf_truth_row != conf_truth_row[search_targ_attr_value]]
            nonlinear_penalty = error_values**2
            attr_rank = diag_confid - (np.sum(nonlinear_penalty))

            # Ensure non-negative rank, if considering small diagonal confidence values
            attr_rank = max(attr_rank, 0.00001) 
            
            #Store results in dictionary
            if target_index in penalized_search_targets.keys():
                penalized_search_targets[target_index].append(attr_rank)
                diag_search_targets[target_index].append( diag_confid )
            else:
                penalized_search_targets[target_index] = [attr_rank]
                diag_search_targets[target_index] = [diag_confid]

    # Calculate total confidence by taking the product across different attributes' ranks
    diag_prod_search_target = {x: np.prod(rank_list) for x,rank_list in diag_search_targets.items()}
    penalized_prod_search_targets = {x: np.prod(rank_list) for x,rank_list in penalized_search_targets.items()}

    #Sort from greatest to least
    sort_diag_search_target =  dict(sorted(diag_search_targets.items(), key=lambda item: item[1], reverse=True))
    sort_penalized_search_target =  dict(sorted(penalized_prod_search_targets.items(), key=lambda item: item[1], reverse=True))
    
    # Return the ordered list for the payload based on the specified method:
    # 1) Confidence determined by diagonal values or positive probabilities
    # 2) Penalized confidence

    if method == 1:
        sorted_search_targets = [a[ordered_index] for ordered_index in sort_diag_search_target.keys()]
    if method == 2:
        sorted_search_targets = [a[ordered_index] for ordered_index in sort_penalized_search_target.keys()]


    return sorted_search_targets #return the list of target descriptions but reordered
 



def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch



if __name__ == "__main__":
    '''Testing initial payload'''
    np.random.seed(42)

    test_search_list = [
            TargetDescription(
                np.eye(13)[1],
                np.eye(35)[2],
                np.eye(8)[3],
                np.eye(8)[4],
            ),
            TargetDescription(
                np.eye(13)[8],
                np.eye(35)[7],
                np.eye(8)[6],
                np.eye(8)[5],
            ),
            TargetDescription(
                np.eye(13)[4],
                np.eye(35)[3],
                np.eye(8)[2],
                np.eye(8)[1],
            )
    ]

    # Generate a test confusion matrix for each target attribute, with rows and columns normalized to sum to 1    
    color_confusion_matrix = np.random.dirichlet(np.ones(8), size=8).T
    test_conf_matrice = { 'shape': np.random.dirichlet(np.ones(13), size=13).T,\
                          'letter': np.random.dirichlet(np.ones(35), size=35).T,\
                          'shape_col': color_confusion_matrix, \
                          'letter_col': color_confusion_matrix }
    
    # Normalizing the test confusion matrix by each column and then each row, 
    # making sure that each truth row sums up to 1 and each predicted column sums close to 1 with little deviation
    for every_key in test_conf_matrice.keys():
        test_conf_matrice[every_key] /= test_conf_matrice[every_key].sum(axis=1, keepdims=True)
        test_conf_matrice[every_key] /= test_conf_matrice[every_key].sum(axis=0, keepdims=True)
    
        
    ordered_test_payload = sort_payload(a = test_search_list, b = test_conf_matrice)

