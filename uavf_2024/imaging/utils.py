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
    

def sort_payload(a: list, confusion_matrice: dict, method = 1):
    '''b: dictionary of np.ndarray for readability purposes
    'shape_m' 'letter_m' 'shape_clr_m' 'letter_clr_m
    List(np.ndarray) = [ shape, letter, shape color, letter color] <- order of confusion matrix
    confusion matrix has rows as the truth and the columns are predictions

    Given a list of the five search target descriptions and a list of confusion matrice given as a 2D numpy array,
    use the 4 description attributes from each target to locate the confidence in each confusion matrix,

    Proposed methods: 
    1) use the confidence defined by the diagonal values of the confusion matrix and subtract a nonlinear penalty 
       determined by false positives

    '''
    #parameters: given a list of target descriptions , given a list of confusion matrixes
    #return the same list but ordered

    sorted_search_targets = []

    if method == 1:
        alpha = 1 # weight of the sum of the truth_row[other_targets_traits]
        beta = 0 # weight of the sum of the predict_col[other_targets_traits]
        search_descr = np.array([[search_target.shape_probs, search_target.letter_probs, \
                                  search_target.shape_col_probs, search_target.letter_col_probs] for search_target in a])
        
        non_zero_indices = np.argwhere(search_descr == 1)
        list_instances = [] #store the search_target description with the result value as a tuple
        for target_index, search_target in enumerate(a):
            transpose_descrp = non_zero_indices.T  #rows are the trait category and the column are the corresponding target trait
            #shape
            #for characteristic in range(0,4):
                # later replace characteristic at 0
            no_repeat_list = transpose_descrp[0][transpose_descrp[target_index] != transpose_descrp[0, target_index]]
            sum_search_confusion = np.sum( np.array(( [confusion_matrice['shape_m'][transpose_descrp[0,target_index], other_target] for other_target in no_repeat_list])) )
            shape_result = confusion_matrice['shape_m'][transpose_descrp[0,target_index], transpose_descrp[0,target_index]] \
                            - alpha * (sum_search_confusion **2 )
            #letter

            #no_repeat_list = transpose_descrp[1][transpose_descrp[1] != transpose_descrp[1, target_index]]
            #sum_search_confusion = np.sum( np.array(( [confusion_matrice['letter_m'][transpose_descrp[1,target_index], other_target] for other_target in no_repeat_list])) )
            #letter_result = confusion_matrice['letter_m'][transpose_descrp[1,target_index], transpose_descrp[1,target_index]] - alpha * (sum_search_confusion **2) 
            
            no_repeat_list = transpose_descrp[1][transpose_descrp[1] != transpose_descrp[1, target_index]]
            sum_search_confusion = np.sum( np.array(( [confusion_matrice['letter_m'][transpose_descrp[1,target_index], other_target] for other_target in no_repeat_list])) )
            letter_result = confusion_matrice['letter_m'][transpose_descrp[1,target_index], transpose_descrp[1,target_index]] - alpha * (sum_search_confusion **2 )

            no_repeat_list = transpose_descrp[2][transpose_descrp[2] != transpose_descrp[2, target_index]]
            sum_search_confusion = np.sum( np.array(( [confusion_matrice['shape_clr_m'][transpose_descrp[2,target_index], other_target] for other_target in no_repeat_list])) )
            shape_clr_result = confusion_matrice['shape_clr_m'][transpose_descrp[2,target_index], transpose_descrp[1,target_index]] - alpha * (sum_search_confusion**2 )


            no_repeat_list = transpose_descrp[1][transpose_descrp[3] != transpose_descrp[3, target_index]]
            sum_search_confusion = np.sum( np.array( [confusion_matrice['letter_clr_m'][transpose_descrp[3,target_index], other_target] for other_target in no_repeat_list] ) )
            letter_clr_result = confusion_matrice['letter_clr_m'][transpose_descrp[3,target_index], transpose_descrp[3,target_index]] - alpha * (sum_search_confusion**2 )

            target_rank = shape_result * letter_result * shape_clr_result * letter_clr_result
            #sc etc etc
            list_instances.append(tuple(search_target, target_rank))

        sorted_instances = sorted(list_instances, key=lambda x: x[1])
            #lc
            # map t he ending value to the target: then later sort it
                         

        if method == 2:
            ''' method 2 will be an attempt to use the L2 score to sort the targets. this would take into consideration the
            false positives and false negatives because there is not a max 5 targets on the field and really don't want to load a bottle that
            has a high risk of dropping on the wrong one that's not even on the list,
            this is only to be implemented after method 1 is completed and tested'''
            sorted_search_targets = 0
        print(" shape of indices ", non_zero_indices)
        #imagine [ [1,2,3,4] [ 5,4,3,2] etc ] <- column 1 shape, column 2 letter, etc


         


    '''
    for search_target in a: 
         assert(isinstance(search_target, TargetDescription))
         the attribute to search for is the index that has the value 1 as its probability
         shape_trait = np.where(search_target.shape_probs == 1)
         letter_trait = np.where(search_target.letter_probs == 1)

         shape_col_trait = np.where(search_target.shape_col_probs == 1)
         letter_col_trait = np.where(search_target.letter_col_probs == 1)

         access diagonal values
         shape_confidence = b[0][shape_trait][shape_trait]
         letter_confidence = b[1][letter_trait][letter_trait]
         shape_col_confidence = b[2][shape_col_trait][shape_col_trait]
         letter_col_confidence = b[3][letter_col_trait][letter_col_trait]

         shape_trust = shape_confidence - (1- shape_confidence)**2
         letter_trust = letter_trait - (1- letter_confidence)**2
         shape_col_trust = shape_col_confidence - (1- shape_col_confidence)**2
         letter_col_trust = letter_col_confidence - (1- letter_col_confidence)**2
         '''
    












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
    sort_payload(a = test_search_list)
    #test_confus_matrix = []