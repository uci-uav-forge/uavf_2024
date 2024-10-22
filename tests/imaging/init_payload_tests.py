import unittest
from uavf_2024.imaging.utils import sort_payload
from uavf_2024.imaging.imaging_types import ProbabilisticTargetDescriptor
import numpy as np

class TestCameraCalibration(unittest.TestCase):
    def test_functionality(self):

        np.random.seed(42)

        test_search_list = [
                ProbabilisticTargetDescriptor(
                    np.eye(9)[1],
                    np.eye(36)[2],
                    np.eye(8)[3],
                    np.eye(8)[4],
                ),
                ProbabilisticTargetDescriptor(
                    np.eye(9)[8],
                    np.eye(36)[7],
                    np.eye(8)[6],
                    np.eye(8)[5],
                ),
                ProbabilisticTargetDescriptor(
                    np.eye(9)[4],
                    np.eye(36)[3],
                    np.eye(8)[2],
                    np.eye(8)[1],
                )
        ]

        color_confusion_matrix = np.random.dirichlet(np.ones(8), size=8).T
        shape_confusion_matrix = np.random.dirichlet(np.ones(9), size=13).T
        letter_confusion_matrix = np.random.dirichlet(np.ones(36), size=36).T
            
        ordered_test_payload = sort_payload(list_payload_targets= test_search_list, 
                                            shape_confusion= shape_confusion_matrix,
                                            letter_confusion= letter_confusion_matrix,
                                            color_confusion= color_confusion_matrix)
        
    def test_absolute_targets(self):
        '''Checks if the algorithm will first select target_1 with a higher color confidence score over target_2.
        '''
        np.random.seed(42)
        target_1 = ProbabilisticTargetDescriptor(
                    np.eye(9)[1],
                    np.eye(36)[2],
                    np.eye(8)[5],
                    np.eye(8)[6])
        target_2 = ProbabilisticTargetDescriptor(
                    np.eye(9)[8],
                    np.eye(36)[7],
                    np.eye(8)[0],
                    np.eye(8)[1])
        
        payload_list = [target_2, target_1]

        color_confusion_matrix = np.array([ [29, 20, 2, 0, 1, 0, 1, 1],   
                                            [10, 28, 5, 2, 1, 1, 0, 1],  
                                            [3, 5, 30, 1, 2, 0, 0, 1],   
                                            [0, 1, 2, 31, 2, 1, 0, 1],   
                                            [2, 1, 3, 2, 32, 1, 0, 1],  
                                            [0, 2, 1, 1, 1, 33, 0, 1],   
                                            [1, 0, 0, 0, 0, 0, 34, 0],   
                                            [1, 1, 1, 1, 1, 1, 1, 35] ]) 
        shape_confusion_matrix = np.round(np.eye(9) )
        letter_confusion_matrix = np.round(np.eye(36))

        confusion_matrices = [shape_confusion_matrix, letter_confusion_matrix, color_confusion_matrix]
        # Normalize all matrices with respects to their columns
        confusion_matrices = [np.round(matrix / matrix.sum(axis=1, keepdims=True), decimals = 4) for matrix in confusion_matrices]

        ordered_test_payload = sort_payload(list_payload_targets= payload_list, 
                                                    shape_confusion= confusion_matrices[0],
                                                    letter_confusion= confusion_matrices[1],
                                                    color_confusion= confusion_matrices[2])
        
        assert( ordered_test_payload == [target_1, target_2])



