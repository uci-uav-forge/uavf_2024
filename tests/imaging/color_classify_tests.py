import unittest
from uavf_2024.imaging.color_classification import ColorClassifier, COLORS_TO_RGB
import numpy as np

class ColorClassificationTest(unittest.TestCase):
    def test_blue_color_classification(self):
        '''
        Test case for blue color classification. 
        This specific shade of blue was misclassified as purple
        when we first made the simple KNN color classifier with 1 example per color.
        '''
        classifier = ColorClassifier()
        test_color = (37, 104, 143)
        blue_index = list(COLORS_TO_RGB.keys()).index('blue')
        scores = classifier.predict(np.array(test_color))

        self.assertEquals(np.argmax(scores), blue_index) 



if __name__ == '__main__':
    unittest.main()