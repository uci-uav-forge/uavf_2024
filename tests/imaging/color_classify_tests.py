import unittest
from uavf_2024.imaging.color_classification import ColorClassifier, COLORS_TO_RGB
import numpy as np
import cv2 as cv
from PIL import Image
import numpy as np
import os
CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
class ColorClassificationTest(unittest.TestCase):
    def test_blue_color_classification(self):
        '''
        Test case for blue color classification. 
        This specific shade of blue was misclassified as purple
        when we first made the simple KNN color classifier with 1 example per color.
        '''
        image_path = CURRENT_FILE_PATH / 'imaging_data/fake_dataset/red_green.jpg'
        image = Image.open(image_path)

        image_array = np.array(image)
        resized_image = image_array.reshape((1, 128, 128, 3))
        classifier = ColorClassifier()
        shape_score, letter_score = classifier.predict(resized_image)
        print(shape_score)
        
        green = list(COLORS_TO_RGB.keys()).index('green')
        
        red = list(COLORS_TO_RGB.keys()).index('red')
        
        self.assertEqual(np.argmax(letter_score), green) 
        self.assertEqual(np.argmax(shape_score), red) 



if __name__ == '__main__':
    unittest.main()