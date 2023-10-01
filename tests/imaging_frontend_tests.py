import unittest
from uavf_2024.imaging.image_processor import ImageProcessor
import cv2 as cv
import os

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class TestImagingFrontend(unittest.TestCase):
    def setUp(self) -> None:
        self.image_processor = ImageProcessor()

    def test_runs_without_crashing(self):
        sample_input = cv.imread(f"{CURRENT_FILE_PATH}/images/gopro-image-5k.png")
        res = self.image_processor.process_image(sample_input)