import numpy as np

from uavf_2024.imaging.utils import generate_tiles
from .imaging_types import FullPrediction,InstanceSegmentationResult
from .letter_classification import LetterClassifier
from .shape_detection import ShapeInstanceSegmenter
from .color_segmentation import color_segmentation
from .color_classification import ColorClassifier

class ImageProcessor:
    def __init__(self):
        '''
        Initialize all models here 
        '''
        self.tile_size = 640
        self.letter_size = 128
        self.shape_detector = ShapeInstanceSegmenter(self.tile_size)
        self.letter_classifier = LetterClassifier(self.letter_size)
        self.color_classifier = ColorClassifier()

    def process_image(self, img: np.ndarray) -> "list[FullPrediction]":
        '''
        img shape should be (height, width, channels)
        (that tuple order is a placeholder for now and we can change it later, but it should be consistent and we need to keep the docstring updated)
        '''

        tiles = generate_tiles(img, self.tile_size)

        shape_results: list[InstanceSegmentationResult] = []

        for tile in tiles:
            # TODO re-implement batch processing
            shapes_detected = self.shape_detector.predict(tile.img)
            for shape in shapes_detected:
                shape.x+=tile.x
                shape.y+=tile.y
                shape_results.append(shape)

        total_results: list[FullPrediction] = []

        for res in shape_results:
            shape_conf = res.confidences

            img_black_bg = res.img * res.mask
            color_seg_result = color_segmentation(img_black_bg)

            only_letter_mask: np.ndarray = color_seg_result.mask * color_seg_result.mask==2
            w,h = only_letter_mask.shape
            zero_padded_letter_silhoutte = np.zeros((self.letter_size, self.letter_size))
            zero_padded_letter_silhoutte[:w, :h]  = only_letter_mask
            # TODO: also do batch processing for letter classification
            letter_conf = self.letter_classifier.predict(zero_padded_letter_silhoutte)

            shape_color_conf = self.color_classifier.predict(color_seg_result.shape_color)
            letter_color_conf = self.color_classifier.predict(color_seg_result.letter_color)

            total_results.append(
                FullPrediction(
                    res.x,
                    res.y,
                    res.width,
                    res.height,
                    TargetDescription(
                        shape_conf,
                        letter_conf,
                        shape_color_conf,
                        letter_color_conf
                    )
                )
            )

        return total_results