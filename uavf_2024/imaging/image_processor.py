import numpy as np

from .utils import batched
from .imaging_types import HWC, FullPrediction, Image, InstanceSegmentationResult, TargetDescription
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

    def process_image(self, img: Image) -> list[FullPrediction]:
        '''
        img shape should be (height, width, channels)
        (that tuple order is a placeholder for now and we can change it later, but it should be consistent and we need to keep the docstring updated)
        '''
        if not isinstance(img, Image):
            raise TypeError("img must be an Image object")
        
        if not img.dim_order == HWC:
            raise ValueError("img must be in HWC order")

        shape_results: list[InstanceSegmentationResult] = []

        batch_size = 3
        for tiles in batched(img.generate_tiles(self.tile_size), batch_size):
            temp = self.shape_detector.predict(tiles)
            if temp is not None: shape_results.extend(temp)

        total_results: list[FullPrediction] = []

        batch_size = 5 # these are small images so we can do a lot at once
        for results in batched(shape_results, batch_size):
            zero_padded_letter_silhouttes = []
            for res in results: # These are all linear operations so not parallelized (yet)
                # Color segmentations
                shape_conf = res.confidences
                img_black_bg = res.img * res.mask
                color_seg_result = color_segmentation(img_black_bg) # Can this be parallelized?
                # deteremine the letter mask
                only_letter_mask: np.ndarray = color_seg_result.mask * color_seg_result.mask==2
                w,h = only_letter_mask.shape
                zero_padded_letter_silhoutte = np.zeros((self.letter_size, self.letter_size))
                zero_padded_letter_silhoutte[:w, :h]  = only_letter_mask
                # Add the mask to a list for batch classification
                zero_padded_letter_silhouttes.append(zero_padded_letter_silhoutte)
                # Classify the colors
                shape_color_conf = self.color_classifier.predict(color_seg_result.shape_color)
                letter_color_conf = self.color_classifier.predict(color_seg_result.letter_color)
                # add to total_results
                letter_conf = None
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
            letter_conf = self.letter_classifier.predict(zero_padded_letter_silhouttes)
            # This goes back and changes the respective letter_conf in total_results
            for i in range(len(results)): 
                total_results[-(len(results)-i)].description.letter_probs = letter_conf[i]

        return total_results