import numpy as np
import os
import cv2 as cv
import time

from .utils import batched
from .imaging_types import HWC, FullPrediction, Image, InstanceSegmentationResult, TargetDescription
from .letter_classification import LetterClassifier
from .shape_detection import ShapeInstanceSegmenter
from .color_segmentation import color_segmentation
from .color_classification import ColorClassifier

class ImageProcessor:
    def __init__(self, debug_path: str = None):
        '''
        Initialize all models here 
        '''
        self.tile_size = 640
        self.letter_size = 128
        self.shape_detector = ShapeInstanceSegmenter(self.tile_size)
        self.letter_classifier = LetterClassifier(self.letter_size)
        self.color_classifier = ColorClassifier()
        self.debug_path = debug_path

    def process_image(self, img: Image) -> list[FullPrediction]:
        '''
        img shape should be (height, width, channels)
        (that tuple order is a placeholder for now and we can change it later, but it should be consistent and we need to keep the docstring updated)
        '''
        if not isinstance(img, Image):
            raise TypeError("img must be an Image object")
        
        if not img.dim_order == HWC:
            raise ValueError("img must be in HWC order")

        if self.debug_path is not None:
            local_debug_path = f"{self.debug_path}/{time.strftime(r'%m-%d-%H-%M-%S')}"
            os.makedirs(local_debug_path, exist_ok=True)
            cv.imwrite(f"{local_debug_path}/original.png", img.get_array())

        shape_results: list[InstanceSegmentationResult] = []

        TILES_BATCH_SIZE = 3
        for tiles in batched(img.generate_tiles(self.tile_size), TILES_BATCH_SIZE):
            temp = self.shape_detector.predict(tiles)
            if temp is not None: shape_results.extend(temp)

        total_results: list[FullPrediction] = []
        if self.debug_path is not None:
            os.makedirs(f"{local_debug_path}/shape_detection", exist_ok=True)
            img_to_draw_on = img.get_array().copy()
            for res in shape_results:
                x,y,w,h = res.x, res.y, res.width, res.height
                cv.rectangle(img_to_draw_on, (x,y), (x+w,y+h), (0,255,0), 2)
            cv.imwrite(f"{local_debug_path}/shape_detection/bounding_boxes.png", img_to_draw_on)


        SHAPES_BATCH_SIZE = 5 # these are small images so we can do a lot at once

        # create debug directory for segmentation and classification
        if self.debug_path is not None:
            os.makedirs(f"{local_debug_path}/segmentation", exist_ok=True)
            os.makedirs(f"{local_debug_path}/letter_classification", exist_ok=True)
        for results in batched(shape_results, SHAPES_BATCH_SIZE):
            results: list[InstanceSegmentationResult] = results # type hinting
            zero_padded_letter_silhouttes = []
            for shape_res in results: # These are all linear operations so not parallelized (yet)
                # Color segmentations
                shape_conf = shape_res.confidences
                img_black_bg = shape_res.img * shape_res.mask
                color_seg_result = color_segmentation(img_black_bg) # Can this be parallelized?

                # deteremine the letter mask
                only_letter_mask: np.ndarray = color_seg_result.mask * (color_seg_result.mask==2)
                w,h = only_letter_mask.shape
                zero_padded_letter_silhoutte = np.zeros((self.letter_size, self.letter_size))
                zero_padded_letter_silhoutte[:w, :h] = only_letter_mask

                # Add the mask to a list for batch classification
                zero_padded_letter_silhouttes.append(zero_padded_letter_silhoutte)
                # Save the color segmentation results
                if self.debug_path is not None:
                    num_files = len(os.listdir(f"{local_debug_path}/letter_classification"))
                    cv.imwrite(f"{local_debug_path}/letter_classification/{num_files}.png", zero_padded_letter_silhoutte*127)
                    cv.imwrite(f"{local_debug_path}/segmentation/{num_files}_input.png", img_black_bg.get_array())
                    cv.imwrite(f"{local_debug_path}/segmentation/{num_files}_output.png", color_seg_result.mask*127)
                # Classify the colors
                shape_color_conf = self.color_classifier.predict(color_seg_result.shape_color)
                letter_color_conf = self.color_classifier.predict(color_seg_result.letter_color)
                # add to total_results
                letter_conf = None
                total_results.append(
                FullPrediction(
                    shape_res.x,
                    shape_res.y,
                    shape_res.width,
                    shape_res.height,
                    TargetDescription(
                        shape_conf,
                        letter_conf,
                        shape_color_conf,
                        letter_color_conf
                    )
                )
            )
            letter_conf = self.letter_classifier.predict(zero_padded_letter_silhouttes)
            # "index math hard for grug brain" - Eric
            # Updates letter probs which were previously set to none just in the most recent batch
            for result, conf in zip(total_results[-len(results):], letter_conf):
                result.description.letter_probs = conf

        return total_results