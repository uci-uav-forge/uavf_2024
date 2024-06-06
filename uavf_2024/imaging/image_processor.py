from __future__ import annotations
from pathlib import Path
import numpy as np
import os
import cv2 as cv
from .utils import batched
from .imaging_types import HWC, FullBBoxPrediction, Image, DetectionResult, ProbabilisticTargetDescriptor
from .letter_classification import LetterClassifier
from .shape_detection import ShapeDetector
from .color_classification import ColorClassifier
from . import profiler
from memory_profiler import profile as mem_profile

def nms_process(shape_results: DetectionResult, thresh_iou):
    #Given shape_results and some threshold iou, determines if there are intersecting bounding boxes that exceed the threshold iou and takes the
    #box that has the maximum confidence
    boxes = np.array([[shape.x, shape.y, shape.x + shape.width, shape.y + shape.height, max(shape.confidences)] for shape in shape_results])
    if len(boxes) == 0:
        return shape_results
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1] #sorts the scores and gives a sorted list of their respective indices 

    keep = [] #empty array that will define what boundaries we choose to keep

    while len(order) > 0:
        idx = order[0]
        keep.append(shape_results[idx])

        xx1 = np.maximum(x1[idx], x1[order[1:]]) #finds the rightmost left edge between most confident and each of the remaining
        yy1 = np.maximum(y1[idx], y1[order[1:]]) #finds the bottommost top edge between most confident and each of the remaining
        xx2 = np.minimum(x2[idx], x2[order[1:]]) #finds the leftmost right edge between most confident and each of the remaining
        yy2 = np.minimum(y2[idx], y2[order[1:]]) #finds the topmost bottom edge between most confident and each of the remaining

        w = np.maximum(0.0, xx2 - xx1 + 1) #returns the overlap width
        h = np.maximum(0.0, yy2 - yy1 + 1) #returns the overlap height

        intersection = w * h #calculates overlap area for each of the boxes
        rem_areas = areas[order[1:]] #areas of all of the blocks except for the one with the highest confidence interval
        union = (rem_areas - intersection) + areas[idx] #calculates union area

        iou = intersection / union #array of iou for all boxes

        mask = iou < thresh_iou #forms an array of bool values depending on whether the particular bounding box exceeds threshold iou
        order = order[1:][mask] #forms an array, excluding the boundary with highest confidence and boundaries that exceed threshold iou

    return keep



class ImageProcessor:
    def __init__(self, 
    debug_path: str | Path | None = None, 
    shape_batch_size = 3, 
    letter_batch_size = 5,
    tile_size = 1080,
    min_tile_overlap = 0,
    conf = 0.05
    ):
        '''
        Initialize all models here 

        `shape_batch_size` is how many tiles we batch up for shape detection inference
        `letter_batch_size` is how many bounding box crops we batch up for letter classification
        '''
        self.tile_size = tile_size
        self.min_tile_overlap = min_tile_overlap
        self.shape_det_weights = None
        self.letter_size = 128
        self.shape_detector = ShapeDetector(self.tile_size, conf)
        self.letter_classifier = LetterClassifier(self.letter_size)
        self.color_classifier = ColorClassifier()
        
        if debug_path:
            self.debug_path = Path(debug_path)
            
            if not self.debug_path.exists():
                self.debug_path.mkdir(parents=True)
            elif not self.debug_path.is_dir():
                raise FileExistsError(f"{str(self.debug_path)} already exists and is not a directory.")
        
        self.thresh_iou = 0.5
        self.num_processed = 0
        self.shape_batch_size = shape_batch_size
        self.letter_batch_size = letter_batch_size

    def get_last_logs_path(self):
        return f"{self.debug_path}/img_{self.num_processed-1}"
    
    def reset_log_directory(self, new_debug_path: str):
        self.debug_path = new_debug_path
        self.num_processed = 0
        os.makedirs(self.debug_path, exist_ok=True)

    def _make_shape_detection(self, img : Image) -> list[DetectionResult]:
        shape_results: list[DetectionResult] = []

        all_tiles = [img.as_tile()]#img.generate_tiles(self.tile_size, self.min_tile_overlap)
        for tiles_batch in batched(all_tiles, self.shape_batch_size):
            temp = self.shape_detector.predict(tiles_batch)
            if temp is not None: shape_results.extend(temp)
        
        shape_results = nms_process(shape_results, self.thresh_iou)

        return shape_results
    
    def _classify_color_and_char(self, shape_results : list[DetectionResult]):
        total_results: list[FullBBoxPrediction] = []
        # create debug directory for segmentation and classification
        for results in batched(shape_results, self.letter_batch_size):
            results: list[DetectionResult] = results # type hinting
            letter_imgs = []
            for shape_res in results: # These are all linear operations so not parallelized (yet)
                # Color segmentations
                shape_conf = shape_res.confidences
                letter_img = cv.resize(shape_res.img.get_array().astype(np.float32), (128,128))
                letter_imgs.append(letter_img)

                # Classify the colors
                letter_color_conf, shape_color_conf = self.color_classifier.predict(letter_img)

                # add to total_results
                letter_conf = None
                total_results.append(
                FullBBoxPrediction(
                    shape_res.x,
                    shape_res.y,
                    shape_res.width,
                    shape_res.height,
                    ProbabilisticTargetDescriptor(
                        shape_conf,
                        letter_conf,
                        shape_color_conf,
                        letter_color_conf
                    ),
                    img_id = self.num_processed,
                    det_id = shape_res.id
                )
            )
            letter_conf = self.letter_classifier.predict(letter_imgs)
            # "index math hard for grug brain" - Eric
            # Updates letter probs which were previously set to none just in the most recent batch
            for result, conf in zip(total_results[-len(results):], letter_conf):
                result.descriptor.letter_probs = conf
            
        return total_results
    
    def process_image(self, img: Image) -> list[FullBBoxPrediction]:
        '''
        img shape should be (height, width, channels)
        (that tuple order is a placeholder for now and we can change it later, but it should be consistent and we need to keep the docstring updated)
        '''
        if not isinstance(img, Image):
            raise TypeError("img must be an Image object")
        if not img.dim_order == HWC:
            raise ValueError("img must be in HWC order")
        
        shape_results = self._make_shape_detection(img)
        total_results = self._classify_color_and_char(shape_results)

        if self.debug_path is not None:
            local_debug_path = f"{self.debug_path}/img_{self.num_processed}"
            os.makedirs(local_debug_path, exist_ok=True)
            for shape_res in shape_results:
                path = f"{local_debug_path}/det_{shape_res.id}"
                os.makedirs(path, exist_ok=True)
                cv.imwrite(f"{path}/input.png", shape_res.img.get_array())

            img_to_draw_on = img.get_array().copy()
            for result in total_results:
                x,y,w,h = result.x, result.y, result.width, result.height
                cv.rectangle(img_to_draw_on, (x,y), (x+w,y+h), (0,255,0), 2)
                shape_col, shape, letter_col, letter = str(result.descriptor.collapse_to_certain()).split(" ")
                shape_col_prob = max(result.descriptor.shape_col_probs)
                shape_prob = max(result.descriptor.shape_probs)
                letter_col_prob = max(result.descriptor.letter_col_probs)
                letter_prob = max(result.descriptor.letter_probs)
                cv.putText(img_to_draw_on, f"{result.det_id}", (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv.putText(img_to_draw_on, f"{shape_col}: {shape_col_prob:.03f}", (x+w,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv.putText(img_to_draw_on, f"{shape}: {shape_prob:.03f}", (x+w,y+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv.putText(img_to_draw_on, f"{letter_col}: {letter_col_prob:.03f}", (x+w,y+40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv.putText(img_to_draw_on, f"{letter}: {letter_prob:.03f}", (x+w,y+60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                pred_descriptor_string = str(result.descriptor)
                with open(f"{local_debug_path}/det_{result.det_id}/descriptor.txt", "w") as f:
                    f.write(pred_descriptor_string)
            cv.imwrite(f"{local_debug_path}/bounding_boxes.png", img_to_draw_on)

        self.num_processed += 1
        return total_results
    
    def process_image_lightweight(self, img : Image) -> list[FullBBoxPrediction]:
        '''
        Processes image and runs shape detection
        Only classifies if there is more than one detection.
        Use case: Zooming in and localizing onto a specific target.
        '''
        if not isinstance(img, Image):
            raise TypeError("img must be an Image object")
        if not img.dim_order == HWC:
            raise ValueError("img must be in HWC order")
        
        shape_results = self._make_shape_detection(img)
        self.num_processed += 1

        if len(shape_results) == 1:
            # Returns shape_results cast as FullBBoxPrediction with no probabilistic target descriptor
            return [FullBBoxPrediction(
                    shape_results.x,
                    shape_results.y,
                    shape_results.width,
                    shape_results.height,
                    ProbabilisticTargetDescriptor(),
                    img_id = self.num_processed,
                    det_id = shape_results.id
                )]
        total_results = self._classify_color_and_char(shape_results)
        return total_results
