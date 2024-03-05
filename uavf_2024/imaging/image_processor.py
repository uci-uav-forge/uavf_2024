from __future__ import annotations
import numpy as np
import os
import cv2 as cv

from .utils import batched
from .imaging_types import HWC, FullBBoxPrediction, Image, InstanceSegmentationResult, ProbabilisticTargetDescriptor
from .letter_classification import LetterClassifier
from .shape_detection import ShapeInstanceSegmenter
from .color_classification import ColorClassifier
from . import profiler
from memory_profiler import profile as mem_profile

def nms_process(shape_results: InstanceSegmentationResult, thresh_iou):
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
    def __init__(self, debug_path: str = None, shape_batch_size = 3, letter_batch_size = 5):
        '''
        Initialize all models here 

        `shape_batch_size` is how many tiles we batch up for shape detection inference
        `letter_batch_size` is how many bounding box crops we batch up for letter classification
        '''
        self.tile_size = 640
        self.letter_size = 128
        self.shape_detector = ShapeInstanceSegmenter(self.tile_size)
        self.letter_classifier = LetterClassifier(self.letter_size)
        self.color_classifier = ColorClassifier()
        self.debug_path = debug_path
        self.thresh_iou = 0.5
        self.num_processed = 0
        self.shape_batch_size = shape_batch_size
        self.letter_batch_size = letter_batch_size

    def process_image(self, img: Image) -> list[FullBBoxPrediction]:
        '''
        img shape should be (height, width, channels)
        (that tuple order is a placeholder for now and we can change it later, but it should be consistent and we need to keep the docstring updated)
        '''
        if not isinstance(img, Image):
            raise TypeError("img must be an Image object")
        
        if not img.dim_order == HWC:
            raise ValueError("img must be in HWC order")

        shape_results: list[InstanceSegmentationResult] = []

        all_tiles = img.generate_tiles(self.tile_size)
        for tiles_batch in batched(all_tiles, self.shape_batch_size):
            temp = self.shape_detector.predict(tiles_batch)
            if temp is not None: shape_results.extend(temp)
        
        shape_results = nms_process(shape_results, self.thresh_iou)

        if self.debug_path is not None:
            local_debug_path = f"{self.debug_path}/img_{self.num_processed}"
            os.makedirs(local_debug_path, exist_ok=True)
            img_to_draw_on = img.get_array().copy()
            for res in shape_results:
                x,y,w,h = res.x, res.y, res.width, res.height
                cv.rectangle(img_to_draw_on, (x,y), (x+w,y+h), (0,255,0), 2)
                cv.putText(img_to_draw_on, str(res.id), (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv.imwrite(f"{local_debug_path}/bounding_boxes.png", img_to_draw_on)

        self.num_processed += 1

        total_results: list[FullBBoxPrediction] = []
        # create debug directory for segmentation and classification
        for results in batched(shape_results, self.letter_batch_size):
            results: list[InstanceSegmentationResult] = results # type hinting
            letter_imgs = []
            for shape_res in results: # These are all linear operations so not parallelized (yet)
                # Color segmentations
                shape_conf = shape_res.confidences
                img_black_bg = shape_res.img * shape_res.mask
                letter_img = cv.resize(img_black_bg.get_array().astype(np.float32), (128,128))
                letter_imgs.append(letter_img)

                if self.debug_path is not None:
                    instance_debug_path = f"{local_debug_path}/det_{shape_res.id}"
                    os.makedirs(instance_debug_path, exist_ok=True)
                    cv.imwrite(f"{instance_debug_path}/input.png", shape_res.img.get_array())
                    cv.imwrite(f"{instance_debug_path}/black_bg.png", img_black_bg.get_array())
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
                    img_id = self.num_processed-1, # at this point it will have been incremented already
                    det_id = shape_res.id
                )
            )
            letter_conf = self.letter_classifier.predict(letter_imgs)
            # "index math hard for grug brain" - Eric
            # Updates letter probs which were previously set to none just in the most recent batch
            for result, conf in zip(total_results[-len(results):], letter_conf):
                result.descriptor.letter_probs = conf
            
        if self.debug_path is not None:
            for result in total_results:
                pred_descriptor_string = str(result.descriptor)
                with open(f"{local_debug_path}/det_{result.det_id}/descriptor.txt", "w") as f:
                    f.write(pred_descriptor_string)

        return total_results
