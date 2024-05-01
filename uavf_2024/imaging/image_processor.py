from __future__ import annotations

import itertools
import warnings
from typing import Iterable

import numpy as np
import os
import cv2 as cv

from .general_classifier import GeneralClassifier
from .utils import batched
from .imaging_types import HWC, FullBBoxPrediction, Image, DetectionResult, ProbabilisticTargetDescriptor, Tile
from .letter_classification import LetterClassifier
from .detection import ShapeDetector, OneDetector
from .color_classification import ColorClassifier
from . import profiler
from memory_profiler import profile as mem_profile


def non_max_suppress(shape_results: list[DetectionResult], thresh_iou: float) -> list[DetectionResult]:
    """
    Given shape_results and some threshold iou, determines if there are intersecting bounding boxes
    that exceed the threshold iou and takes the box that has the maximum confidence
    """
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


class NewImageProcessor:
    """
    Image processing pipeline implementing detection-classification two-shot architecture.
    """
    TILE_SIZE = 640
    IOU_THRESHOLD = 0.5

    def __init__(self, debug_dir: str = "debug/img_proc/", detector_batch_size=4, classifier_batch_size=8):
        self.debug_dir = debug_dir
        self.detector_batch_size = detector_batch_size
        self.classifier_batch_size = classifier_batch_size

        self.yolo_detector = OneDetector()
        self.general_classifier = GeneralClassifier()

        self.num_processed = 0

    def _detect_targets(self, img: Image, tile_min_overlap=64, confidence_threshold=0.25):
        tiles = img.generate_tiles(self.TILE_SIZE, tile_min_overlap)
        batched_tiles: Iterable[tuple[Tile]] = batched(tiles, self.detector_batch_size)

        predictions: Iterable[DetectionResult] = itertools.chain(
            *(self.yolo_detector.predict(batch, confidence_threshold) for batch in batched_tiles)
        )

        nms_predictions = non_max_suppress(list(predictions), self.IOU_THRESHOLD)

        if self.debug_dir is not None:
            self.write_debug_img(img, nms_predictions)

        return nms_predictions

    def write_debug_img(self, img: Image, detection_results: list[DetectionResult]) -> None:
        if self.debug_dir is None:
            warnings.warn("No debug directory set, skipping debug image writing")
            return

        debug_path = f"{self.debug_dir}/img_{self.num_processed}"
        os.makedirs(debug_path, exist_ok=True)
        img_to_draw_on = img.get_array().copy()
        for res in detection_results:
            x, y, w, h = res.x, res.y, res.width, res.height
            cv.rectangle(img_to_draw_on, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(img_to_draw_on, "target", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imwrite(f"{debug_path}/bounding_boxes.png", img_to_draw_on)

    def _classify_targets(
        self,
        img: Image,
        detection_results: Iterable[DetectionResult]
    ) -> Iterable[ProbabilisticTargetDescriptor]:
        """
        Crop the bounding boxes from the image and classify them using the general classifier
        """
        for detection_batch in batched(detection_results, self.classifier_batch_size):
            detection_batch: tuple[DetectionResult]
            cropped_images = (
                img.make_sub_image(det.x, det.y, det.width, det.height) for det in detection_batch
            )
            yield from self.general_classifier.predict(cropped_images)

    def _create_full_bbox_results(
        self,
        detection_results: Iterable[DetectionResult],
        classifier_outputs: Iterable[ProbabilisticTargetDescriptor]
    ):
        """
        Zips/combines the detection results and classifier outputs into FullBBoxPredictions
        """
        for detection, classification in zip(detection_results, classifier_outputs):
            detection: DetectionResult
            classification: ProbabilisticTargetDescriptor

            yield FullBBoxPrediction(
                detection.x,
                detection.y,
                detection.width,
                detection.height,
                classification,
                img_id=self.num_processed,
                det_id=detection.id
            )

            self.num_processed += 1

    def process_image(self, img: Image, tile_min_overlap=64, confidence_threshold=0.25):
        """
        Process an image through the detection-classification pipeline
        """
        detection_results = self._detect_targets(img, tile_min_overlap, confidence_threshold)
        classifier_outputs = self._classify_targets(img, detection_results)
        full_bbox_results = self._create_full_bbox_results(detection_results, classifier_outputs)
        return list(full_bbox_results)


class ImageProcessor:
    def __init__(self, debug_path: str = None, shape_batch_size = 3, letter_batch_size = 5):
        '''
        Initialize all models here 

        `shape_batch_size` is how many tiles we batch up for shape detection inference
        `letter_batch_size` is how many bounding box crops we batch up for letter classification
        '''
        self.tile_size = 640
        self.letter_size = 128
        self.shape_detector = ShapeDetector(self.tile_size)
        self.letter_classifier = LetterClassifier(self.letter_size)
        self.color_classifier = ColorClassifier()
        self.debug_path = debug_path
        self.thresh_iou = 0.5
        self.num_processed = 0
        self.shape_batch_size = shape_batch_size
        self.letter_batch_size = letter_batch_size

    def get_last_logs_path(self):
        return f"{self.debug_path}/img_{self.num_processed-1}"

    def _make_shape_detection(self, img : Image, tile_min_overlap = 64) -> list[DetectionResult]:
        shape_results: list[DetectionResult] = []

        all_tiles = img.generate_tiles(self.tile_size, tile_min_overlap)
        for tiles_batch in batched(all_tiles, self.shape_batch_size):
            temp = self.shape_detector.predict(tiles_batch)
            if temp is not None: shape_results.extend(temp)
        
        shape_results = non_max_suppress(shape_results, self.thresh_iou)

        if self.debug_path is not None:
            local_debug_path = f"{self.debug_path}/img_{self.num_processed}"
            os.makedirs(local_debug_path, exist_ok=True)
            img_to_draw_on = img.get_array().copy()
            for res in shape_results:
                x,y,w,h = res.x, res.y, res.width, res.height
                cv.rectangle(img_to_draw_on, (x,y), (x+w,y+h), (0,255,0), 2)
                cv.putText(img_to_draw_on, str(res.id), (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv.imwrite(f"{local_debug_path}/bounding_boxes.png", img_to_draw_on)

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

                if self.debug_path is not None:
                    local_debug_path = f"{self.debug_path}/img_{self.num_processed}"
                    instance_debug_path = f"{local_debug_path}/det_{shape_res.id}"
                    os.makedirs(instance_debug_path, exist_ok=True)
                    cv.imwrite(f"{instance_debug_path}/input.png", shape_res.img.get_array())
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
    
    def process_image(self, img: Image, tile_min_overlap = 64) -> list[FullBBoxPrediction]:
        '''
        img shape should be (height, width, channels)
        (that tuple order is a placeholder for now and we can change it later, but it should be consistent and we need to keep the docstring updated)
        '''
        if not isinstance(img, Image):
            raise TypeError("img must be an Image object")
        if not img.dim_order == HWC:
            raise ValueError("img must be in HWC order")
        
        shape_results = self._make_shape_detection(img, tile_min_overlap)
        self.num_processed += 1
        total_results = self._classify_color_and_char(shape_results)
        return total_results
    
    def process_image_lightweight(self, img : Image, tile_min_overlap = 64) -> list[FullBBoxPrediction]:
        '''
        Processes image and runs shape detection
        Only classifies if there is more than one detection.
        Use case: Zooming in and localizing onto a specific target.
        '''
        if not isinstance(img, Image):
            raise TypeError("img must be an Image object")
        if not img.dim_order == HWC:
            raise ValueError("img must be in HWC order")
        
        shape_results = self._make_shape_detection(img, tile_min_overlap)
        self.num_processed += 1

        if len(shape_results) == 1:
            # Returns shape_results cast as FullBBoxPrediction with no probabilistic target descriptor
            return [FullBBoxPrediction(
                    shape_results.x,
                    shape_results.y,
                    shape_results.width,
                    shape_results.height,
                    ProbabilisticTargetDescriptor(),
                    img_id = self.num_processed-1, # at this point it will have been incremented already
                    det_id = shape_results.id
                )]
        total_results = self._classify_color_and_char(shape_results)
        return total_results
