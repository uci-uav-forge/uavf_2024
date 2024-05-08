import os
import cv2 as cv
import numpy as np

from uavf_2024.imaging import ProbabilisticTargetDescriptor
from uavf_2024.imaging.color_classification import ColorClassifier
from uavf_2024.imaging.detection import ShapeDetector
from uavf_2024.imaging.image_processor import non_max_suppress
from uavf_2024.imaging.imaging_types import FullBBoxPrediction, HWC, Image, DetectionResult
from uavf_2024.imaging.letter_classification import LetterClassifier
from uavf_2024.imaging.utils import batched


class OldImageProcessor:
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