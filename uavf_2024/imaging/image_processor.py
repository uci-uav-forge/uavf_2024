import numpy as np
import cv2 as cv

from .imaging_types import HWC, FullPrediction, Image, InstanceSegmentationResult, TargetDescription
from .letter_classification import LetterClassifier
from .shape_detection import ShapeInstanceSegmenter
from .color_segmentation import color_segmentation
from .color_classification import ColorClassifier

def nms_process(shape_results: InstanceSegmentationResult, thresh_iou):
    boxes = np.array([[shape.x, shape.y, shape.x + shape.width, shape.y + shape.height, max(shape.confidences)] for shape in shape_results])
    order: list[int] = []
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
    def __init__(self):
        '''
        Initialize all models here 
        '''
        self.tile_size = 640
        self.letter_size = 128
        self.shape_detector = ShapeInstanceSegmenter(self.tile_size)
        self.letter_classifier = LetterClassifier(self.letter_size)
        self.color_classifier = ColorClassifier()
        self.thresh_iou = 0.5
    



    def process_image(self, img: Image, debug = True) -> "list[FullPrediction]":
        '''
        img shape should be (height, width, channels)
        (that tuple order is a placeholder for now and we can change it later, but it should be consistent and we need to keep the docstring updated)
        '''
        if not isinstance(img, Image):
            raise TypeError("img must be an Image object")
        
        if not img.dim_order == HWC:
            raise ValueError("img must be in HWC order")

        shape_results: list[InstanceSegmentationResult] = []
        img_2 = img.get_array().copy()
        for tile in img.generate_tiles(self.tile_size):
            
            # TODO re-implement batch processing
            # Draws rectangles representing the segmentation of the image
            cv.rectangle(img_2, (tile.x, tile.y), (tile.x+self.tile_size, tile.y+self.tile_size), (255,0,0), 1)
            shapes_detected = self.shape_detector.predict(tile.img)
            for shape in shapes_detected:
                shape.x+=tile.x
                shape.y+=tile.y
                shape_results.append(shape)
        
        shape_results = nms_process(shape_results, self.thresh_iou)
        if debug == True:
            for shape in shape_results:
                cv.rectangle(img_2, (shape.x, shape.y), (shape.x + shape.width, shape.y + shape.height), (0,0,255), 1)
                cv.putText(
                        img = img_2,
                        text = f"Confidence: {max(shape.confidences)}",
                        org = (shape.x + shape.width, shape.y + shape.height),
                        fontFace = cv.FONT_HERSHEY_DUPLEX,
                        fontScale = 3.0,
                        color = (125, 246, 55),
                        thickness = 3
                    )

        cv.imwrite("visualizations/tile_viz.png", img_2)

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