import numpy as np
from dataclasses import dataclass
import math
from .imaging_types import FullPrediction,InstanceSegmentationResult,TargetDescription
from .letter_classification import LetterClassifier
from .shape_detection import ShapeInstanceSegmenter
from .color_segmentation import color_segmentation
from .color_classification import ColorClassifier
import itertools

@dataclass
class Tile:
    img: np.ndarray
    x: int
    y: int

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

    def _split_to_tiles(self, img: np.ndarray) -> list[Tile]:
        h, w = img.shape[:2]
        n_horizontal_tiles = math.ceil(w / self.tile_size)
        n_vertical_tiles = math.ceil(h / self.tile_size)
        all_tiles: list[Tile] = []
        v_indices = np.linspace(0, h - self.tile_size, n_vertical_tiles).astype(int)
        h_indices = np.linspace(0, w - self.tile_size, n_horizontal_tiles).astype(int)

        for v, h in itertools.product(v_indices, h_indices):
            tile = img[v:v + self.tile_size, h:h + self.tile_size]
            all_tiles.append(Tile(tile, h, v))

        return all_tiles

    def process_image(self, img: np.ndarray) -> list[FullPrediction]:
        '''
        img shape should be (channels, width, height)
        (that tuple order is a placeholder for now and we can change it later, but it should be consistent and we need to keep the docstring updated)

        '''

        tiles = self._split_to_tiles(img)

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