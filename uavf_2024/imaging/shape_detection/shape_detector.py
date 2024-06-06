from __future__ import annotations
import warnings
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import numpy as np
import torch
from ..imaging_types import Tile, DetectionResult, img_coord_t, SHAPES
import os
from .. import profiler

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

classes_2023_remapping = {
  0: 0, # circle
  1: 7, # cross
  2: 0, # heptagon -> circle
  3: 5, # hexagon -> pentagon
  4: 0, # octagon -> circle
  5: 5, # pentagon
  6: 2, # quartercircle
  7: 4, # rectangle
  8: 1, # semicircle
  9: 4, # square -> rectangle
  10: 8, # star
  11: 4, # trapezoid -> rectangle
  12: 3, # triangle
  13: 9, # person
}

class ShapeDetector:
    def __init__(self, img_size: int, conf=0.25):
        self.shape_model = YOLO(f"{CURRENT_FILE_PATH}/weights/seg-v8n-2023.pt")
        rand_input = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
        self.shape_model.predict(list(rand_input), verbose=False)
        self.num_processed = 0
        self.conf = conf
        self.cnf_matrix = {'circle' : [0.83, 0, 0, 0, 0, .01, 0, 0, 0],
                            'semicircle': [.01, .67, .28, .02, .05, .03, 0, 0, .01],
                            'quartercircle': [0, .18, .43, 0, .41, .17, 0, 0, 0],
                            'triangle': [0, .03, 0, .91, .01, 0, 0, 0, 0],
                            'rectangle': [.01, 0, .19, 0, .46, .08, 0, 0, 0],
                            'pentagon': [.10, .03, .08, 0, .01, .68, 0, 0, 0],
                            'star': [0, .01, 0, .04, 0, 0, .97, .02, 0],
                            'cross': [0, .04, 0, .01, 0, 0, 0, .96, .03],
                            'person': [0, .01, 0, .01, .01, 0, 0, 0, .91]
                                }


    @profiler
    def predict(self, tiles: tuple[Tile]) -> list[DetectionResult]:
        imgs_list = [tile.img.get_array() for tile in tiles if tile is not None]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictions: list[Results] = self.shape_model.predict(imgs_list, verbose=False, conf=self.conf, device=device)

        full_results = []
        for img_index, single_pred in enumerate(predictions):
            if not isinstance(single_pred.boxes, Boxes):
                warnings.warn("ShapeInstanceSegmenter.predict() could not extract Boxes from YOLO output")
                continue
        
            boxes: Boxes = single_pred.boxes
            for box, prob, cls in zip(boxes.xywh, boxes.conf, boxes.cls):
                x,y,w,h = box.int()
                x-=int(w/2) # adjust to make x,y the top left
                y-=int(h/2)
                confidences = np.zeros(9)
                new_cls = classes_2023_remapping[cls.int().item()]
                confidences[new_cls] = prob
                full_results.append(
                    DetectionResult(
                        x=img_coord_t(x.item())+tiles[img_index].x,
                        y=img_coord_t(y.item())+tiles[img_index].y,
                        width=img_coord_t(w.item()),
                        height=img_coord_t(h.item()),
                        confidences = np.array(self.cnf_matrix[SHAPES[new_cls]]),
                        img = tiles[img_index].img.make_sub_image(x, y, w, h),
                        id = self.num_processed
                    )
                )
                self.num_processed += 1
            return full_results
