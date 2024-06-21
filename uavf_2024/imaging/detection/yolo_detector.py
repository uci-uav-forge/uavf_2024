from __future__ import annotations
from pathlib import Path
import warnings
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import numpy as np
from ..imaging_types import Tile, DetectionResult, img_coord_t, SHAPES
import os
from .. import profiler


class YOLODetector:
    """
    Wrapper class for YOLO-based detection.
    """
    def __init__(self, img_size: int, model_path: str | Path, confusion_matrix: dict[str, list[float]]):
        self.yolo = YOLO(Path(model_path))
        rand_input = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
        self.yolo.predict(list(rand_input), verbose=False)
        self.num_processed = 0
        self.confusion_matrix = confusion_matrix

    @profiler
    def predict(self, tiles: tuple[Tile], confidence_threshold=0.25) -> list[DetectionResult]:
        imgs_list = [tile.img.get_array() for tile in tiles if tile is not None]
        predictions: list[Results] = self.yolo(imgs_list, verbose=False, conf=confidence_threshold)

        full_results = []
        for img_index, single_pred in enumerate(predictions):
            if not isinstance(single_pred.boxes, Boxes):
                warnings.warn(f"{self.__class__.__name__}.predict() could not extract Boxes from YOLO output")
                continue

            boxes: Boxes = single_pred.boxes
            for box, prob, cls in zip(boxes.xywh, boxes.conf, boxes.cls):
                x, y, w, h = box.int()
                x -= int(w / 2)  # adjust to make x,y the top left
                y -= int(h / 2)
                confidences = np.zeros(9)
                confidences[cls.int()] = prob
                full_results.append(
                    DetectionResult(
                        x=img_coord_t(x.item()) + tiles[img_index].x,
                        y=img_coord_t(y.item()) + tiles[img_index].y,
                        width=img_coord_t(w.item()),
                        height=img_coord_t(h.item()),
                        confidences=np.array(self.confusion_matrix[SHAPES[cls.int()]]),
                        img=tiles[img_index].img.make_sub_image(x, y, w, h),
                        id=self.num_processed
                    )
                )
                self.num_processed += 1
        
        return full_results
