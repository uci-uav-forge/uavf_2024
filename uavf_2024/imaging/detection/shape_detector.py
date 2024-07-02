from __future__ import annotations
import os

from .yolo_detector import YOLODetector

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class ShapeDetector(YOLODetector):
    CONFUSION_MATRIX: dict[str, list[float]] = {
        'circle': [0.83, 0, 0, 0, 0, .01, 0, 0, 0],
        'semicircle': [.01, .67, .28, .02, .05, .03, 0, 0, .01],
        'quartercircle': [0, .18, .43, 0, .41, .17, 0, 0, 0],
        'triangle': [0, .03, 0, .91, .01, 0, 0, 0, 0],
        'rectangle': [.01, 0, .19, 0, .46, .08, 0, 0, 0],
        'pentagon': [.10, .03, .08, 0, .01, .68, 0, 0, 0],
        'star': [0, .01, 0, .04, 0, 0, .97, .02, 0],
        'cross': [0, .04, 0, .01, 0, 0, 0, .96, .03],
        'person': [0, .01, 0, .01, .01, 0, 0, 0, .91]
    }

    def __init__(
        self,
        img_size: int = 640,
        model_path: str = f"{CURRENT_FILE_PATH}/weights/v8n-best.pt",
        confusion_matrix: dict[str, list[float]] | None = None,
    ):
        super().__init__(
            img_size,
            model_path,
            confusion_matrix if confusion_matrix is not None else ShapeDetector.CONFUSION_MATRIX
        )
