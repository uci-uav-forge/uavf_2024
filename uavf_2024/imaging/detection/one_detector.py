import os
from pathlib import Path
from .yolo_detector import YOLODetector


_CURRENT_FILE_DIR = Path(os.path.realpath(__file__)).parent


class OneDetector(YOLODetector):
    """
    Single-class target detection; first shot in two-shot pipeline.
    """

    # Placeholder
    # TODO: Update with actual confusion matrix. These are copied values.
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
        model_path: str = "weights/v8n-isaac-target-only.pt",
        confusion_matrix: dict[str, list[float]] | None = None,
    ):
        super().__init__(
            img_size,
            _CURRENT_FILE_DIR / model_path,
            confusion_matrix if confusion_matrix is not None else OneDetector.CONFUSION_MATRIX
        )
