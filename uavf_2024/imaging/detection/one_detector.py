import os
from pathlib import Path
from .yolo_detector import YOLODetector


_CURRENT_FILE_DIR = Path(os.path.realpath(__file__)).parent


class OneDetector(YOLODetector):
    """
    Single-class target detection; first shot in two-shot pipeline.
    """

    # Placeholder
    # TODO: Update with actual confusion matrix
    CONFUSION_MATRIX: dict[str, list[float]] = {
        'target': [0.9, 0.1]
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
