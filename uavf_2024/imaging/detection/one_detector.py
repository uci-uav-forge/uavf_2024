from .yolo_detector import YOLODetector


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
        model_path: str = "weights/isaac_target_only.pt",
        confusion_matrix: dict[str, list[float]] | None = None,
    ):
        super().__init__(
            img_size,
            model_path,
            confusion_matrix if confusion_matrix is not None else OneDetector.CONFUSION_MATRIX
        )
