from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
from dataclasses import dataclass
from ..imaging_types import InstanceSegmentationResult
import os
import torch

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class ShapeInstanceSegmentor:
    def __init__(self, img_size):
        self.shape_model = YOLO(f"{CURRENT_FILE_PATH}/weights/seg-v8n.pt")
        rand_input = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
        self.shape_model.predict(list(rand_input), verbose=False)


    def predict(self, img: np.ndarray) -> list[InstanceSegmentationResult]:
        '''
        Currently assumes batch size is 1
        TODO: refactor for batch processing
        '''
        raw_output: list[Results] = self.shape_model.predict(img)
        rawer_output = self.shape_model.model(torch.Tensor(img).permute(2,0,1).unsqueeze(0))
        single_pred = raw_output[0]
        masks = single_pred.masks
        if masks is None:
            return []
        boxes = single_pred.boxes
        full_results = []
        for box, mask, prob, cls in zip(boxes.xywh, masks.data, boxes.conf, boxes.cls):
            x,y,w,h = box.int()
            confidences = np.zeros(13) # TODO: change this to 8 for new model
            confidences[cls.int()] = prob
            full_results.append(
                InstanceSegmentationResult(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidences = confidences,
                    mask = mask[x:x+w, y:y+h].unsqueeze(2).numpy(),
                    img = img[x:x+w, y:y+h]
                )
            )
        return full_results