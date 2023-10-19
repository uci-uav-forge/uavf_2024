import warnings
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import numpy as np
from ..imaging_types import Image, InstanceSegmentationResult, img_coord_t
import os

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class ShapeInstanceSegmenter:
    def __init__(self, img_size):
        self.shape_model = YOLO(f"{CURRENT_FILE_PATH}/weights/seg-v8n.pt")
        rand_input = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
        self.shape_model.predict(list(rand_input), verbose=False)


    def predict(self, img: Image) -> list[InstanceSegmentationResult]:
        '''
        Currently assumes batch size is 1
        TODO: refactor for batch processing
        '''
        raw_output: list[Results] = self.shape_model.predict(img)
        single_pred = raw_output[0]
        
        masks = single_pred.masks
        if masks is None:
            return []
        
        if not isinstance(single_pred.boxes, Boxes):
            warnings.warn("ShapeInstanceSegmenter.predict() could not extract Boxes from YOLO output")
            return []
        
        boxes: Boxes = single_pred.boxes
        full_results = []
        for box, mask, prob, cls in zip(boxes.xywh, masks.data, boxes.conf, boxes.cls):
            x,y,w,h = box.int()
            x-=int(w/2) # adjust to make x,y the top left
            y-=int(h/2)
            confidences = np.zeros(13) # TODO: change this to 8 for new model
            confidences[cls.int()] = prob
            full_results.append(
                InstanceSegmentationResult(
                    x=img_coord_t(x.item()),
                    y=img_coord_t(y.item()),
                    width=img_coord_t(w.item()),
                    height=img_coord_t(h.item()),
                    confidences = confidences,
                    mask = mask[x:x+w, y:y+h].unsqueeze(2).numpy(),
                    img = np.array(img[x:x+w, y:y+h])
                )
            )
        return full_results