from .imaging_types import FullPrediction
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def visualize_predictions(img: np.ndarray, predictions: list[FullPrediction], file_name):
    img = img.copy()
    for pred in predictions:
        x, y = pred.x, pred.y
        w, h = pred.width, pred.height
        cv.rectangle(img, [x,y], [x+w,y+h], (0,0,255),1)
    cv.imwrite(f"/home/ws/uavf_2024/visualizations/{file_name}.png", img)