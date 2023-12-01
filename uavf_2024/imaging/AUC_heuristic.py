import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from ultralytics.engine.results import Results
import torch


import scikitplot as skplt
import matplotlib.pyplot as plt


def roc_auc(pred, truth):
    # pred: array of class probability dist for each image
    # truth: array of true labels for each image
    # Ex: pred = [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3]]
    #     truth = [2, 0, 1]
    # Required: Truth must have at least one instance of each class
    #   Ex: [0, 2, 3] is missing class 1
    # returns: np.array of AUC for each class, micro-average auc, macro-average auc
    

    # Need all classes to have at least one instance in validation dataset, so drop unused columns from pred and shift truth accordingly
    set_truth = set(truth)
    n = max(max(set_truth),pred.shape[1])
    drop_cols = [col for col in list(range(n)) if col not in set_truth]

    pred = np.delete(pred, drop_cols, axis=1) # drop columns from pred
    # adjust truth for dropped columns by shifting for each missing class
    adjust_map = dict() # {adjusted_truth : real truth class}
    for idx, col in enumerate(truth): 
        truth[idx] = col - sum(1 for drop_col in drop_cols if col > drop_col) 
        adjust_map[truth[idx]] = col

    # plot roc curve
    axes = skplt.metrics.plot_roc(truth, pred)
    legend = axes.get_legend().get_texts()              # list[Text('sample')]
    auc_strs = [clas.get_text() for clas in legend]     # Ex: 'ROC curve of class 2 (area = 0.97)', 'micro-average ROC curve (area = 0.49)'
    axes.get_legend().remove()

    # populate output array
    out = np.zeros(n+2)
    for text in auc_strs[:-2]:                          # populate auc for each class
        class_num = adjust_map[int(text.split(' ')[4])]
        auc = float(text.split(' ')[7][:-1])
        out[class_num] = auc
    # populate micro and macro averages
    out[-2] = float(auc_strs[-2].split(' ')[5][:-1])    # micro-average is at second to last index
    out[-1] = float(auc_strs[-1].split(' ')[5][:-1])    # macro-average is at last index

    return out, axes

def run_letter_classification():
    # from letter_tests.py
    # but adjusted to output prob distribution for each image

    imgs_path = CURRENT_FILE_PATH + "/imaging_data/letter_dataset/images"
    labels_path = CURRENT_FILE_PATH + "/imaging_data/letter_dataset/labels"

    y_probasA = []
    y_trueA = []
    for img_file_name in os.listdir(imgs_path):
        img = cv.imread(f"{imgs_path}/{img_file_name}")
        raw_output = letter_classifier.model.predict(img)
        pred = raw_output[0].probs.data.numpy()
        with open(f"{labels_path}/{img_file_name.split('.')[0]}.txt") as f:
            truth = int(f.read(2))
        y_trueA.append(truth)
        y_probasA.append(pred)
    return np.array(y_trueA), np.array(y_probasA)

# y_true, y_probas = run_letter_classification()
