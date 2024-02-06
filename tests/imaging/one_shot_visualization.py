import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from uavf_2024.imaging.one_shot.dataset import SUASDataset

from uavf_2024.imaging.one_shot.one_shot import SUASYOLO
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torchinfo import summary

# img_file = "data/images/tiny_train/image0.png"
# label_file = "data/labels/tiny_train/image0.txt"
# img = cv.imread(img_file)

# with open(label_file, "r") as f:
#     for line in f.readlines():
#         line = line.strip().split()
#         x, y, w, h = map(float, line[1:])
#         x1 = int((x - w / 2) * img.shape[1])
#         y1 = int((y - h / 2) * img.shape[0])
#         x2 = int((x + w / 2) * img.shape[1])
#         y2 = int((y + h / 2) * img.shape[0])
#         cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         # draw label
#         cv.putText(img, line[0], (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         # predict and draw
#         model = SUASYOLO(num_classes = 17, cell_resolution=10).to(DEVICE)
#         model.load_state_dict(torch.load("tiny_train.pt"))
#         model.eval()
#         boxes, classes = model.predict(torch.tensor(cv.imread(img_file)).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE))
#         for box, classes in zip(boxes, classes):
#             x1, y1, x2, y2 = (box*640).to("cpu").type(torch.int).tolist()
#             cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             # draw label
#             class_pred = classes.argmax()
#             cv.putText(img, str(class_pred), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

'''
Operates in-place on `img`
'''
def display_boxes(boxes: torch.Tensor, classes: torch.Tensor, objectness: torch.Tensor, color, thickness, img, centers_only=False):

    for box, classes, objectness_score in zip(boxes, classes, objectness):
        class_pred = classes.argmax().item()
        if centers_only:
            x, y = (box*640).to("cpu").type(torch.int).tolist()
            cv.circle(img, (x, y), 5, color, thickness)
            cv.putText(img, str(class_pred), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv.putText(img, f'{objectness_score.item():.2f}', (x+10, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            x1, y1, x2, y2 = (box*640).to("cpu").type(torch.int).tolist()
            if x1-x2 == 0 or y1-y2 == 0:
                continue
            cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            # draw label
            cv.putText(img, str(class_pred), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv.putText(img, f'{objectness_score.item():.2f}', (x1+10, y2+20), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def get_display_figures(model: SUASYOLO, dataset: SUASDataset, n=5, centers_only=False):
    figures = []
    for i in range(n):
        img, label = dataset[i]
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        boxes, objectness, _shape_colors, _letter_colors, classes, _letter_classes = model.process_predictions(label.unsqueeze(0))
        boxes = boxes[objectness > 0]
        classes = classes[objectness > 0]
        objectness = objectness[objectness > 0]
        display_boxes(boxes, classes, objectness, (0,255,0),3,img, centers_only=centers_only)
        pred_boxes, pred_conf, _pred_shape_color, _pred_letter_color, pred_classes, _pred_letter_classes, _raw_preds = model.predict(torch.tensor(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE))
        display_boxes(pred_boxes, pred_classes, pred_conf, (0,0,255),1,img, centers_only=centers_only)
        fig = plt.figure()
        plt.title(f"Boxes for image {i}")
        plt.imshow(img)
        plt.axis("off")
        figures.append(fig)
    return figures 

if __name__=='__main__':
    model = SUASYOLO(num_classes = 17).to(DEVICE)
    n_cells = model.num_cells
    dataset = SUASDataset("imaging_data/fullsize_dataset/images/5k.png", "imaging_data/fullsize_dataset/labels/image0.txt", 17, n_cells = n_cells)
    model.load_state_dict(torch.load("yolo.pt"))
    model.eval()
    figs = get_display_figures(model, dataset)
    for fig in figs:
        fig.show()
        plt.show()


