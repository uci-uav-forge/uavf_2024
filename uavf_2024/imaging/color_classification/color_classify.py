import numpy as np
import torch
from torchvision import transforms
import os
import torch.nn as nn
from uavf_2024.imaging.imaging_types import COLOR_INDICES

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class ColorModel(nn.Module):
    def __init__(self, num_classes):
        super(ColorModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.letter_dense = nn.Sequential(
            nn.Linear(128 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1),
        )

        self.shape_dense = nn.Sequential(
            nn.Linear(128 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)

        letter_output = self.letter_dense(x)
        shape_output = self.shape_dense(x)

        return letter_output, shape_output

class ColorClassifier:
    def __init__(self):
        model_path = CURRENT_FILE_PATH + "/trained_model.pth"
        num_classes = 8
        self.model = self.load_model(model_path, num_classes)
        self.transform = transforms.ToTensor()

    def load_model(self, model_path, num_classes):
        model = ColorModel(num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns probabilities for each color"""
        image_tensor = self.transform(image).unsqueeze(0)  # Add a batch dimension
        
        with torch.no_grad():
            predicted_letter, predicted_shape = self.model.forward(image_tensor)

        # _, predicted_shape = torch.max(predicted_shape, 1)
        # _, predicted_letter = torch.max(predicted_letter, 1)
        return predicted_letter.cpu()[0].numpy(), predicted_shape.cpu()[0].numpy()
