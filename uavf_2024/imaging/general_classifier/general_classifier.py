from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from ..imaging_types import SHAPES, COLORS, CHARACTERS, Image, ProbabilisticTargetDescriptor


class GeneralClassifier(nn.Module):
    """
    Custom model to classify color, shape, and character from cropped bbox.
    """
    def __init__(self, model_path: str | None = None, device=torch.device("cuda:0")):
        """
        TODO: Design and implement model architecture.
        """
        super().__init__()
        self.device = device

        # TODO: Implement model loading
        self.model_path = model_path

    def predict(self, images_batch: Iterable[Image]) -> list[ProbabilisticTargetDescriptor]:
        """
        Passes the input through the model and transforms the output into a ProbabilisticTargetDescriptor.

        TODO: Implement this.
        """
        gpu_batch = self.create_gpu_tensor_batch(images_batch)
        raw: torch.Tensor = self.forward(gpu_batch)

        # NOTE: Empty placeholder
        return [
            ProbabilisticTargetDescriptor(
                np.array([0.0] * len(SHAPES)),
                np.array([0.0] * len(COLORS)),
                np.array([0.0] * len(CHARACTERS)),
                np.array([0.0] * len(COLORS)),
            ) for _ in images_batch
        ]

    def create_gpu_tensor_batch(self, images_batch: Iterable[Image]) -> torch.Tensor:
        return torch.stack(
            [torch.tensor(img.get_array()) for img in images_batch]
        ).to(device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement this
        """
        raise NotImplemented
