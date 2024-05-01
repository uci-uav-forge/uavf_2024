from typing import NamedTuple, Iterable

import torch
import torch.nn as nn

from ..imaging_types import SHAPES, COLORS, CHARACTERS, Image


class GeneralClassifierOutput(NamedTuple):
    shape_confs: torch.Tensor
    shape_color_confs: torch.Tensor
    character_confs: torch.Tensor
    character_color_confs: torch.Tensor

    @property
    def shape(self):
        return SHAPES[torch.argmax(self.shape_confs).item()]

    @property
    def shape_confidence(self):
        return torch.max(self.shape_confs).item()

    @property
    def shape_color(self):
        return COLORS[torch.argmax(self.shape_color_confs).item()]

    @property
    def shape_color_confidence(self):
        return torch.max(self.shape_color_confs).item()

    @property
    def character(self):
        return CHARACTERS[torch.argmax(self.character_confs).item()]

    @property
    def character_confidence(self):
        return torch.max(self.character_confs).item()

    @property
    def character_color(self):
        return COLORS[torch.argmax(self.character_color_confs).item()]

    @property
    def character_color_confidence(self):
        return torch.max(self.character_color_confs).item()


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

    def predict(self, images_batch: Iterable[Image]) -> GeneralClassifierOutput:
        """
        Passes the input through the model and transforms the output into a GeneralClassifierOutput.

        TODO: Implement this.
        """
        gpu_batch = self.create_gpu_tensor_batch(images_batch)
        raw: torch.Tensor = self.forward(gpu_batch)

        return GeneralClassifierOutput(
            torch.tensor([0.0 for _ in range(len(SHAPES))]),
            torch.tensor([0.0 for _ in range(len(COLORS))]),
            torch.tensor([0.0 for _ in range(len(CHARACTERS))]),
            torch.tensor([0.0 for _ in range(len(COLORS))])
        )

    def create_gpu_tensor_batch(self, images_batch: Iterable[Image]) -> torch.Tensor:
        return torch.stack(
            [torch.tensor(img.get_array()) for img in images_batch]
        ).to(device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement this
        """
        raise NotImplemented
