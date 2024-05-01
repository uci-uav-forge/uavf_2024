from typing import NamedTuple

import torch
import torch.nn as nn

from ..imaging_types import SHAPES, COLORS, CHARACTERS


class GeneralClassifierOutput(NamedTuple):
    shape_probs: torch.Tensor
    shape_color_probs: torch.Tensor
    character_probs: torch.Tensor
    character_color_probs: torch.Tensor

    @property
    def shape(self):
        return SHAPES[torch.argmax(self.shape_probs).item()]

    @property
    def shape_color(self):
        return COLORS[torch.argmax(self.shape_color_probs).item()]

    @property
    def character(self):
        return CHARACTERS[torch.argmax(self.character_probs).item()]

    @property
    def character_color(self):
        return COLORS[torch.argmax(self.character_color_probs).item()]


class GeneralClassifier(nn.Module):
    """
    Custom model to classify color, shape, and character from cropped bbox.
    """
    def __init__(self, model_path: str | None = None):
        """
        TODO: Design and implement model architecture.
        """
        super().__init__()
        # TODO: Implement model loading
        self.model_path = model_path

    def forward(self, x: torch.Tensor) -> GeneralClassifierOutput:
        """
        Forward pass of the model.

        TODO: Implement this.
        """
        return GeneralClassifierOutput(
            shape_probs=torch.tensor([0.0 for _ in range(len(SHAPES))]),
            shape_color_probs=torch.tensor([0.0 for _ in range(len(COLORS))]),
            character_probs=torch.tensor([0.0 for _ in range(len(CHARACTERS))]),
            character_color_probs=torch.tensor([0.0 for _ in range(len(COLORS))])
        )

