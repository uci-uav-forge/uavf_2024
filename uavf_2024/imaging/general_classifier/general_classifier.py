import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from uavf_2024.imaging.imaging_types import SHAPES, COLORS, CHARACTERS, Image, ProbabilisticTargetDescriptor, CHW
from . import ResNet, resnet18


_CURRENT_FILE_PATH = Path(os.path.realpath(__file__)).parent


class GeneralClassifier:
    """
    Custom model to classify color, shape, and character from cropped bbox.
    """
    def __init__(
        self, 
        model_relative_path: str = "best_69.pt", 
        model_factory: Callable[[Sequence[int]], ResNet] = resnet18, 
        device=torch.device("cuda:0")
    ):
        super().__init__()
        self.device = device
        
        self.model_path = _CURRENT_FILE_PATH / model_relative_path

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not (self.model_path.suffix == ".pth" or self.model_path.suffix == ".pt"):
            raise ValueError(f"Model file must be a PyTorch model file: {self.model_path}")
        
        # Load the model
        self.model_path = self.model_path
        self.model = model_factory([len(SHAPES), len(COLORS), len(CHARACTERS), len(COLORS)])
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(device=self.device)

    @staticmethod
    def _format_image(image: Image) -> Image:
        """
        Formats the image to be passed to the model.
        
        Pad and resize the image to 224x224, change the dimension order to CHW, and normalize to [0, 1] float32.
        """
        arr = image.make_square(224).get_array().astype(np.float32) / 255.0
        square = Image(arr, image.dim_order)
        square.change_dim_order(CHW)
        return square
    
    def predict(self, images_batch: Iterable[Image]) -> list[ProbabilisticTargetDescriptor]:
        """
        Passes the input through the model and transforms the output into a ProbabilisticTargetDescriptor.
        """
        square_crops_chw = map(__class__._format_image, images_batch)
        gpu_batch = self.create_gpu_tensor_batch(square_crops_chw)
        
        # List of batches, one for each of the heads
        # I.e., the shape is (category, batch_size, num_classes)
        raw: list[torch.Tensor] = self.model(gpu_batch)
        shape_dists, shape_color_dists, character_dists, character_color_dists = raw

        return [
            ProbabilisticTargetDescriptor(
                *map(lambda t: t.cpu().numpy(), tensors)
            ) for tensors
            in zip(shape_dists, shape_color_dists, character_dists, character_color_dists)
        ]
    
    def create_gpu_tensor_batch(self, images_batch: Iterable[Image]) -> torch.Tensor:
        return torch.stack(
            [torch.tensor(img.get_array()) for img in images_batch]
        ).to(device=self.device)
