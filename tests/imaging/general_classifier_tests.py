from typing import Sequence, get_type_hints
from unittest import TestCase

import torch

from uavf_2024.imaging.general_classifier import GeneralClassifier, ResNet, resnet18, resnet34, resnet50


class ResNetTests(TestCase):
    NUM_CLASSES = [2, 3, 4]
    INPUT_SHAPE = (3, 3, 224, 224)
    
    def _resnet_sanity_check(self, model: ResNet):
        data = torch.randn(ResNetTests.INPUT_SHAPE)
        
        output: list[torch.Tensor] = model(data)
        
        for i, tensor in enumerate(output):
            self.assertEqual(tensor.shape, (ResNetTests.INPUT_SHAPE[0], 1, 1, ResNetTests.NUM_CLASSES[i]))
    
    def test_resnet18(self):
        model = resnet18(ResNetTests.NUM_CLASSES)
        
        self._resnet_sanity_check(model)
        
    def test_resnet34(self):
        model = resnet34(ResNetTests.NUM_CLASSES)
        
        self._resnet_sanity_check(model)
        
    def test_resnet50(self):
        model = resnet50(ResNetTests.NUM_CLASSES)
        
        self._resnet_sanity_check(model)
        