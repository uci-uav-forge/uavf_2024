import unittest
import numpy as np
import torch

from uavf_2024.imaging.imaging_types import Image, ImageDimensionsOrder, HWC, CHW, CHANNELS, HEIGHT, WIDTH


class ImageClassTest(unittest.TestCase):
    
    def test_initialization_np(self):
        np_numbers = np.arange(3 * 4 * 5)
        array_chw = np_numbers.reshape((3, 4, 5))
        array_hwc = np_numbers.reshape((5, 4, 3))
        
        valid_chw = Image(array_chw, CHW)
        valid_hwc = Image(array_hwc, HWC)
        
        self.assertTrue((valid_chw.get_array() == array_chw).all())
        self.assertTrue((valid_hwc.get_array() == array_hwc).all())
        
        self.assertEqual(valid_chw.dim_order, CHW)
        self.assertEqual(valid_hwc.dim_order, HWC)
        
        self.assertRaises(Exception, Image, array_hwc, CHW)
        self.assertRaises(Exception, Image, array_chw, HWC)
        
        too_many_axes = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        self.assertRaises(Exception, Image, too_many_axes, HWC)
        
        self.assertRaises(Exception, Image, array_hwc, ImageDimensionsOrder(WIDTH, HEIGHT, CHANNELS))

    def test_wrong_initialization(self):
        l = [[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ] * 4] * 4
        
        self.assertRaises(Exception, Image, l, HWC)
    
    def test_equality_torch(self):
        numbers = torch.arange(3 * 4 * 5)
        array_chw = numbers.reshape((3, 4, 5))
        array_hwc = numbers.reshape((5, 4, 3))
        
        image_1 = Image(array_chw, CHW)
        image_2 = Image(array_hwc, HWC)
        
        self.assertNotEqual(image_1, image_2)
        
        zeros = torch.zeros((3, 3, 3))
        
        hwc_cube = Image(zeros, HWC)
        chw_cube = Image(zeros, CHW)
        
        self.assertNotEqual(hwc_cube, chw_cube)
        self.assertNotEqual(hwc_cube, zeros)
        
        self.assertEqual(Image(torch.zeros((3, 4, 5)), CHW), Image(torch.zeros((3, 4, 5)), CHW))
        
    def test_get_set_item_torch(self):
        numbers = torch.arange(3 * 4 * 5)
        array_chw = numbers.reshape((3, 4, 5))
        
        image = Image(array_chw, CHW)
        
        image[0, 0, 0] = 100
        
        self.assertEqual(image[0, 0, 0], 100)
        
        image[0, 0, :] = torch.tensor([0, 1, 2, 3, 4])
        
        for i in range(5):
            self.assertEqual(image[0, 0, i], i)

