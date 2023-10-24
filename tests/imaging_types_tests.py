from itertools import pairwise
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
            
    def test_simple_tiling(self):
        """
        Tests simple, no-overlap tiling.
        """
        
        test_dims = (4000, 4000, 3)
        img = Image(np.arange(test_dims[0] * test_dims[1] * test_dims[2], dtype=np.uint8).reshape(test_dims))
        tile_size = 500
        
        # Test that the generator yields the correct number of tiles of the correct shape
        tile_count = 0
        for tile in img.generate_tiles(tile_size, 0):
            # Check that the tile is the correct shape
            self.assertEqual(tile.img.shape, (tile_size, tile_size, 3))
            tile_count += 1

        self.assertEqual(tile_count, 4000 // 500 * 4000 // 500)
        
    def test_overlap_tiling(self):
        """
        Tests tiling with overlap.
        Checks that tiles are the correct shape and that they cover the entire image.
        Does NOT check that the overlap is correct.
        """
        for length in range(50, 55):
            test_dims = (length, length, 3)
            img_values = np.arange(test_dims[0] * test_dims[1] * test_dims[2])
            img = Image(img_values.reshape(test_dims))
            all_values = set(img_values)
            
            tile_size = 10
            tiles = [tile for tile in img.generate_tiles(tile_size, 1)]
            
            self.assertEqual(len(tiles), 36)
            
            for tile in tiles:
                # Check that the tile is the correct shape
                self.assertEqual(tile.img.shape, (tile_size, tile_size, 3))
                    
                # Remove from the set of values in the image the values in the tile
                all_values -= set(tile.img.get_array().flatten())
                
            # Check that all values in the image were covered by the tiles
            self.assertEqual(len(all_values), 0)

    def test_uniform_spacing(self):
        def test_with_params(width,height,tile_size,overlap):
            img = Image(np.zeros([width,height,3]))
            tile_size = 15
            overlap  = 5
            
            x_coords = set() 
            y_coords = set() 
            for tile in img.generate_tiles(tile_size, overlap):
                x_coords.add(tile.x)
                y_coords.add(tile.y)

            diffs_x = set()
            diffs_y = set()
            for x1, x2 in pairwise(sorted(x_coords)):
                diffs_x.add(x2-x1)

            for y1, y2 in pairwise(sorted(y_coords)):
                diffs_y.add(y2-y1)
            
            self.assertLessEqual(len(diffs_x), 2)
            self.assertLessEqual(len(diffs_y), 2)

            self.assertLessEqual(max(diffs_x)-min(diffs_x), 1)
            self.assertLessEqual(max(diffs_y)-min(diffs_y), 1)

        for i in range(64, 128, 7):
            for j in range(64, 128, 4):
                tile_size = np.random.randint(5,30)
                test_with_params(i,j,tile_size,np.random.randint(0,tile_size))

