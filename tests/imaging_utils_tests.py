import unittest
import numpy as np
from itertools import pairwise

from uavf_2024.imaging.utils import generate_tiles

class ImagingUtilsTest(unittest.TestCase):
    def test_simple_tiling(self):
        """
        Tests simple, no-overlap tiling.
        """
        
        test_dims = (4000, 4000, 3)
        img = np.arange(test_dims[0] * test_dims[1] * test_dims[2], dtype=np.uint8).reshape(test_dims)
        tile_size = 500
        
        # Test that the generator yields the correct number of tiles of the correct shape
        tile_count = 0
        for tile in generate_tiles(img, tile_size, 0):
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
            img = img_values.reshape(test_dims)
            all_values = set(img_values)
            
            tile_size = 10
            tiles = [tile for tile in generate_tiles(img, tile_size, 1)]
            
            self.assertEqual(len(tiles), 36)
            
            for tile in tiles:
                # Check that the tile is the correct shape
                self.assertEqual(tile.img.shape, (tile_size, tile_size, 3))
                    
                # Remove from the set of values in the image the values in the tile
                all_values -= set(tile.img.flatten())
                
            # Check that all values in the image were covered by the tiles
            self.assertEqual(len(all_values), 0)

    def test_uniform_spacing(self):
        def test_with_params(width,height,tile_size,overlap):
            img = np.zeros([width,height,3])
            tile_size = 15
            overlap  = 5
            
            x_coords = set() 
            y_coords = set() 
            for tile in generate_tiles(img, tile_size, overlap):
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

