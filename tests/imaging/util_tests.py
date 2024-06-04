import unittest
import matplotlib.pyplot as plt
from uavf_2024.imaging.utils import make_ortho_vectors
import torch
from torch.nn.functional import cosine_similarity

class TestUtilities(unittest.TestCase):
    def test_make_ortho_vectors(self, verbose=False):
        n = 100
        m = 10
        rand_vecs = torch.rand(n, 3)

        ortho_vecs = make_ortho_vectors(rand_vecs, m)

        # assert that each set of ortho vecs are all orthogonal to the original vector
        for i in range(n):
            for j in range(m):
                cos_sim = cosine_similarity(rand_vecs[i], ortho_vecs[i,j], dim=0)
                assert cos_sim < 0.1, f"Cosine: {cos_sim}"

        if verbose:
            print("TODO: 3d visualization not implemented yet :(")

if __name__ == "__main__":
    TestUtilities().test_make_ortho_vectors(True)