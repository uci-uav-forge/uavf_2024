from uavf_2024.gnc.util import calculate_turn_angles_deg
import numpy as np
import unittest

class TestCalculateTurnAngles(unittest.TestCase):
    def test_perpendicular_turn_angles(self):
        expected = [90.0, 90.0, 90.0]
        actual = calculate_turn_angles_deg([np.array([0, 10]), np.array([10, 10]), np.array([10, 0]), np.array([0, 0]), np.array([0, 10])])
        self.assertEqual(expected, actual)

    def test_obtuse_turn_angles(self):
        expected = [135.0, 135.0]
        actual = calculate_turn_angles_deg([np.array([0, 0]), np.array([1, -1]), np.array([-1, -1]), np.array([0, 0])])
        self.assertEqual(expected, actual)

    def test_acute_and_obtuse_turn_angles(self):
        expected = [45.0, 135.0, 45.0]
        actual = calculate_turn_angles_deg([np.array([0, 0]), np.array([-5, 0]), np.array([-6, 1]), np.array([-1, 1]), np.array([0, 0])])
        self.assertEqual(expected, actual)

    def test_zero_turn_angle(self):
        expected = [45.0, 90.0, 90.0, 0.0]
        actual = calculate_turn_angles_deg([np.array([0, 2]), np.array([2, 4]), np.array([2, 7]), np.array([0, 7]), np.array([0, 5]), np.array([0, 2])])
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()