import unittest
import numpy as np
import math
from uavf_2024.gnc.conversions import convert_quaternion_to_euler_angles, convert_NED_ENU_in_inertial, convert_NED_ENU_in_body

class TestConversionFunctions(unittest.TestCase):
    def setUp(self):
        self.quat_1, self.euler_1 = np.array([1, 0, 0, 0]), np.array([0, 0, 0])                                        # identity quaternion
        self.quat_2, self.euler_2 = np.array([0, 1, 0, 0]), np.array([math.pi, 0, 0])                                  # rotate 180 degrees around x-axis
        self.quat_3, self.euler_3 = np.array([0, 0, 1, 0]), np.array([math.pi, 0, math.pi])                            # rotate 180 degrees around y-axis
        self.quat_4, self.euler_4 = np.array([0, 0, 0, 1]), np.array([0, 0, math.pi])                                  # rotate 180 degrees around z-axis
        self.quat_5, self.euler_5 = np.array([math.sqrt(0.5), math.sqrt(0.5), 0, 0]), np.array([math.pi / 2, 0, 0])    # rotate 90 degrees around x-axis
        self.quat_6, self.euler_6 = np.array([math.sqrt(0.5), 0, math.sqrt(0.5), 0]), np.array([0, math.pi / 2, 0])    # rotate 90 degrees around y-axis
        self.quat_7, self.euler_7 = np.array([math.sqrt(0.5), 0, 0, math.sqrt(0.5)]), np.array([0, 0, math.pi / 2])    # rotate 90 degrees around z-axis
    
        self.NED_1 = np.array([10, 20, -30])
        self.NED_2 = np.array([-20, 50, 35])
        self.NED_3 = np.array([-30, 40, 0])

        self.ENU_1_inertial = np.array([20, 10, 30])
        self.ENU_2_inertial = np.array([50, -20, -35])
        self.ENU_3_inertial = np.array([40, -30, 0])

        self.ENU_1_body = np.array([10, -20, 30])
        self.ENU_2_body =  np.array([-20, -50, -35])
        self.ENU_3_body =np.array([-30, -40, 0])

    def test_convert_quaternion_to_euler_angles(self):
        self.assertTrue(np.array_equal(convert_quaternion_to_euler_angles(self.quat_1), self.euler_1))
        self.assertTrue(np.allclose(convert_quaternion_to_euler_angles(self.quat_2), self.euler_2, 1e-7))
        self.assertTrue(np.allclose(convert_quaternion_to_euler_angles(self.quat_3), self.euler_3, 1e-7))
        self.assertTrue(np.allclose(convert_quaternion_to_euler_angles(self.quat_4), self.euler_4, 1e-7))
        self.assertTrue(np.allclose(convert_quaternion_to_euler_angles(self.quat_5), self.euler_5, 1e-7))
        self.assertTrue(np.allclose(convert_quaternion_to_euler_angles(self.quat_6), self.euler_6, 1e-7))
        self.assertTrue(np.allclose(convert_quaternion_to_euler_angles(self.quat_7), self.euler_7, 1e-7))

    def test_convert_NED_ENU_in_inertial(self):
        self.assertTrue(np.array_equal(convert_NED_ENU_in_inertial(self.NED_1), self.ENU_1_inertial))
        self.assertTrue(np.array_equal(convert_NED_ENU_in_inertial(self.ENU_1_inertial), self.NED_1))
        self.assertTrue(np.array_equal(convert_NED_ENU_in_inertial(self.NED_2), self.ENU_2_inertial))
        self.assertTrue(np.array_equal(convert_NED_ENU_in_inertial(self.ENU_2_inertial), self.NED_2))
        self.assertTrue(np.array_equal(convert_NED_ENU_in_inertial(self.NED_3), self.ENU_3_inertial))
        self.assertTrue(np.array_equal(convert_NED_ENU_in_inertial(self.ENU_3_inertial), self.NED_3))

    def test_convert_NED_ENU_in_body(self):
        self.assertTrue(np.array_equal(convert_NED_ENU_in_body(self.NED_1), self.ENU_1_body))
        self.assertTrue(np.array_equal(convert_NED_ENU_in_body(self.ENU_1_body), self.NED_1))
        self.assertTrue(np.array_equal(convert_NED_ENU_in_body(self.NED_2), self.ENU_2_body))
        self.assertTrue(np.array_equal(convert_NED_ENU_in_body(self.ENU_2_body), self.NED_2))
        self.assertTrue(np.array_equal(convert_NED_ENU_in_body(self.NED_3), self.ENU_3_body))
        self.assertTrue(np.array_equal(convert_NED_ENU_in_body(self.ENU_3_body), self.NED_3))


if __name__ == '__main__':
    unittest.main()
