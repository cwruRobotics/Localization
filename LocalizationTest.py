import unittest
import numpy as np
from Localization import OBLocalization


class OBTest(unittest.TestCase):
    def test_matrix(self):
        l = OBLocalization(0.5, 1.5, 0.2)
        l.corners = np.array([[3, -2, 0.5], [3, -3.5, 0.5], [3, -2, 0], [3, -3.5, 0]])
        np.testing.assert_array_equal(l.getRotationMatrix(), np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                                      "some rotation")
        l.corners = np.array([[1.5, 0, 0.5], [0, 0, 0.5], [1.5, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(l.getRotationMatrix(), np.eye(3), "No transform")
        l.corners = 0.95 * np.array([[1.5, 0, 0.5], [0, 0, 0.5], [1.5, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(l.getRotationMatrix(), np.eye(3),
                                      "observed distance between balls not equal to expected")

    def test_angle(self):
        l = OBLocalization(0.5, 1.5, 0.2)
        l.corners = np.array([[3, -2, 0.5], [3, -3.5, 0.5], [3, -2, 0], [3, -3.5, 0]])
        self.assertEqual(l.getTheta(), np.pi / 2, "Facing +x direction")
        l.corners *= -1
        l.corners[0][2] = 0.5
        l.corners[1][2] = 0.5
        self.assertEqual(l.getTheta(), -np.pi / 2, "Facing -x direction")
        l.corners = np.array([[1.5, 0, 0.5], [0, 0, 0.5], [1.5, 0, 0], [0, 0, 0]])
        self.assertEqual(l.getTheta(), 0, "Facing +y direction")
        l.corners = np.array([[0, 2, 0.5], [1.5, 2, 0.5], [0, 2, 0], [1.5, 2, 0]])
        self.assertEqual(l.getTheta(), np.pi, "Facing -y direction")

    def test_order_corners(self):
        l = OBLocalization(0.5, 1.5, 0.2)
        e = np.array([[3, -2, 0.5], [3, -3.5, 0.5], [3, -2, 0], [3, -3.5, 0]])
        for i in range(10):
            l.corners = e.copy()
            while (l.corners == e).all():
                np.random.shuffle(l.corners)
            l.orderCorners()
            np.testing.assert_array_equal(l.corners, np.array([[3, -2, 0.5], [3, -3.5, 0.5], [3, -2, 0], [3, -3.5, 0]]))


if __name__ == '__main__':
    unittest.main()
