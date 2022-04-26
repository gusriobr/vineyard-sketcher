import unittest

import numpy as np

from sketcher.image.patches import sliding_window


class TestPatches(unittest.TestCase):

    def test_one_slide(self):
        """
        Image and patch same size, one patch as result
        :return:
        """
        image_size = (10, 10, 3)
        image = np.zeros(image_size)
        # slide over the image and the x,y coordinates returned
        coords = [(image, x, y) for image, x, y in sliding_window(image, image_size)]

        self.assertEqual(1, len(coords))
        patch, x, y = coords[0]
        self.assertEqual(image_size, patch.shape)
        self.assertEqual(0, x)
        self.assertEqual(0, y)

    def test_slide_without_rest(self):
        """
        Image and patch same size, one patch as result
        :return:
        """
        image_size = (5, 5, 3)
        window_size = (2, 3)
        step_size = 2
        image = np.zeros(image_size)
        # slide over the image and the x,y coordinates returned
        coords = [(image, x, y) for image, x, y in sliding_window(image, window_size, step_size, treat_borders=False)]

        self.assertEqual(2, len(coords))
        patch, x, y = coords[0]
        self.assertEqual(window_size, patch.shape[:2])

    def test_slide_with_rest(self):
        """
        Image and patch same size, one patch as result
        :return:
        """
        image_size = (5, 5, 3)
        window_size = (2, 3)
        step_size = 2
        image = np.zeros(image_size)
        # slide over the image and the x,y coordinates returned
        patches = list(sliding_window(image, window_size, step_size, treat_borders=True))

        self.assertEqual(6, len(patches))
        coords = [(x, y) for _, x, y in patches]
        expected_coords = [(0, 0), (2, 0), (0, 2), (2, 2), (0, 3), (2, 3)]
        self.assertTrue(all(item in coords for item in expected_coords))


if __name__ == "__main__":
    unittest.main()
