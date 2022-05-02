import unittest

import numpy as np

from sketcher.image.patches import sliding_window, batched_sliding_window


class TestSlidingWindow(unittest.TestCase):

    def test_one_slide(self):
        """
        Image and patch same size, one patch as result
        :return:
        """
        image_size = (10, 10, 3)
        image = np.zeros(image_size)
        expected_patches = 1

        # slide over the image and the x,y coordinates returned
        coords = [(image, x, y) for image, x, y in sliding_window(image, image_size)]

        self.assertEqual(expected_patches, len(coords))
        patch, x, y = coords[0]
        self.assertEqual(image_size, patch.shape)
        self.assertEqual(0, x)
        self.assertEqual(0, y)

    def test_slide_without_borders(self):
        """
        Image and patch same size, one patch as result
        :return:
        """
        image_size = (5, 5, 3)
        window_size = (2, 3)
        step_size = 2
        expected_patches = 4
        image = np.zeros(image_size)
        # slide over the image and the x,y coordinates returned
        coords = [(image, x, y) for image, x, y in sliding_window(image, window_size, step_size, treat_borders=False)]

        self.assertEqual(expected_patches, len(coords))
        patch, x, y = coords[0]
        self.assertEqual(window_size, patch.shape[:2])

    def test_slide_with_borders(self):
        """
        Image and patch same size, one patch as result
        :return:
        """
        image_size = (5, 5, 3)
        window_sizes = [(2, 3), (3, 3), (3, 2), (4, 4), (5, 5)]
        step_size = 2
        exp_patches = [6, 4, 6, 4, 1]
        for window_size, expected_patches in zip(window_sizes, exp_patches):
            image = np.zeros(image_size)
            # slide over the image and the x,y coordinates returned
            patches = list(sliding_window(image, window_size, step_size, treat_borders=True))

            self.assertEqual(expected_patches, len(patches))


class TestBatchSlidingWindow(unittest.TestCase):

    def test_batched_slides(self):
        """
        Image and patch same size, one patch as result
        :return:
        """
        image_size = (5, 5, 3)
        window_size = (2, 3)
        step_size = 2
        image = np.zeros(image_size)
        expected_patches = 6

        bsizes = [1, 3, 5, 6, 7, 20]

        # slide over the image and the x,y coordinates returned
        for batch_size in bsizes:
            batches = [(image, positions) for image, positions in
                       batched_sliding_window(image, window_size, step_size, batch_size=batch_size,
                                              treat_borders=True)]
            total = 0
            for batch in batches:
                self.assertLessEqual(len(batch[0]), batch_size)
                images, positions = batch
                self.assertEqual(images.shape[0], len(positions))
                self.assertEqual(window_size, images[0].shape[0:2])
                total += len(batch[0])
            self.assertEqual(expected_patches, total,
                             "Invalid number of items, expected {} got {}".format(expected_patches, total))


if __name__ == "__main__":
    unittest.main()
