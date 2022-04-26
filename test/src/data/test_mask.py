import os
import unittest

import numpy as np

import cfg_test as tcfg
from skimage import io

from image.mask import MaskMerger, clean_instaces


class TestMaskMerger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.iterations = cls._load_images()

    @classmethod
    def _load_images(self):
        # iterations
        return [
            ['image_0_0_0.png'],
            ['image_0_100_0.png'],
            ['image_0_200_0.png'],
            ['image_0_300_0.png'],
            ['image_0_300_1.png', 'image_0_300_2.png'],
            ['image_0_400_0.png', 'image_0_400_1.png'],
            ['image_0_500_0.png', 'image_0_500_1.png'],
            ['image_0_600_0.png', 'image_0_600_1.png'],
            ['image_0_700_0.png'],
            ['image_0_800_0.png']
        ]

    def test_apply(self):
        base_folder = tcfg.resource("masks")
        out_img = np.zeros((512, 512 * 9), dtype=np.uint8)
        r = 0
        merger = MaskMerger()
        for iter in self.iterations:
            # load files
            masks = np.zeros((512, 512, len(iter)), dtype=np.uint8)
            for i in range(0, len(iter)):
                masks[:, :, i] = io.imread(os.path.join(base_folder, iter[i])).astype(np.uint8)
            pos = list(map(int, iter[0].split("_")[1:3]))
            pos.reverse()
            merger.apply(out_img, masks, pos)

            ids, counts = np.unique(out_img, return_counts=True)
            factor = 255 // len(ids)
            io.imsave("/tmp/salida_{}.png".format(r), out_img * factor)
            r += 1
        # comprobamos el número de instancias
        clean_instaces(out_img)

        ids, counts = np.unique(out_img, return_counts=True)
        factor = 255 // len(ids)
        io.imsave("/tmp/output_image.png".format(r), out_img * factor)
        self.assertEqual(3, len(ids))  # 0,1,2




if __name__ == "__main__":
    unittest.main()
