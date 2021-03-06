import logging
import os
import sys

import numpy as np
import skimage
import tensorflow as tf
from skimage import io
import timeit

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # disable gpu
    # print("Disabling GPU -> Using CPU for inference")
    # tf.config.set_visible_devices([], 'GPU')

ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import model as modellib
from sketcher import cfg
from sketcher.train.train import VineyardConfig
from sketcher.image.raster import georeference_image
from sketcher.image.mask import MaskMerger
from sketcher.image.patches import batched_sliding_window

cfg.configLog()

debug = False


def comp_file(filename):
    return os.path.join(cfg.root_folder(), filename)


def get_file(filepath):
    return cfg.file_path(filepath)
    # basepath = os.path.dirname(os.path.abspath(__file__))
    # return os.path.join(basepath, filepath)


def get_folder(filepath):
    f_path = cfg.file_path(filepath)
    if not os.path.exists(f_path):
        os.makedirs(f_path)
    return f_path


def predict_batch(model, images, positions, output_mask, mask_merger):
    # set the batchsize
    dummy_predictions = 0
    if len(images) < model.config.BATCH_SIZE:  # if image array is shorter than expected batch_size, fulfill with zeros
        dummy_predictions = model.config.BATCH_SIZE - len(images)
        zeros = np.zeros((dummy_predictions,) + images.shape[-3:])
        images = np.vstack([images, zeros])
    # batch size < model batch, fullfil with zeros a

    detection = model.detect(images, verbose=0)
    for i in range(0, len(images) - dummy_predictions):
        masks = detection[i]['masks']
        x, y = positions[i]
        if masks.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            # mask = (np.sum(masks, -1, keepdims=True) >= 1)
            mask_merger.apply(output_mask, masks, (x, y))
            if debug:
                for i in range(0, masks.shape[-1]):
                    skimage.io.imsave("/tmp/salidas/{}_{}_{}_mask.png".format(y, x, i), masks[:, :, i])


#
# def predict(model, image, output_mask, x, y, mask_merger):
#     # Detect objects
#     r = model.detect([image], verbose=1)[0]
#     # Color splash
#     masks = r['masks']
#     if masks.shape[-1] > 0:
#         # We're treating all instances as one, so collapse the mask into one layer
#         # mask = (np.sum(masks, -1, keepdims=True) >= 1)
#         mask_merger.apply(output_mask, masks, (x, y))
#         for i in range(0, masks.shape[-1]):
#             print("{}_{}".format(y, x))
#             if debug:
#                 skimage.io.imsave("/tmp/salidas/{}_{}_{}_mask.png".format(y, x, i), masks[:, :, i])


def apply_model(image_path, output_path, model, step_size, window_size, batch_size=1):
    # load the input image
    image = read_img(image_path)

    # grab the dimensions of the input image and crop the image such
    # that it tiles nicely when we generate the training data +
    # labels
    (h, w) = image.shape[:2]

    output_img = np.zeros((h, w, 1), dtype=np.uint16)
    mask_merger = MaskMerger()

    for images, positions in batched_sliding_window(image, window_size, step_size, batch_size=batch_size):
        start = timeit.default_timer()
        predict_batch(model, images, positions, output_img, mask_merger)
        execution_time = timeit.default_timer() - start
        print("Batch {} images exec time: {}".format(len(images), (execution_time)))
        # y_pred = predict(model, img, x, y, mask_merger)
        if debug:
            for i in range(0, images.shape[0]):
                f_name = "/tmp/salidas/{}_{}_image_output.jpeg".format(positions[0][0], positions[0][1])
                skimage.io.imsave(f_name, output_img)
                f_name = "/tmp/salidas/{}_{}_image_input.jpeg".format(positions[0][0], positions[0][1])
                skimage.io.imsave(f_name, images[i])

    # normalizar
    norm_factor = 255 // output_img.max()
    output_img = output_img * norm_factor

    skimage.io.imsave(output_path, output_img)


def read_img(path):
    rimg = io.imread(path)
    return rimg


class InferenceConfig(VineyardConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8


if __name__ == '__main__':
    # load srs model
    logs = '/media/gus/workspace/wml/vineyard-sketcher/logs'
    input_folder = '/media/gus/data/rasters/aerial/pnoa/2020/'

    output_folder = '/media/gus/workspace/wml/vineyard-sketcher/results/iteration3/predictions/'

    input_images = [os.path.join(input_folder, f_image) for f_image in os.listdir(input_folder) if
                    f_image.endswith(".tif")]
    # input_images = [x for x in input_images if 'PNOA_CYL_2020_25cm_OF_etrsc_rgb_hu30_h05_0345_8-6.tif' in x]

    input_images = [
        # '/media/gus/workspace/wml/vineyard-sketcher/test/resources/aerial0.tif',
        # '/media/gus/workspace/wml/vineyard-sketcher/test/resources/aerial1.tif',
        '/media/gus/workspace/wml/vineyard-sketcher/test/resources/aerial2.tif'
    ]
    input_images.sort()

    ######################
    ### CONFIG
    ######################
    patch_size = 512

    config = InferenceConfig()
    batch_size = config.IMAGES_PER_GPU
    weights_path = '/media/gus/workspace/wml/vineyard-sketcher/results/iteration3/mask_rcnn_vineyard_0037.h5'
    # weights_path = '/media/gus/workspace/wml/vineyard-sketcher/results/iteration2/mask_rcnn_vineyard_0030.h5'

    # model initialization
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs)
    model.load_weights(weights_path, by_name=True)

    total = len(input_images)
    tag = "sketcher"
    for idx, input in enumerate(input_images):
        logging.info("Processing image {} of {} - {}".format(idx, total, input))
        filename = os.path.basename(input)
        base, ext = os.path.splitext(filename)
        outf = os.path.join(output_folder, "{}_{}{}".format(base, tag, ext))
        apply_model(input, outf, model, window_size=(512, 512), step_size=384, batch_size=batch_size)

        logging.info("Applying geolocation info.")
        rimg = read_img(outf)
        # rimg = rimg[:, :, np.newaxis] solo con opencv
        georeference_image(rimg, input, outf, scale=1, bands=1)
        logging.info("Finished with file {}.".format(input))
        logging.info("Generated file {}.".format(outf))

        # plt.show()
        logging.info("========================================")
        logging.info("Model inference  on raster finished.")
        logging.info("========================================")
