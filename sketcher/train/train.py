"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import datetime
import os
import sys

import imgaug
import numpy as np
import skimage.draw
import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # disable gpu
    tf.config.set_visible_devices([], 'GPU')

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.backbone.basic_cnn import cnn_graph, compute_backbone_shapes
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from sketcher.data.dataset import VineyardDataset

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class VineyardConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "vineyard"
    EPOCHS = 40

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 100

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    BACKBONE = cnn_graph
    #
    # # Only useful if you supply a callable to BACKBONE. Should compute
    # # the shape of each layer of the FPN Pyramid.
    # # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = compute_backbone_shapes


############################################################
#  Dataset
############################################################


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = VineyardDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VineyardDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")

    # augmentation = imgaug.augmenters.Sometimes(0.7, [
    #     imgaug.augmenters.Rot90((1, 3)),
    #     imgaug.augmenters.ShearX((-10, 10)),
    #     imgaug.augmenters.GaussianBlur(sigma=(0.0, 0.5))
    # ])
    augmentation = imgaug.augmenters.SomeOf((1, 3), [
        imgaug.augmenters.Identity(),
        imgaug.augmenters.Rot90((1, 3)),
        imgaug.augmenters.Fliplr(),
        imgaug.augmenters.ShearX((-10, 10)),
        imgaug.augmenters.GaussianBlur(sigma=(0.0, 0.8))
    ])
    augmentation = None

    model.train(dataset_train, dataset_val, augmentation=augmentation,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS,
                layers='heads')
    print("Train finished successfully.")


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/targets/x86_64-linux/lib" + ":" + os.environ["LD_LIBRARY_PATH"]

    # Parse command line arguments
    # sys.argv.extend(
    #     ["predict", "--weights", "/media/gus/workspace/wml/vineyard-sketcher/logs/balloon20220415T0925/mask_rcnn_balloon_0016.h5",
    #      "--image", "/media/gus/workspace/wml/vineyard-sketcher/resources/dataset/v2/val/img7.jpeg"])

    # sys.argv.extend(
    #     ["train", "--dataset", "/media/gus/workspace/wml/Mask_RCNN_TF2/datasets/balloon" "--weights", "coco"])
    #
    sys.argv.extend(
        ["train", "--dataset", "/media/gus/workspace/wml/vineyard-sketcher/resources/dataset/v2", "--weights", "vineyard"])
    """
    sys.argv.extend(
        ["predict", "--weights",
         '/media/gus/workspace/wml/vineyard-sketcher/results/iteration2/mask_rcnn_vineyard_0030.h5',
         "--image",
         '/media/gus/workspace/wml/vineyard-sketcher/results/iteration2/pred4.jpeg'])
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'predict'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color predict effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color predict effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "predict":
        assert args.image or args.video, \
            "Provide --image or --video to apply color predict"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VineyardConfig()
    else:
        class InferenceConfig(VineyardConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    elif args.weights.lower() == "vineyard":
        # Start from ImageNet trained weights
        weights_path = '/media/gus/workspace/wml/vineyard-sketcher/resources/model/vny/checkpoint.h5'
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() != "none":
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "predict":
        detect_and_color(model, image_path=args.image,
                         video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
