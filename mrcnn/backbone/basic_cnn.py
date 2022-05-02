import math

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL

layers = tf.keras.layers
keras = tf.keras


def build(inputs, num_classes=2, augmentation_layer=None):
    x = inputs
    # Image augmentation block
    if augmentation_layer is not None:
        x = augmentation_layer(inputs)

    x = tf.keras.layers.experimental.preprocessing.Resizing(144, 144)(x)
    # Entry block
    x = KL.Conv2D(32, 3, strides=2, padding="same")(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation("relu")(x)

    x = KL.Conv2D(64, 3, padding="same")(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512]:
        x = KL.Activation("relu")(x)
        x = KL.SeparableConv2D(size, 3, padding="same")(x)
        x = KL.BatchNormalization()(x)

        x = KL.Activation("relu")(x)
        x = KL.SeparableConv2D(size, 3, padding="same")(x)
        x = KL.BatchNormalization()(x)

        x = KL.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = KL.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = KL.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = KL.SeparableConv2D(1024, 3, padding="same")(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation("relu")(x)

    x = KL.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = KL.Dropout(0.25)(x)
    outputs = KL.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def cnn_graph(self, input_image, stage5=False, train_bn=True):
    x = input_image

    # Image augmentation block
    # print("________________ {}".format(augmentation_layer))
    # if augmentation_layer is not None:
    #     x = augmentation_layer(input_image)

    # x = tf.keras.layers.experimental.preprocessing.Resizing(144, 144)(x)
    # Entry block
    x = KL.Conv2D(32, 3, strides=2, padding="same")(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation("relu")(x)

    C1 = x = KL.Conv2D(64, 3, padding="same")(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    feature_layers = []
    for size in [128, 256, 512]:
        x = KL.Activation("relu")(x)
        x = KL.SeparableConv2D(size, 3, padding="same")(x)
        x = KL.BatchNormalization()(x)

        x = KL.Activation("relu")(x)
        x = KL.SeparableConv2D(size, 3, padding="same")(x)
        x = KL.BatchNormalization()(x)

        x = KL.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = KL.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = KL.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        feature_layers.append(x)

    x = KL.SeparableConv2D(1024, 3, padding="same")(x)
    x = KL.BatchNormalization()(x)
    if stage5:
        C5 = KL.Activation("relu")(x)
    else:
        C5 = None

    return [C1] + feature_layers + [C5]


def compute_backbone_shapes(self, image_shape, config):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
          int(math.ceil(image_shape[1] / stride))]
         for stride in config.BACKBONE_STRIDES])


if __name__ == '__main__':
    # try to load weights

    w_path = "/media/gus/workspace/wml/vineyard-sketcher/resources/model/cnnv1.md/check1"
    input_image = KL.Input(shape=[None, None, 3], name="input_image")
    # model = cnn_graph(input_image)
    model = build(input_image)

    model.load_weights(w_path)



    model.summary()
    print("Loaded!")
