def sliding_window(image, window_size=(32, 32), step_size=5, treat_borders=True):
    """
    creates sequences of images from ana original input image sliding a window across the input extracting patches
    :param image: input image
    :param step_size:  stride size, same in w and h direction
    :param window_size: (h,w) of the output images
    :param treat_borders: if True, when the input image size is not a multiple of the patch, at the end of each row, the
     path is moved to the last position to return an image with the remaining pixels.
    :return: image of "window_size" shape
    """
    # treat image - patch size differences
    if window_size[0] > image.shape[0]:
        window_size[0] = image.shape[0]
    if window_size[1] > image.shape[1]:
        window_size[1] = image.shape[1]
    # if image and patch size are the same, at least one image must be returned
    if window_size[0] == image.shape[0]:
        y_values = [0]
    else:
        y_values = list(range(0, image.shape[0] - window_size[0], step_size))
    if window_size[1] == image.shape[1]:
        x_values = [0]
    else:
        x_values = list(range(0, image.shape[1] - window_size[1], step_size))

    if treat_borders and y_values[-1] + window_size[0] != image.shape[0]:
        y_values.append(image.shape[0] - window_size[0])
    if treat_borders and x_values[-1] + window_size[1] != image.shape[1]:
        x_values.append(image.shape[1] - window_size[1])

    for y in y_values:
        for x in x_values:
            yield image[y:y + window_size[0], x:x + window_size[1], :], x, y
