import numpy as np


def clean_instaces(image, min_area=20):
    """
    Removes smal instances
    :param image:
    :return:
    """
    ids, counts = np.unique(image, return_counts=True)
    removables = [ids for i, ids in enumerate(ids) if counts[i] < min_area]
    for r in range(0, len(removables)):
        image[image == removables[r]] = 0


def get_max_intersect(instances, counts, min_intersect=20):
    max_counts = -1
    max_instance = None
    for i in range(0, len(instances)):
        if instances[i] == 0:
            continue
        if counts[i] > max_counts and counts[i] > min_intersect:
            max_counts = counts[i]
            max_instance = instances[i]
    return max_instance


class MaskMerger:
    """

    """

    def __init__(self):
        self.current_instance_id = 0
        self.min_intersect = 50

    def apply(self, img, mask_list, position, size=512):
        x, y = position
        # check in the image if there's an area that overlaps with current mask
        for i in range(0, mask_list.shape[-1]):
            mask = mask_list[:, :, i].reshape((512, 512, 1))
            # intersect masks that share the same area
            inters = np.where(img[y:y + size, x:x + size] * mask > 1, img[y:y + size, x:x + size], 0)

            instances, counts = np.unique(inters, return_counts=True)
            # find the instance ids that overlaps with current mask and keep the one of the bigger area (px2)
            instance_id = get_max_intersect(instances, counts, self.min_intersect)
            if instance_id is None:
                # no overlapped instance is found, create a new instance_id
                self.current_instance_id += 1
                instance_id = self.current_instance_id
            img[y:y + size, x:x + size] = np.where(mask, instance_id, img[y:y + size, x:x + size])
