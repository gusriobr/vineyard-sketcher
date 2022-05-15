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
                self.current_instance_id += 5 # avoid consecutive Ids
                instance_id = self.current_instance_id
            img[y:y + size, x:x + size] = np.where(mask, instance_id, img[y:y + size, x:x + size])


class PrimeIdMasMerger:
    """

    """

    def __init__(self):
        self.current_instance_id = 0
        self.min_intersect = 50
        self.primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
                       103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
                       199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
                       313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
                       433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557,
                       563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661,
                       673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
                       811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937,
                       941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049,
                       1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
                       1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223]

    def apply(self, img, mask_list, position, size=512):
        x, y = position
        # check in the image if there's an area that overlaps with current mask
        for i in range(0, mask_list.shape[-1]):
            mask = mask_list[:, :, i].reshape((512, 512, 1))
            instance_id = self.current_instance_id

            img[y:y + size, x:x + size] = np.where(mask, img[y:y + size, x:x + size] * instance_id,
                                                   img[y:y + size, x:x + size])
