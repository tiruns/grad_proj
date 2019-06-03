import numpy as np
import matplotlib.pyplot as plt


def generate_mask(mask_size, num_holes, size_holes, border, expand_dim):
    mask = np.ones(shape=mask_size, dtype=np.float32)
    w_high = mask_size[0] - border - size_holes + 1
    h_high = mask_size[1] - border - size_holes + 1
    w_coords = np.random.randint(border, w_high, size=(num_holes,))
    h_coords = np.random.randint(border, h_high, size=(num_holes,))
    z = w_coords[0] + size_holes
    for x in range(num_holes):
        mask[w_coords[x]:w_coords[x] + size_holes, h_coords[x]:h_coords[x] + size_holes] = 0
    if expand_dim:
        mask = np.expand_dims(mask, 0)
    return mask


class MaskSet:
    def __init__(self, count, mask_size=(192, 192), num_holes=3, size_holes=24, border=32, expand_dim=True):
        self.masks = list()
        for _ in range(count):
            self.masks.append(generate_mask(mask_size, num_holes, size_holes, border, expand_dim))
        return

    def select(self, count):
        total = len(self.masks)
        indices = np.random.randint(0, total, size=count)
        masks = list()
        for index in indices:
            masks.append(self.masks[index])
        return np.array(masks)


if __name__ == '__main__':
    # s = MaskSet(5, mask_size=(160, 128), num_holes=2, size_holes=24, border=32, expand_dim=False)
    # plt.imshow(s.select(1)[0], cmap='gray')
    # print(s.select(1)[0].shape)
    # plt.show()
    pass