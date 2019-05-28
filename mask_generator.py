import numpy as np
from matplotlib import pyplot


def generate_mask(mask_size=192, num_holes=3, size_holes=24, border=32, expand_dim=False):
    mask = np.ones(shape=(mask_size, mask_size), dtype=np.float32)
    high = mask_size - border - size_holes + 1
    coords = np.random.randint(border, high, size=(num_holes, 2))
    for coord in coords:
        mask[coord[0]:coord[0] + size_holes, coord[1]:coord[1] + size_holes] = 0
    if expand_dim:
        mask = np.expand_dims(mask, 0)
    return mask


class MaskSet:
    def __init__(self, count):
        self.masks = list()
        for _ in range(count):
            self.masks.append(generate_mask(expand_dim=True))
        return

    def select(self, count):
        total = len(self.masks)
        indices = np.random.randint(0, total, size=count)
        masks = list()
        for index in indices:
            masks.append(self.masks[index])
        return np.array(masks)


if __name__ == '__main__':
    m = generate_mask()
    print(m.shape)
    # pyplot.imshow(mask, cmap='gray')
    # pyplot.show()
