import os
import numpy as np
import matplotlib.pyplot as plt


class ImageSet:
    def __init__(self, path, is_binary=False, is_uint8=True,require_cwh=False):
        """
        :type path: str
        """
        if not path.endswith(('/', '\\')):
            path = path + '/'
        # print(path)
        self.root = path
        self.items = os.listdir(path)
        self.is_binary = is_binary
        self.is_uint8 = is_uint8
        self.require_cwh = require_cwh
        return

    def _read_images(self, indices):
        images = list()
        for x in indices:
            image: np.ndarray = plt.imread(self.root + self.items[x])
            image = image.astype(dtype=np.float32)
            if self.is_uint8:
                image = image * (1.0 / 255.0)
            if self.is_binary:
                image = np.expand_dims(image, 3)
            if self.require_cwh:
                image = np.transpose(image, (2, 0, 1))
            images.append(image)
        return np.array(images)

    def select(self, count):
        total = len(self.items)
        indices = np.random.randint(0, total, size=count)
        return self._read_images(indices)

    def pick(self, count, offset):
        total = len(self.items)
        indices = list()
        for x in range(offset, offset + count):
            if x >= total:
                indices.append(x - total)
            else:
                indices.append(x)
        return self._read_images(indices)


if __name__ == '__main__':
    # folder = ImageSet("../d")
    pass
