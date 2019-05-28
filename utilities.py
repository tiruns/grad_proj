import numpy as np


def tensor_to_numpy(tensor, index=0):
    array = tensor.cpu().detach().numpy()[index]
    array = np.transpose(array, (1, 2, 0))
    if array.shape[2] == 1:
        array = array.reshape([array.shape[0], array.shape[1]])
    return array


def restore_image(image):
    image = np.transpose(image, (1, 2, 0))
    if image.shape[2] == 1:
        image = image.reshape([image.shape[0], image.shape[1]])
    return image
