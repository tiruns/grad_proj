import numpy as np
import math


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


# images: float32 [0.0, 255.0)
def cal_psnr(image1, image2):
    val = (np.mean(np.square(image1 - image2)))
    psnr_val = float("inf")
    if val > 1e-6:
        psnr_val = 10 * np.log10(255 * 255 / val)
    return psnr_val


if __name__ == '__main__':
    pass
