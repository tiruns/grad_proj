import models
import storage
import mask_generator
import numpy as np
import matplotlib.pyplot as plt
import torch
import utilities
import os

EVALUATE_DIR = "../evaluate_min/"
IMAGE_ENCODER_DIR = "../outputs/train_image_encoder_min/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    image_encoder = models.ImageEncoder().to(DEVICE)
    image_decoder = models.ImageDecoderMin().to(DEVICE)

    image_encoder.load_state_dict(torch.load(IMAGE_ENCODER_DIR + "image_encoder.params"))
    image_decoder.load_state_dict(torch.load(IMAGE_ENCODER_DIR + "image_decoder.params"))

    test_set = storage.ImageSet("../datasets/celeba_160_test_min/", is_uint8=False, require_cwh=True)

    mask = mask_generator.generate_mask(mask_size=(160, 128), num_holes=2, size_holes=24, border=32, expand_dim=False)
    plt.imsave(EVALUATE_DIR + "mask.png", mask, cmap="gray")
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, 0)
    mask = torch.from_numpy(mask).to(DEVICE)
    r_mask = 1.0 - mask

    _ = input("Press enter to continue:")

    for x in range(0, 100):
        real_image = test_set.pick(1, x)
        real_image = torch.from_numpy(real_image).to(DEVICE)

        image_features = image_encoder(torch.mul(real_image, mask))
        fake_image = image_decoder(image_features)
        recon_image = torch.mul(fake_image, r_mask) + torch.mul(real_image, mask)
        # recon_image = fake_image
        real_image = utilities.tensor_to_numpy(real_image)
        recon_image = utilities.tensor_to_numpy(recon_image)
        plt.imsave(EVALUATE_DIR + str(x) + "_real.png", real_image)
        plt.imsave(EVALUATE_DIR + str(x) + "_recon.png", recon_image)
