import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
import numpy as np
import os
import models
import mask_generator
import utilities
import storage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHES = 96000
VALIDATION_INTERVAL = 16
PREVIEW_INTERVAL = 128
SAVING_INTERVAL = 512
MASK_ENCODER_DIR = "../outputs/train_mask_encoder/"
OUTPUT_DIR = "../outputs/train_image_encoder_min/"


class Train:
    def __init__(self):
        self.images = storage.ImageSet("../datasets/celeba_160/", is_uint8=False, require_cwh=True)
        self.val_images = storage.ImageSet("../datasets/celeba_160_val/", is_uint8=False, require_cwh=True)
        self.masks = mask_generator.MaskSet(320, mask_size=(160, 128), num_holes=2, size_holes=24, border=32)

        self.image_encoder = models.ImageEncoder().to(DEVICE)
        self.image_decoder = models.ImageDecoderMin().to(DEVICE)
        self.image_discriminator = models.ImageDiscriminator().to(DEVICE)

        if os.path.exists(OUTPUT_DIR + "image_decoder.params"):
            self.image_encoder.load_state_dict(torch.load(OUTPUT_DIR + "image_encoder.params"))
            self.image_decoder.load_state_dict(torch.load(OUTPUT_DIR + "image_decoder.params"))
            self.image_discriminator.load_state_dict(torch.load(OUTPUT_DIR + "image_discriminator.params"))

        gen_params = list()
        for param in self.image_encoder.parameters():
            gen_params.append(param)
        for param in self.image_decoder.parameters():
            gen_params.append(param)
        dis_params = list()
        for param in self.image_discriminator.parameters():
            dis_params.append(param)

        self.mse_loss = nn.MSELoss().cuda()
        self.bce_loss = nn.BCELoss().cuda()
        self.gen_mse_opti = torch.optim.Adam(gen_params, lr=1e-2)
        self.gen_bce_opti = torch.optim.Adam(gen_params, lr=2e-3)
        self.dis_opti = torch.optim.Adam(dis_params, lr=2e-3)

        self.ones = torch.ones([BATCH_SIZE, 1]).to(DEVICE)
        self.zeros = torch.zeros([BATCH_SIZE, 1]).to(DEVICE)

        return

    def prepare_tensors(self, size, is_train=True):
        real_images = None
        if is_train:
            real_images = self.images.select(size)
        else:
            real_images = self.val_images.select(size)
        real_images = torch.from_numpy(real_images).to(DEVICE)
        masks = self.masks.select(size)
        masks = torch.from_numpy(masks).to(DEVICE)
        return real_images, masks

    def optimize_bce(self, epoch):
        real_images, masks = self.prepare_tensors(BATCH_SIZE)

        # dis: real images
        real_quality = self.image_discriminator(real_images)

        self.dis_opti.zero_grad()
        loss = self.bce_loss(real_quality, self.ones)
        loss.backward()
        self.dis_opti.step()

        # dis fake images
        image_features = self.image_encoder(torch.mul(real_images, masks))
        fake_images = self.image_decoder(image_features)
        fake_quality = self.image_discriminator(fake_images)

        self.dis_opti.zero_grad()
        loss = self.bce_loss(fake_quality, self.zeros)
        loss.backward()
        self.dis_opti.step()

        # gen: bce
        real_images, masks = self.prepare_tensors(BATCH_SIZE)

        image_features = self.image_encoder(torch.mul(real_images, masks))
        fake_images = self.image_decoder(image_features)
        fake_quality = self.image_discriminator(fake_images)

        self.gen_bce_opti.zero_grad()
        loss = self.bce_loss(fake_quality, self.ones)
        loss.backward()
        self.gen_bce_opti.step()

        return

    def optimize_mse(self, epoch):
        real_images, masks = self.prepare_tensors(BATCH_SIZE)

        # mse
        image_features = self.image_encoder(torch.mul(real_images, masks))
        fake_images = self.image_decoder(image_features)

        self.gen_mse_opti.zero_grad()
        loss = self.mse_loss(fake_images, real_images)
        loss.backward()
        self.gen_mse_opti.step()

        return

    def validate(self, epoch):
        real_images, masks = self.prepare_tensors(BATCH_SIZE, is_train=False)

        image_features = self.image_encoder(torch.mul(real_images, masks))
        fake_images = self.image_decoder(image_features)
        fake_quality = self.image_discriminator(fake_images)
        real_quality = self.image_discriminator(real_images)

        rec_loss = self.mse_loss(real_images, fake_images).item()
        adv_loss = self.bce_loss(fake_quality, self.ones)
        dis_loss = (self.bce_loss(real_quality, self.ones).item() +
                    self.bce_loss(fake_quality, self.zeros).item()) * 0.5

        print("Batch: {0:d}\tRecLoss: {1:.4f}\tAdvLoss: {2:.4f}\tDisLoss: {3:.4f}".format(
            epoch, rec_loss, adv_loss, dis_loss))

        return

    def save_model(self, epoch):
        path = OUTPUT_DIR + "checkpoints/"
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.image_encoder.state_dict(), path + "image_encoder_" + str(epoch) + ".params")
        torch.save(self.image_decoder.state_dict(), path + "image_decoder_" + str(epoch) + ".params")
        torch.save(self.image_discriminator.state_dict(), path + "image_discriminator_" + str(epoch) + ".params")
        return

    def generate_preview(self, epoch):
        path = OUTPUT_DIR + "previews/"
        if not os.path.exists(path):
            os.mkdir(path)

        real_images, masks = self.prepare_tensors(2, is_train=False)
        masked_images = torch.mul(real_images, masks)
        image_features = self.image_encoder(masked_images)
        fake_images = self.image_decoder(image_features)

        rows, cols = 2, 3
        fig, axs = plt.subplots(rows, cols)
        for a in range(rows):
            for b in range(cols):
                axs[a, b].axis('off')

        axs[0, 0].imshow(utilities.tensor_to_numpy(real_images, 0))
        axs[0, 1].imshow(utilities.tensor_to_numpy(masked_images, 0), cmap='gray')
        axs[0, 2].imshow(utilities.tensor_to_numpy(fake_images, 0))
        axs[1, 0].imshow(utilities.tensor_to_numpy(real_images, 1))
        axs[1, 1].imshow(utilities.tensor_to_numpy(masked_images, 1), cmap='gray')
        axs[1, 2].imshow(utilities.tensor_to_numpy(fake_images, 1))

        fig.savefig(path + "preview_" + str(epoch) + '.jpg', dpi=320)
        plt.close()
        return

    def run(self):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)

        self.image_encoder.train()
        self.image_decoder.train()
        self.image_discriminator.train()

        for epoch in range(EPOCHES):
            self.optimize_mse(epoch)
            if epoch % 4 == 0:
                self.optimize_bce(epoch)

            if epoch == 0 or (epoch + 1) % VALIDATION_INTERVAL == 0:
                self.validate(epoch)
            if epoch == 0 or (epoch + 1) % PREVIEW_INTERVAL == 0:
                self.generate_preview(epoch)
            if (epoch + 1) % SAVING_INTERVAL == 0:
                self.save_model(epoch)

        return


if __name__ == '__main__':
    app = Train()
    app.run()
