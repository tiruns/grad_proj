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
BATCH_SIZE = 8
EPOCHES = 960000
VALIDATION_INTERVAL = 16
PREVIEW_INTERVAL = 128
SAVING_INTERVAL = 1024
MASK_ENCODER_DIR = "../grad_proj_outputs/train_mask_encoder/"
OUTPUT_DIR = "../grad_proj_outputs/train_image_encoder/"


class Train:
    def __init__(self):
        self.images = storage.ImageSet("../datasets/avatar_test/train/", require_cwh=True)
        self.val_images = storage.ImageSet("../datasets/avatar_test/validation/", require_cwh=True)
        self.masks = mask_generator.MaskSet(320)

        self.mask_encoder = models.MaskEncoder().to(DEVICE)
        self.image_encoder = models.ImageEncoder().to(DEVICE)
        self.image_decoder = models.ImageDecoder().to(DEVICE)
        self.image_discriminator = models.ImageDiscriminator().to(DEVICE)

        self.mask_encoder.load_state_dict(torch.load(MASK_ENCODER_DIR + "mask_encoder.params"))

        for param in self.mask_encoder.parameters():
            param.requires_grad = False

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
        self.gen_bce_opti = torch.optim.Adam(gen_params, lr=4e-3)
        self.dis_opti = torch.optim.Adam(dis_params, lr=4e-4)

        self.ones = torch.ones([BATCH_SIZE, 1]).to(DEVICE)
        self.zeros = torch.zeros([BATCH_SIZE, 1]).to(DEVICE)

        return

    def forward(self, images, masks):
        real_images = torch.from_numpy(images).to(DEVICE)
        real_masks = torch.from_numpy(masks).to(DEVICE)

        image_features = self.image_encoder(torch.mul(real_images, real_masks))
        mask_features = self.mask_encoder(real_masks)
        fake_images = self.image_decoder(image_features, mask_features)
        real_quality = self.image_discriminator(real_images)
        fake_quality = self.image_discriminator(fake_images)
        return real_images, fake_images, real_quality, fake_quality

    def validate(self, epoch):
        val_images = self.val_images.select(BATCH_SIZE)
        val_masks = self.masks.select(BATCH_SIZE)
        real_images, fake_images, real_quality, fake_quality = self.forward(val_images, val_masks)
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

        val_images = self.val_images.select(2)
        val_masks = self.masks.select(2)

        _0, fake_images, _1, _2 = self.forward(val_images, val_masks)

        rows, cols = 2, 3
        fig, axs = plt.subplots(rows, cols)
        for a in range(rows):
            for b in range(cols):
                axs[a, b].axis('off')

        masked_image_0 = list()
        for channel in val_images[0]:
            masked_image_0.append(np.multiply(channel, val_masks[0]).reshape([192, 192]))
        masked_image_1 = list()
        for channel in val_images[1]:
            masked_image_1.append(np.multiply(channel, val_masks[1]).reshape([192, 192]))

        axs[0, 0].imshow(utilities.restore_image(val_images[0]))
        axs[0, 1].imshow(utilities.restore_image(np.array(masked_image_0)), cmap='gray')
        axs[0, 2].imshow(utilities.tensor_to_numpy(fake_images, 0))
        axs[1, 0].imshow(utilities.restore_image(val_images[1]))
        axs[1, 1].imshow(utilities.restore_image(np.array(masked_image_1)), cmap='gray')
        axs[1, 2].imshow(utilities.tensor_to_numpy(fake_images, 1))

        fig.savefig(path + "preview_" + str(epoch) + '.jpg', dpi=320)
        plt.close()
        return

    def run(self):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)

        self.mask_encoder.eval()
        self.image_encoder.train()
        self.image_decoder.train()
        self.image_discriminator.train()

        for epoch in range(EPOCHES):
            images = self.images.select(BATCH_SIZE)
            masks = self.masks.select(BATCH_SIZE)

            real_images, fake_images, real_quality, fake_quality = self.forward(images, masks)

            # Step 1: optimize discriminator
            self.dis_opti.zero_grad()
            loss = self.bce_loss(real_quality, self.ones)
            loss.backward(retain_graph=True)
            self.dis_opti.step()

            self.dis_opti.zero_grad()
            loss = self.bce_loss(fake_quality, self.zeros)
            loss.backward(retain_graph=True)
            self.dis_opti.step()

            # Step 2: optimize encoder & decoder
            self.gen_mse_opti.zero_grad()
            loss = self.mse_loss(real_images, fake_images)
            loss.backward(retain_graph=True)
            self.gen_mse_opti.step()

            self.gen_bce_opti.zero_grad()
            loss = self.bce_loss(fake_quality, self.ones)
            loss.backward()
            self.gen_bce_opti.step()

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
