import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
import os
import models
import mask_generator
import utilities

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHES = 12800
VALIDATION_INTERVAL = 64
PREVIEW_INTERVAL = 256
SAVING_INTERVAL = 1024
OUTPUT_DIR = "../outputs/train_mask_encoder/"


class Train:
    def __init__(self):
        self.samples = mask_generator.MaskSet(640)
        self.val_samples = mask_generator.MaskSet(32)
        self.mask_encoder = models.MaskEncoder().to(DEVICE)
        self.mask_decoder = models.MaskDecoder().to(DEVICE)

        params = list()
        for param in self.mask_encoder.parameters():
            params.append(param)
        for param in self.mask_decoder.parameters():
            params.append(param)

        self.loss = nn.MSELoss().cuda()
        self.opti = torch.optim.Adam(params, lr=2e-3)
        return

    def forward(self, samples):
        real_masks = torch.from_numpy(samples).to(DEVICE)
        features = self.mask_encoder(real_masks)
        fake_masks = self.mask_decoder(features)
        return real_masks, fake_masks

    def validate(self, epoch):
        real_masks, fake_masks = self.forward(self.samples.select(BATCH_SIZE))
        loss = self.loss(real_masks, fake_masks)
        print("Batch: {0:d}\tLoss: {1:.4f}".format(epoch, loss.item()))
        return

    def save_model(self, epoch):
        path = OUTPUT_DIR + "checkpoints/"
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.mask_encoder.state_dict(), path + "mask_encoder_" + str(epoch) + ".params")
        torch.save(self.mask_decoder.state_dict(), path + "mask_decoder_" + str(epoch) + ".params")
        return

    def generate_preview(self, epoch):
        path = OUTPUT_DIR + "previews/"
        if not os.path.exists(path):
            os.mkdir(path)

        real_masks, fake_masks = self.forward(self.val_samples.select(2))
        rows, cols = 2, 2
        fig, axs = plt.subplots(rows, cols)
        for a in range(rows):
            for b in range(cols):
                axs[a, b].axis('off')

        axs[0, 0].imshow(utilities.tensor_to_numpy(real_masks, 0), cmap='gray')
        axs[1, 0].imshow(utilities.tensor_to_numpy(real_masks, 1), cmap='gray')
        axs[0, 1].imshow(utilities.tensor_to_numpy(fake_masks, 0), cmap='gray')
        axs[1, 1].imshow(utilities.tensor_to_numpy(fake_masks, 1), cmap='gray')

        fig.savefig(path + "preview_" + str(epoch) + '.jpg', dpi=320)
        plt.close()
        return

    def run(self):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)

        self.mask_decoder.train()
        self.mask_decoder.train()

        for epoch in range(EPOCHES):
            self.opti.zero_grad()
            real_masks, fake_masks = self.forward(self.samples.select(BATCH_SIZE))
            loss = self.loss(real_masks, fake_masks)
            loss.backward()
            self.opti.step()

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
