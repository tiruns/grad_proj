import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import modules


class MaskEncoder(nn.Module):
    def __init__(self):
        super(MaskEncoder, self).__init__()
        self.model = nn.Sequential(
            modules.DSBlock(1, 16, batch_norm=False),
            modules.DSBlock(16, 32, dropout=0.1),
            modules.DSBlock(32, 64),
            modules.DSBlock(64, 96, dropout=0.1),
            modules.DSBlock(96, 128),
            modules.GroupLinear(2, 128, 6, dropout=0.2))
        return

    def forward(self, x):
        x = self.model(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self):
        super(MaskDecoder, self).__init__()
        self.model = nn.Sequential(
            modules.USBlock(128, 96),
            modules.USBlock(96, 64, dropout=0.1),
            modules.USBlock(64, 32),
            modules.USBlock(32, 16, dropout=0.1),
            modules.USBlock(16, 1, batch_norm=False, use_sigmoid=True))
        return

    def forward(self, x):
        x = self.model(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = nn.Sequential(
            modules.DSBlock(3, 32, batch_norm=False),
            modules.DSBlock(32, 64, dropout=0.1),
            modules.ResBlock(64),
            modules.DSBlock(64, 128),
            modules.ResBlock(128),
            modules.DSBlock(128, 256, dropout=0.1),
            modules.ResBlock(256),
            modules.DSBlock(256, 512),
            modules.GroupLinear(8, 512, 6, dropout=0.2))
        return

    def forward(self, x):
        x = self.model(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()

        self.image_us1 = modules.USBlock(512+128, 256)
        self.image_res1 = modules.ResBlock(256)
        self.image_us2 = modules.USBlock(256+96, 128, dropout=0.1)
        self.image_res2 = modules.ResBlock(128)

        self.mask_us1 = modules.USBlock(128, 96)
        self.mask_us2 = modules.USBlock(96, 64, dropout=0.1)

        self.merged = nn.Sequential(
            modules.USBlock(128 + 64, 64),
            modules.ResBlock(64),
            modules.USBlock(64, 32, dropout=0.1),
            modules.USBlock(32, 3, batch_norm=False, use_sigmoid=True))
        return

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.image_us1(x)
        x = self.image_res1(x)
        y = self.mask_us1(y)

        x = torch.cat((x,y), dim=1)
        x = self.image_us2(x)
        x = self.image_res2(x)
        y = self.mask_us2(y)

        x = torch.cat((x,y), dim=1)
        x = self.merged(x)

        return x


class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        self.model = nn.Sequential(
            modules.DSBlock(3, 24, padding=False, batch_norm=False),
            modules.DSBlock(24, 40, padding=False, dropout=0.1),
            modules.DSBlock(40, 64, padding=False),
            modules.DSBlock(64, 96, padding=False, dropout=0.1),
            modules.DSBlock(96, 128,padding=False))
        self.fc1 = nn.Linear(128*5*5, 256)
        self.fc2 = nn.Linear(256, 1)
        pass

    def forward(self, x):
        x = self.model(x)
        x = x.view([x.size()[0], -1])
        x = self.fc1(x)
        x = f.leaky_relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

