import torch
import torch.optim
import torch.nn as nn
import torch.functional as f
import torchvision
import numpy as np
import models
import modules

if __name__ == '__main__':
    bce_loss = nn.BCELoss().cuda()

    ones = torch.ones([8, 1])
    zeros = torch.zeros([8, 1])
    rand = torch.randn([8, 1])
    rand = torch.sigmoid(rand)
    print(bce_loss(zeros, ones).item())
