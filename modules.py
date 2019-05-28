import torch
import torch.nn as nn
import torch.nn.functional as f


# Down-Sampling Block
class DSBlock(nn.Module):
    def __init__(self, in_chennels, out_chennels, padding=True, dropout=None, batch_norm=True):
        super(DSBlock, self).__init__()
        self.conv = nn.Conv2d(in_chennels, out_chennels,
                              kernel_size=3, stride=2)
        if padding:
            self.pad = nn.ReflectionPad2d((0, 1, 0, 1))
        else:
            self.pad = None

        if dropout is not None:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None

        if batch_norm:
            self.bn = nn.BatchNorm2d(out_chennels)
        else:
            self.bn = None
        return

    def forward(self, x):
        x = self.conv(x)
        if self.pad is not None:
            x = self.pad(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.bn is not None:
            x = self.bn(x)
        x = f.leaky_relu(x)
        return x


def bilinear_upsample(x):
    return f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


# Up-Sampling Block
class USBlock(nn.Module):
    def __init__(self, in_chennels, out_chennels, dropout=None, batch_norm=True, use_sigmoid=False):
        super(USBlock, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.conv = nn.Conv2d(in_chennels, out_chennels,
                              kernel_size=3)
        self.pad = nn.ReflectionPad2d(1)

        if dropout is not None:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None

        if batch_norm:
            self.bn = nn.BatchNorm2d(out_chennels)
        else:
            self.bn = None
        return

    def forward(self, x):
        x = bilinear_upsample(x)
        x = self.conv(x)
        x = self.pad(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        else:
            x = f.leaky_relu(x)
        return x


# Group-Based Linear Layer
class GroupLinear(nn.Module):
    def __init__(self, groups, channels, map_size, dropout=None):
        super(GroupLinear, self).__init__()
        self.groups = groups
        self.channels = channels
        self.map_size = map_size
        self.linear_nodes = int(map_size * map_size * channels / groups)
        check = map_size * map_size * channels % groups
        if check != 0:
            raise Exception('Invalid parameters for GroupLinear')
        self.fc = nn.Linear(self.linear_nodes, self.linear_nodes)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        return

    def forward(self, x):
        x = x.view([x.size()[0], self.groups, self.linear_nodes])
        x = self.fc(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view([x.size()[0], self.channels, self.map_size, self.map_size])
        x = f.leaky_relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, num_channels, dropout=None, batch_norm=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3)
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3)
        self.pad2 = nn.ReflectionPad2d(1)

        if dropout is not None:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None

        if batch_norm:
            self.bn = nn.BatchNorm2d(num_channels)
        else:
            self.bn = None
        return

    def forward(self, x):
        y = self.conv1(x)
        y = self.pad1(y)
        y = f.leaky_relu(y)
        y = self.conv2(y)
        y = self.pad2(y)
        if self.dropout is not None:
            y = self.dropout(y)
        if self.bn is not None:
            y = self.bn(y)
        y = f.leaky_relu(y)
        x = x + y
        return x
