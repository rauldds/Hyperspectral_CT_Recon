import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from config import ResType

# Simple scaling layer
class ScaledLayer(nn.Module):
    def __init__(self, init_value = 1):
        super(ScaledLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.W

# Deconvolution layer: upsample, convolution and apply activation
class Deconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, kernel=(3, 3), padding='same', stride=(1, 1)):
        super(Deconv2D, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, stride=stride)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3, 3), padding='same', stride=(1, 1), use_bn=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, stride=stride)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, kernel=(3, 3), padding='same', stride=(1, 1)):
        super(ResidualBlock, self).__init__()

        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else None

    def forward(self, x):
        identity = x
        out = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        out = F.relu(out)
        out = self.bn2(self.conv2(out)) if self.use_bn else self.conv2(out)

        out = out + identity
        out = F.relu(out)

        return out
