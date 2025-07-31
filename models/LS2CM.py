import torch
import torch.nn as nn
from torch.backends import cudnn
from scipy import io
import numpy as np
import random
import math


class LS2CM(nn.Module):
    def __init__(self, input_channels, output_channels, relu=True):
        super(LS2CM, self).__init__()
        reduction_N = math.ceil(output_channels / 2)
        self.point_wise = nn.Conv2d(input_channels, reduction_N, kernel_size=1, padding=0)
        self.depth_wise = nn.Conv2d(reduction_N, reduction_N, kernel_size=3, padding=1, groups=reduction_N)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Sequential()
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x_1 = self.point_wise(x)
        x_2 = self.depth_wise(x_1)
        x = torch.cat((x_2, x_1), dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResLS2CM(nn.Module):
    def __init__(self, inp, oup):
        super(ResLS2CM, self).__init__()

        self.ResBlock = nn.Sequential(
            LS2CM(inp, oup, relu=True),
            LS2CM(oup, oup, relu=False),
        )

        if inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.ResBlock(x) + self.shortcut(x)


class BaseNet(nn.Module):

    def __init__(self, band, classes):
        super(BaseNet, self).__init__()

        self.stem = nn.Conv2d(band, 16, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.Block1 = ResLS2CM(16, 16)
        self.Block2 = ResLS2CM(16, 32)
        self.Block3 = ResLS2CM(32, 64)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, classes)

    def forward(self, x):
        x = x.squeeze(1)
        x1 = self.stem(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.Block1(x1)
        x2 = self.Block2(x2)
        x2 = self.Block3(x2)

        x3 = self.avgpool(x2)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc1(x3)

        return x3


def LS2CMNet(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = BaseNet(band=204, classes=16)
    elif dataset == 'pu':
        model = BaseNet(band=103, classes=9)
    elif dataset == 'whulk':
        model = BaseNet(band=270, classes=9)
    elif dataset == 'whuhh':
        model = BaseNet(band=270, classes=22)
    elif dataset == 'hrl':
        model = BaseNet(band=176, classes=14)
    elif dataset == 'IP':
        model = BaseNet(band=200, classes=16)
    elif dataset == 'whuhc':
        model = BaseNet(band=274, classes=16)
    elif dataset == 'BS':
        model = BaseNet(band=145, classes=14)
    elif dataset == 'HsU':
        model = BaseNet(band=144, classes=15)
    elif dataset == 'KSC':
        model = BaseNet(band=176, classes=13)
    elif dataset == 'pc':
        model = BaseNet(band=102, classes=9)

    return model


if __name__ == '__main__':
    # A Lightweight Spectral-Spatial Convolution Module for Hyperspectral Image Classification

    t = torch.randn(size=(64, 1, 200, 7, 7))
    print("input shape:", t.shape)
    net = LS2CMNet(dataset='IP', patch_size=7)
    print("output shape:", net(t).shape)


    from fvcore.nn import FlopCountAnalysis, flop_count_table

    net.eval()
    flops = FlopCountAnalysis(net, t)
    print(flop_count_table(flops))

    from thop import profile, clever_format

    flops, params = profile(net, inputs=(t,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)