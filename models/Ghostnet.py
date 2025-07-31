import math

import torch
from torch import nn


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out = channel // reduction
        if out == 0:
            out = 1
        self.fc = nn.Sequential(
            nn.Linear(channel, out),
            nn.ReLU(inplace=True),
            nn.Linear(out, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride == 2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class BaseNet(nn.Module):

    def __init__(self, band, classes, dropout=True):
        super(BaseNet, self).__init__()

        self.stem = nn.Conv2d(band, 16, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.2)

        self.BottleNeck1 = GhostBottleneck(16, 16, 16, 1, 1, True)
        self.BottleNeck2 = GhostBottleneck(16, 48, 24, 3, 1, True)
        self.BottleNeck3 = GhostBottleneck(24, 72, 24, 1, 1, True)

        self.conv5 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(72)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(72, 216)
        self.bn3 = nn.BatchNorm1d(216)
        self.fc2 = nn.Linear(216, classes)

    def forward(self, x):
        x = x.squeeze(1)
        x1 = self.stem(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if self.use_dropout:
            x1 = self.dropout(x1)

        x2 = self.BottleNeck1(x1)
        x2 = self.BottleNeck2(x2)
        x2 = self.BottleNeck3(x2)

        x3 = self.conv5(x2)
        x3 = self.bn2(x3)
        x3 = self.relu(x3)
        x3 = self.avgpool(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc1(x3)
        x3 = self.bn3(x3)
        if self.use_dropout:
            x1 = self.dropout(x1)
        x3 = self.fc2(x3)

        return x3

def GhostNet(dataset, patch_size):
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
    # Ghostnet for Hyperspectral Image Classification

    t = torch.randn(size=(64, 1, 200, 7, 7))
    print("input shape:", t.shape)
    net = GhostNet(dataset='IP', patch_size=7)
    print("output shape:", net(t).shape)


    from fvcore.nn import FlopCountAnalysis, flop_count_table

    net.eval()
    flops = FlopCountAnalysis(net, t)
    print(flop_count_table(flops))

    from thop import profile, clever_format

    flops, params = profile(net, inputs=(t,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)