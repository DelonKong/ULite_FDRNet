import torch
from torch import nn


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out = channel // reduction
        if out == 0:
            out = 1
        self.fc = nn.Sequential(
            nn.Linear(channel, out, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class COS2M(nn.Module):
    def __init__(self, input_channels, reduction_N=16):
        super(COS2M, self).__init__()
        self.point_wise = nn.Conv2d(input_channels, reduction_N, kernel_size=1, padding=0, bias=False)
        self.depth_wise = nn.Conv2d(reduction_N, reduction_N, kernel_size=3, padding=1, groups=reduction_N, bias=False)
        self.conv3D = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 3), padding=(0, 0, 1),
                                stride=(1, 1, 1), bias=False)
        self.bn = nn.BatchNorm2d(reduction_N)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.squeeze(1)
        x_1 = self.point_wise(x)
        x_2 = self.depth_wise(x_1)
        # DSC
        x_3 = x_1.unsqueeze(1)
        x_3 = self.conv3D(x_3)
        x_3 = x_3.squeeze(1)

        x = torch.cat((x_2, x_3), dim=1)

        return x


class NPAF(nn.ReLU):
    def __init__(self, dim, act_num=1):
        super(NPAF, self).__init__()
        self.act_num = act_num
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)

    def forward(self, x):
        return self.bn(torch.nn.functional.conv2d(
            super(NPAF, self).forward(x),
            self.weight, padding=self.act_num, groups=self.dim))


class BaseNet(nn.Module):

    def __init__(self, band, classes, dropout=True):
        super(BaseNet, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.4)

        self.lwm = COS2M(band)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.act1 = NPAF(32)

        self.lwm2 = COS2M(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.se0 = SELayer(32)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.mask3 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False)
        #         self.mask3.weight.data = torch.ones(self.mask3.weight.size())
        self.bn3 = nn.BatchNorm2d(16)
        self.act2 = NPAF(16)
        self.se1 = SELayer(16)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(16, classes, bias=False)

    def forward(self, x):

        x1 = self.lwm(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.lwm2(x1)
        x2 = self.bn2(x2)
        x2 = self.act1(x2)
        x2 = self.se0(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.act2(x3)
        x3 = self.se1(x3)
        if self.use_dropout:
            x3 = self.dropout(x3)

        x3 = self.avgpool(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc1(x3)

        return x3


def CLOLN(dataset, patch_size):
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
    # Channel-Layer-Oriented Lightweight Spectral–Spatial Network for Hyperspectral Image Classification

    t = torch.randn(size=(1, 1, 103, 9, 9))
    print("input shape:", t.shape)
    net = CLOLN(dataset='pu', patch_size=9)
    print("output shape:", net(t).shape)


    from fvcore.nn import FlopCountAnalysis, flop_count_table

    net.eval()
    flops = FlopCountAnalysis(net, t)
    print(flop_count_table(flops))

    from thop import profile, clever_format

    flops, params = profile(net, inputs=(t,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)