import torch.nn.functional as F
import torch.nn as nn
import torch

expand = 1


class AD(nn.Module):
    def __init__(self, channel=1, ksize=3, expand=expand):
        super(AD, self).__init__()
        self.channel = channel
        self.ksize = ksize
        self.conv = nn.Conv2d(channel, 9 * expand * channel, 3, 1, padding=1, bias=False, groups=channel)
        self.init_w()

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(0.)
                m.weight.data[:, :, 1, 1].fill_(1.)

    def forward(self, x):
        weight = self.conv.weight.data
        o, i, wh, ww = weight.shape
        weight = weight.view(o, i, wh * ww)
        self.conv.weight.data = F.gumbel_softmax(weight, hard=True).view(o, i, wh, ww)
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.view(b, self.channel, -1, h, w)
        x = torch.max(x, 2)[0]
        return x


class AE(nn.Module):
    def __init__(self, channel=1, ksize=3, expand=expand):
        super(AE, self).__init__()
        self.channel = channel
        self.ksize = ksize
        self.conv = nn.Conv2d(channel, 9 * expand * channel, 3, 1, padding=1, bias=False, groups=channel)
        self.init_w()

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(0.)
                m.weight.data[:, :, 1, 1].fill_(1.)

    def forward(self, x):
        weight = self.conv.weight.data
        o, i, wh, ww = weight.shape
        weight = weight.view(o, i, wh * ww)
        self.conv.weight.data = F.gumbel_softmax(weight, hard=True).view(o, i, wh, ww)
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.view(b, self.channel, -1, h, w)
        x = torch.min(x, 2)[0]
        return x


class AEMP(nn.Module):
    def __init__(self, classes=16, band=4, order=5):
        super(AEMP, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(band, 16, 7, 1, 7 // 2, bias=False),
                                  nn.BatchNorm2d(16),
                                  nn.LeakyReLU())
        channel = 16

        self.AOs = nn.ModuleList([AE(channel, 3) for i in range(order)])
        self.ACs = nn.ModuleList([AD(channel, 3) for i in range(order)])

        self.fcs = nn.Sequential(
            nn.BatchNorm2d(channel * 2 * order + channel + band),
            nn.Dropout(0.5),
            nn.Conv2d(channel * 2 * order + channel + band, classes, 1, 1, 0),
        )

    def maxpool(self, x):
        return F.max_pool2d(x, 3, 1, 1)

    def minpool(self, x):
        return -F.max_pool2d(-x, 3, 1, 1)

    def forward(self, inp):
        inp = inp.squeeze(1)
        x = self.conv(inp)
        # x = inp
        aos = []
        temp = x
        for ao in self.AOs:
            temp = ao(temp)
            aos.append(temp)

        acs = []
        temp = x
        for ac in self.ACs:
            temp = ac(temp)
            acs.append(temp)

        feat = torch.cat(aos + [x] + acs + [inp], 1)
        logist = self.fcs(feat)
        return logist


class AEMPLoss(nn.Module):
    def __init__(self, ):
        super(AEMPLoss, self).__init__()

    def forward(self, x, label, trainMask):
        label = torch.where(trainMask == 1, label, -1 * torch.ones_like(label).to(x.device))
        return F.cross_entropy(x, label, ignore_index=-1)


def AMF(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = AEMP(band=204, classes=16)
    elif dataset == 'pu':
        model = AEMP(band=103, classes=9)
    elif dataset == 'whulk':
        model = AEMP(band=270, classes=9)
    elif dataset == 'whuhh':
        model = AEMP(band=270, classes=22)
    elif dataset == 'hrl':
        model = AEMP(band=176, classes=14)
    elif dataset == 'IP':
        model = AEMP(band=200, classes=16)
    elif dataset == 'whuhc':
        model = AEMP(band=274, classes=16)
    elif dataset == 'BS':
        model = AEMP(band=145, classes=14)
    elif dataset == 'HsU':
        model = AEMP(band=144, classes=15)
    elif dataset == 'KSC':
        model = AEMP(band=176, classes=13)
    elif dataset == 'pc':
        model = AEMP(band=102, classes=9)

    return model


if __name__ == '__main__':
    # Adaptive Morphology Filter: A Lightweight Module for Deep Hyperspectral Image Classification

    t = torch.randn(size=(64, 1, 103, 7, 7))
    print("input shape:", t.shape)
    net = AMF(dataset='pu', patch_size=7)
    print("output shape:", net(t).shape)


    from fvcore.nn import FlopCountAnalysis, flop_count_table

    net.eval()
    flops = FlopCountAnalysis(net, t)
    print(flop_count_table(flops))

    from thop import profile, clever_format

    flops, params = profile(net, inputs=(t,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
    
    
