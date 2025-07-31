import torch.nn as nn
import os
import numpy as np
import torch
import shutil
from torch.autograd import Variable
from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'Mcs_sepConv_3x3',
    'Mcs_sepConv_5x5',
    'Mcs_sepConv_7x7',
]
#---houston
HSI=Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('Mcs_sepConv_3x3', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('Mcs_sepConv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('Mcs_sepConv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 3), ('Mcs_sepConv_5x5', 1), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'Mcs_sepConv_3x3': lambda C, stride, affine: Mcs_sepConv(C, C, 3, stride, 1, affine=affine),
    'Mcs_sepConv_5x5': lambda C, stride, affine: Mcs_sepConv1(C, C, 5, stride, 2, affine=affine),
    'Mcs_sepConv_7x7': lambda C, stride, affine: Mcs_sepConv2(C, C, 7, stride, 3, affine=affine),
}


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Mcs_sepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Mcs_sepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0), groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding), groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),

        )
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(C_in)
        self.ca1 = ChannelAttention(C_in)
        self.sa1 = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.op(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, :, :])], dim=1)
        out = self.bn(out)
        return out


class Mcs_sepConv1(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Mcs_sepConv1, self).__init__()
        self.op = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), groups=C_in, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),

            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),

        )
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(C_in)
        self.ca1 = ChannelAttention(C_in)
        self.sa1 = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.op(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        return x


class Mcs_sepConv2(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Mcs_sepConv2, self).__init__()
        self.op = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), groups=C_in, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),

            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=C_in, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),

            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(C_in)
        self.ca1 = ChannelAttention(C_in)
        self.sa1 = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.op(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkHSI(nn.Module):

    def __init__(self, nBand, num_classes, genotype, layers=3, auxiliary=False, C=16):
        super(NetworkHSI, self).__init__()
        self.drop_path_prob = 0
        self._layers = layers
        self.auxiliary = auxiliary
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(nBand, 10, 1, 1, 0, bias=False),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        input = input.squeeze(1)

        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits


def LMSS_NAS(dataset, patch_size=7):

    model = None
    genotype = HSI
    if dataset == 'sa':
        model = NetworkHSI(
                        nBand=204,
                        num_classes=16,
                        genotype=genotype
                        )
    elif dataset == 'pu':
        model = NetworkHSI(
                        nBand=103,
                        num_classes=9,
            genotype=genotype
                        )
    elif dataset == 'whulk':
        model = NetworkHSI(
                        nBand=270,
                        num_classes=9,
            genotype=genotype
        )
    elif dataset == 'hrl':
        model = NetworkHSI(
                        nBand=176,
                        num_classes=14,
            genotype=genotype
                        )
    elif dataset == 'whuhh':
        model = NetworkHSI(
                        nBand=270,
                        num_classes=22,
            genotype=genotype
                        )
    elif dataset == 'whuhc':
        model = NetworkHSI(
                        nBand=274,
                        num_classes=16,
            genotype=genotype
                        )
    elif dataset == 'IP':
        model = NetworkHSI(
                        nBand=200,
                        num_classes=16,
            genotype=genotype
                        )
    elif dataset == 'BS':
        model = NetworkHSI(
                        nBand=145,
                        num_classes=14,
                        )
    elif dataset == 'HsU':
        model = NetworkHSI(
                        nBand=144,
                        num_classes=15,
            genotype=genotype
                        )
    elif dataset == 'KSC':
        model = NetworkHSI(
                        nBand=176,
                        num_classes=13,
            genotype=genotype
                        )
    elif dataset == 'pc':
        model = NetworkHSI(
                        nBand=102,
                        num_classes=9,
            genotype=genotype
                        )
    return model

if __name__ == "__main__":
    # Lightweight Multiscale Neural Architecture Search With Spectral–Spatial Attention for Hyperspectral Image Classification

    device = torch.device("cuda:{}".format(0))

    t = torch.randn(size=(64, 1, 103, 7, 7)).to(device)
    net = LMSS_NAS(dataset='pu', patch_size=7)

    net.to(device)
    print("output shape:", net(t).shape)

    from fvcore.nn import FlopCountAnalysis, flop_count_table

    net.eval()
    flops = FlopCountAnalysis(net, t)
    print(flop_count_table(flops))

    from thop import profile, clever_format

    flops, params = profile(net, inputs=(t,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)