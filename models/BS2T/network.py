"""
[1] R. Song, Y. Feng, W. Cheng, Z. Mu, and X. Wang, “BS2T: Bottleneck Spatial–Spectral Transformer for Hyperspectral Image Classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–17, 2022, doi: 10.1109/TGRS.2022.3185640.

"""


import torch
from torch import nn
import math

import sys

from models.BS2T.global_module.TransformerBlock import BottleStack
from models.BS2T.global_module.activation import mish

sys.path.append('/')


class overall(nn.Module):
    def __init__(self, band, classes, patch_size):
        super(overall, self).__init__()

        # spectral branch
        self.name = 'BS2T'
        self.transform23 = BottleStack(dim=36, fmap_size=patch_size, dim_out=12, proj_factor=2, downsample=False, heads=3, dim_head=2, rel_pos_emb=False, activation=nn.ReLU)
        self.transform11 = BottleStack(dim=60, fmap_size=patch_size, dim_out=60, proj_factor=5, downsample=False, heads=3, dim_head=4, rel_pos_emb=False, activation=nn.ReLU)
        self.transform24 = BottleStack(dim=48, fmap_size=patch_size, dim_out=12, proj_factor=2, downsample=False, heads=3, dim_head=2, rel_pos_emb=False, activation=nn.ReLU)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=200,
            nhead=2,
            dim_feedforward=128,
            dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new()
            # swish()
            mish()
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        # kernel_3d = math.floor((band - 6) / 2)  # 其他数据集
        kernel_3d = math.ceil((band - 6) / 2)  # pu数据集
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))  # kernel size随数据变化

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        self.conv25 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                      kernel_size=(3, 3, 2), stride=(1, 1, 1)),
            nn.Sigmoid()
        )

        self.batch_norm_spectral = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new(),
            # swish(),
            mish(),
            nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new(),
            # swish(),
            mish(),
            nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(120, classes)  # ,
            # nn.Softmax()
        )


        # fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # [BS, s, c, p, p] -> [BS, s, p, p, c]
        X = X.permute(0, 1, 3, 4, 2)
        # spectral
        x11 = self.conv11(X)
        # print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)

        x13 = torch.cat((x11, x12), dim=1)
        # print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        # print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        # print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        x10 = self.transform11(x16)
        # x1 = x16
        x1 = torch.mul(x10, x16)


        # x1 = X.permute(2, 1, 0)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.transform23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.transform24(x24)
        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        x20 = self.transform11(x25)
        # x2 = x25
        x2 = torch.mul(x20, x25)


        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.batch_norm_spatial(x2)
        x2 = self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)
        x_pre = torch.cat((x1, x2), dim=1)

        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output

def BS2T(dataset, patch_size):
    model = None
    if dataset == 'sa':
        model = overall(band=204, classes=16, patch_size=patch_size)

    elif dataset == 'pu':
        model = overall(band=103, classes=9, patch_size=patch_size)

    elif dataset == 'whulk':
        model = overall(band=270, classes=9, patch_size=patch_size)

    elif dataset == 'hrl':
        model = overall(band=176, classes=14, patch_size=patch_size)

    elif dataset == 'IP':
        model = overall(band=200, classes=16, patch_size=patch_size)

    elif dataset == 'whuhc':
        model = overall(band=274, classes=16, patch_size=patch_size)

    elif dataset == 'BS':
        model = overall(band=145, classes=14, patch_size=patch_size)
    elif dataset == 'HsU':
        model = overall(band=144, classes=15, patch_size=patch_size)
    elif dataset == 'KSC':
        model = overall(band=176, classes=13, patch_size=patch_size)
    elif dataset == 'pc':
        model = overall(band=102, classes=9, patch_size=patch_size)
    elif dataset == 'whuhh':
        model = overall(band=270, classes=22, patch_size=patch_size)

    return model


if __name__ == "__main__":
    # t = torch.randn(size=(3, 1, 11, 11, 204))
    t = torch.randn(size=(3, 1, 103, 11, 11))
    print("input shape:", t.shape)
    net = BS2T(dataset='pu', patch_size=11)
    print("output shape:", net(t).shape)
