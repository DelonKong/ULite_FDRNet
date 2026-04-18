# -*- coding: utf-8 -*-
# The public source code is implemented based on tensorflow,
# now it is reimplemented with pytorch.
"""
M. Q. Alkhatib, S. Kumar Roy, and A. Jamali,
“MixerSENet: A Lightweight Framework for Efficient Hyperspectral Image Classification,”
IEEE Geoscience and Remote Sensing Letters, vol. 22, pp. 1–5, 2025, doi: 10.1109/LGRS.2025.3616338.
"""


import torch
import torch.nn as nn
from torchinfo import summary


class ActivationBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(x)
        x = self.bn(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: int = 8) -> None:
        super().__init__()
        hidden_dim = max(1, channels // se_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w).view(b, c, 1, 1)
        return x * w


class MixerBlock(nn.Module):
    """
    轻量 Mixer block：
    1) 多尺度 depthwise 卷积做 token mixing
    2) pointwise 1x1 Conv 做 channel mixing
    """

    def __init__(self, filters: int, mlp_dim: float = 1.0) -> None:
        super().__init__()
        hidden_dim = max(8, int(filters * mlp_dim))

        self.dw3 = nn.Conv2d(
            filters, filters, kernel_size=3, padding=1, groups=filters, bias=True
        )
        self.dw5 = nn.Conv2d(
            filters, filters, kernel_size=5, padding=2, groups=filters, bias=True
        )
        self.dw7 = nn.Conv2d(
            filters, filters, kernel_size=7, padding=3, groups=filters, bias=True
        )

        self.pw1 = nn.Conv2d(filters, hidden_dim, kernel_size=1, bias=True)
        self.act_bn = ActivationBlock(hidden_dim)
        self.pw2 = nn.Conv2d(hidden_dim, filters, kernel_size=1, bias=True)
        self.bn_out = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        pos_emb1 = self.dw3(x)
        pos_emb2 = self.dw5(x)
        pos_emb3 = self.dw7(x)

        x = residual + pos_emb1 + pos_emb2 + pos_emb3
        x = self.pw1(x)
        x = self.act_bn(x)
        x = self.pw2(x)
        x = self.bn_out(x)
        x = x + residual
        return x


class overall(nn.Module):
    """
    保留 overall 作为模型具体实现。
    输入支持：
    - [B, 1, C, H, W]  -> 你的高光谱 patch 常见形式
    - [B, C, H, W]
    """

    def __init__(
        self,
        patch_size: int,
        in_chans: int,
        num_classes: int,
        depth: int = 1,
        mlp_dim: float = 1.0,
        filters: int = 64,
        se_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.filters = filters

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, filters, kernel_size=1, stride=1, padding=0, bias=True),
            ActivationBlock(filters),
        )

        self.blocks = nn.Sequential(
            *[MixerBlock(filters=filters, mlp_dim=mlp_dim) for _ in range(depth)]
        )

        self.se = SEBlock(filters, se_ratio=se_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            # [B, 1, C, H, W] -> [B, C, H, W]
            if x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                raise ValueError(
                    f"Expected input shape [B, 1, C, H, W], but got {tuple(x.shape)}"
                )
        elif x.ndim != 4:
            raise ValueError(
                f"Expected input to be 4D or 5D, but got shape {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_chans:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_chans}, got {x.shape[1]}"
            )

        x = self.stem(x)
        x = self.blocks(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x


def MixerSENet(dataset,
          patch_size=9,
          mlp_ratios=1,
          depths_te=1
          ):
    model = None
    if dataset == 'sa':
        model = overall(patch_size=patch_size,
                        in_chans=204,
                        num_classes=16,
                        depth=depths_te,
                        mlp_dim=mlp_ratios
                        )
    elif dataset == 'pu':
        model = overall(patch_size=patch_size,
                        in_chans=103,
                        num_classes=9,
                        depth=depths_te,
                        mlp_dim=mlp_ratios
                        )
    elif dataset == 'whulk':
        model = overall(patch_size=patch_size,
                        in_chans=270,
                        num_classes=9,
                        depth=depths_te,
                        mlp_dim=mlp_ratios
                        )
    elif dataset == 'hrl':
        model = overall(patch_size=patch_size,
                        in_chans=176,
                        num_classes=14,
                        depth=depths_te,
                        mlp_dim=mlp_ratios
                        )
    elif dataset == 'whuhh':
        model = overall(patch_size=patch_size,
                        in_chans=270,
                        num_classes=22,
                        depth=depths_te,
                        mlp_dim=mlp_ratios
                        )
    elif dataset == 'whuhc':
        model = overall(patch_size=patch_size,
                        in_chans=274,
                        num_classes=16,
                        depth=depths_te,
                        mlp_dim=mlp_ratios
                        )
    elif dataset == 'IP':
        model = overall(patch_size=patch_size,
                        in_chans=200,
                        num_classes=16,
                        depth=depths_te,
                        mlp_dim=mlp_ratios
                        )
    elif dataset == 'BS':
        model = overall(patch_size=patch_size,
                        in_chans=145,
                        num_classes=14,
                        )
    elif dataset == 'HsU':
        model = overall(patch_size=patch_size,
                        in_chans=144,
                        num_classes=15,
                        )
    elif dataset == 'KSC':
        model = overall(patch_size=patch_size,
                        in_chans=176,
                        num_classes=13,
                        )
    elif dataset == 'pc':
        model = overall(patch_size=patch_size,
                        in_chans=102,
                        num_classes=9,
                        )
    return model

if __name__ == "__main__":

    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    t = torch.randn(size=(64, 1, 103, 9, 9)).to(device)
    net = MixerSENet(dataset='pu', patch_size=9)
    net.to(device)
    print("output shape:", net(t).shape)
    with torch.no_grad():
        sum = summary(net, input_size=(1, 1, t.shape[2], t.shape[-2], t.shape[-1]), verbose=0)
        print(sum)
        print(sum.trainable_params)
