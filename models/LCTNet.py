# -*- coding: utf-8 -*-
# Since there is no public code, it is implemented based on the content of the documentation.
"""
Z. Zhong, C. Liang, M. Yang, and D. Wang,
“LCTNet: Lightweight Convolution-Transformer Network for Hyperspectral Image Classification,”
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing,
vol. 19, pp. 7844–7857, 2026, doi: 10.1109/JSTARS.2026.3665704.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvBNAct2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Swish() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvBNAct3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding=0,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = Swish() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = Swish()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EntropyGate(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, entropy: torch.Tensor) -> torch.Tensor:
        ratio = torch.sigmoid(self.fc2(F.relu(self.fc1(entropy.unsqueeze(-1)))))
        return ratio.squeeze(-1) * 0.9 + 0.1


class SpectralExtractionDimReduction(nn.Module):
    def __init__(self, spectral_bands: int, out_channels: int = 64) -> None:
        super().__init__()
        self.conv3d_1 = ConvBNAct3d(1, 8, kernel_size=(7, 1, 1), stride=(3, 1, 1), padding=0)
        self.conv3d_2 = ConvBNAct3d(8, 8, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=0)

        reduced_bands = self._conv_out_dim(self._conv_out_dim(spectral_bands, 7, 3), 3, 2)
        if reduced_bands <= 0:
            raise ValueError(f"Invalid spectral bands {spectral_bands} for SEDR.")
        self.reduced_bands = reduced_bands
        self.conv1x1 = ConvBNAct2d(8 * reduced_bands, out_channels, kernel_size=1, padding=0, groups=8)

    @staticmethod
    def _conv_out_dim(length: int, kernel_size: int, stride: int, padding: int = 0) -> int:
        return (length + 2 * padding - kernel_size) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        b, c, l, h, w = x.shape
        x = x.reshape(b, c * l, h, w)
        x = self.conv1x1(x)
        return x


class ECA(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if t % 2 == 1 else t + 1
        k_size = max(3, k_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return torch.sigmoid(y)


class FrequencyGroupedChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.eca_x = ECA(channels)
        self.eca_y = ECA(channels)
        self.eca_z = ECA(channels)
        self.freq_fuse = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True, groups=4)
        gate_hidden = max(8, channels // 4)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, gate_hidden, kernel_size=1, bias=True),
            Swish(),
            nn.Conv2d(gate_hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft_map = torch.fft.fft2(x, dim=(-2, -1))
        energy = fft_map.abs().mean(dim=(-2, -1))
        _, idx = torch.sort(energy, dim=1, descending=True)
        topk = x.shape[1] // 2
        mask = torch.zeros_like(energy)
        mask.scatter_(1, idx[:, :topk], 1.0)
        mask = mask.unsqueeze(-1).unsqueeze(-1)

        y = x * mask
        z = x * (1.0 - mask)

        a_x = self.eca_x(x)
        a_y = self.eca_y(y)
        a_z = self.eca_z(z)

        a_f = self.freq_fuse(torch.cat([a_y, a_z], dim=1))
        g = self.gate(torch.cat([a_f, a_x], dim=1))
        w = g * a_f + (1.0 - g) * a_x
        return x * w


class ReceptiveFieldAttentionConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_rf_channels = channels * kernel_size * kernel_size

        self.spatial_conv = nn.Conv2d(
            channels,
            self.num_rf_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.attn_conv = nn.Conv2d(
            channels,
            self.num_rf_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=channels,
            bias=True,
        )
        self.out_conv = ConvBNAct2d(channels, channels, kernel_size=kernel_size, stride=kernel_size, padding=0, groups=4)
        self.act = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k = self.kernel_size

        frf = self.act(self.spatial_conv(x))
        frf = frf.view(b, c, k * k, h, w)

        arf = self.attn_conv(self.avg_pool(x)).view(b, c, k * k, h, w)
        arf = torch.softmax(arf, dim=2)

        weighted = frf * arf
        weighted = weighted.view(b, c, k, k, h, w)
        weighted = weighted.permute(0, 1, 4, 2, 5, 3).contiguous().view(b, c, h * k, w * k)
        return self.out_conv(weighted)


class SpatialSpectralFeatureMarking(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 32, rfa_kernel: int = 3) -> None:
        super().__init__()
        self.rfa = ReceptiveFieldAttentionConv(in_channels, kernel_size=rfa_kernel)
        self.fgca = FrequencyGroupedChannelAttention(in_channels)
        self.fuse = ConvBNAct2d(in_channels * 2, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rfa = self.rfa(x)
        x_fgca = self.fgca(x)
        x = torch.cat([x_rfa, x_fgca], dim=1)
        return self.fuse(x)


class DynamicSparseAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, attn_dropout: float = 0.0, proj_dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.pos_conv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.entropy_gate = EntropyGate(hidden_dim=16)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def _sparsify(self, attn: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        b, heads, n, _ = attn.shape
        sparse = torch.full_like(attn, float("-inf"))
        for bi in range(b):
            for hi in range(heads):
                keep = max(1, min(n, int(torch.ceil(ratio[bi, hi] * n).item())))
                topk_idx = attn[bi, hi].topk(k=keep, dim=-1).indices
                row_index = torch.arange(n, device=attn.device).unsqueeze(-1).expand(n, keep)
                sparse[bi, hi][row_index, topk_idx] = attn[bi, hi][row_index, topk_idx]
        return sparse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        qkv = self.qkv_proj(x)
        qkv = self.pos_conv(qkv)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.flatten(2).transpose(1, 2).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.flatten(2).transpose(1, 2).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.flatten(2).transpose(1, 2).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        prob = torch.softmax(attn, dim=-1)
        entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=-1).mean(dim=-1)
        ratio = self.entropy_gate(entropy)

        sparse_attn = self._sparsify(attn, ratio)
        sparse_attn = torch.softmax(sparse_attn, dim=-1)
        sparse_attn = self.attn_drop(sparse_attn)

        out = torch.matmul(sparse_attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        out = self.proj_drop(self.proj(out))
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out


class DSFormerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = DynamicSparseAttention(dim=dim, heads=heads, proj_dropout=dropout)
        self.norm2 = LayerNorm2d(dim)
        hidden_dim = max(dim, int(dim * mlp_ratio))
        self.mlp = MLP(dim=dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        y = self.norm2(x).permute(0, 2, 3, 1)
        y = self.mlp(y)
        y = y.permute(0, 3, 1, 2)
        x = x + y
        return x


class overall(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_chans: int,
        num_classes: int,
        depth: int = 2,
        mlp_dim: float = 1.0,
        embed_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depth = depth

        self.sedr = SpectralExtractionDimReduction(spectral_bands=in_chans, out_channels=64)
        self.ssfm = SpatialSpectralFeatureMarking(in_channels=64, out_channels=embed_dim, rfa_kernel=3)
        self.dsformer = nn.Sequential(
            *[
                DSFormerBlock(dim=embed_dim, heads=num_heads, mlp_ratio=mlp_dim, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape [B, 1, L, H, W], but got {tuple(x.shape)}")
        x = self.sedr(x)
        x = self.ssfm(x)
        x = self.dsformer(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x


def LCTNet(dataset,
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
    net = LCTNet(dataset='pu', patch_size=9)
    net.to(device)
    print("output shape:", net(t).shape)
    with torch.no_grad():
        sum = summary(net, input_size=(1, 1, t.shape[2], t.shape[-2], t.shape[-1]), verbose=0)
        print(sum)
        print(sum.trainable_params)
