# -*- coding: utf-8 -*-
import warnings
from functools import partial
import torch
import torch.nn as nn
from torchinfo import summary

"""
D. Kong, S. Zhang, X. Yu, Y. Lu, S. Yang, and J. Zhang, 
“Ultralightweight progressive feature disentanglement and recomposition network for hyperspectral image classification,” 
Neural Networks, vol. 203, p. 109200, Nov. 2026, doi: 10.1016/j.neunet.2026.109200.
https://github.com/DelonKong/ULite_FDRNet
"""


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features,
                 out_features=None,  # default: in_features
                 hidden_features=None,
                 drop=0.
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension hidden_features and out_features
        output: (B, N, C)
        """
        x = self.fc1(x)  # (B, N, C) -> (B, N, hidden_features)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # (B, N, hidden_features) -> (B, N, out_features)
        x = self.drop(x)
        return x


class FFN3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = FDRConv3dBN(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = FDRConv3dBN(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Conv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, act_layer=None):
        super(Conv2dBN, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act_layer = None
        if act_layer is not None:
            self.act_layer = act_layer()

        # nn.init.constant_(self.bn.weight, bn_weight_init)
        # nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act_layer is not None:
            x = self.act_layer(x)
        return x


class Conv3dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                 dilation=1, groups=1, act_layer=None):
        super(Conv3dBN, self).__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )

        self.bn = nn.BatchNorm3d(out_channels)
        self.act_layer = None
        if act_layer is not None:
            self.act_layer = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act_layer is not None:
            x = self.act_layer(x)
        return x


class FDRConv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, kernel_size=1, stride=1, act_layer=None):
        super().__init__()

        assert out_channels % (ratio * (ratio - 1)) == 0, "out_channels must equal to k*ratio*(ratio-1)"

        self.ratio = ratio
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            act_layer() if act_layer else nn.Identity(),
        )

        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            act_layer() if act_layer else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)
        return torch.cat([x1, x2], dim=1)


class FDRConv3dBN(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, kernel1=1, kernel2=None,
                 stride=1, padding1=0, padding2=None, act_layer=None, Norm=None):
        super().__init__()

        assert out_channels % (ratio * (ratio - 1)) == 0, "out_channels must equal to k*ratio*(ratio-1)"

        kernel2 = kernel2 or kernel1
        padding2 = padding2 or padding1

        self.ratio = ratio
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel1, stride, padding1, bias=False),
            Norm(init_channels) if Norm else nn.Identity(),
            act_layer() if act_layer else nn.Identity(),
        )

        self.cheap_conv = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, kernel2, stride, padding2, groups=init_channels, bias=False),
            Norm(new_channels) if Norm else nn.Identity(),
            act_layer() if act_layer else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)
        return torch.cat([x1, x2], dim=1)


class ScaleFDR3D(nn.Module):
    def __init__(self, out_channels_3d=3, num_layer=1, ratio_3d=2, act_layer=nn.GELU, Scale=3):
        super(ScaleFDR3D, self).__init__()

        assert out_channels_3d >= 4, "out_channels_3d must >= 4"

        # self.conv3d_1 = nn.Conv3d(1, out_channels=out_channels_3d, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_1 = nn.Sequential(
            FDRConv3dBN(1, out_channels=out_channels_3d, kernel1=1, kernel2=3, padding1=0, padding2=1),
            FDRConv3dBN(out_channels_3d, out_channels=out_channels_3d, kernel1=1, kernel2=3, padding1=0, padding2=1)
        )

        self.multi = nn.ModuleList([
            nn.Sequential(
                MultiScale3D(out_channels_3d, out_channels_3d, kernel_size_base=Scale, axial_size=(False, True, True),
                             act_layer=act_layer, branch_ratio=ratio_3d),
                MultiScale3D(out_channels_3d, out_channels_3d, kernel_size_base=Scale, axial_size=(True, False, False),
                             act_layer=act_layer, branch_ratio=ratio_3d)
                # ==========================================================
                # MultiScale3D(out_channels_3d, out_channels_3d, kernel_size_base=5, axial_size=(True, False, False),
                #              act_layer=act_layer, branch_ratio=ratio_3d),
                # MultiScale3D(out_channels_3d, out_channels_3d, kernel_size_base=5, axial_size=(False, True, False),
                #              act_layer=act_layer, branch_ratio=ratio_3d),
                # MultiScale3D(out_channels_3d, out_channels_3d, kernel_size_base=5, axial_size=(False, False, True),
                #              act_layer=act_layer, branch_ratio=ratio_3d),
                # ==========================================================
                # MultiScale3D(out_channels_3d, out_channels_3d, kernel_size_base=5, axial_size=(True, True, True),
                #              act_layer=act_layer, branch_ratio=ratio_3d),
            )
            for _ in range(num_layer)])

    def forward(self, x):
        # x ->: [batchsize, channel, bands, p, p]
        x = self.conv3d_1(x)
        res = x
        for multi in self.multi:
            x = multi(x)
        x = x + res
        return x


class MultiScale3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size_base=3, axial_size=(False, False, True),
                 act_layer=nn.GELU, branch_ratio=2):
        super(MultiScale3D, self).__init__()

        out_channels = out_channels or in_channels
        assert out_channels % (branch_ratio * (branch_ratio - 1)) == 0, "out_channels must equal to k*ratio*(ratio-1)"

        init_channels = out_channels // branch_ratio
        multi_channels_1 = (out_channels - init_channels) // 2
        multi_channels_2 = init_channels - multi_channels_1
        self.split_indexes = (multi_channels_1, multi_channels_2)
        init_channels = out_channels - (multi_channels_1 + multi_channels_2)
        self.axial_size = axial_size
        self.active_axes = [i for i, val in enumerate(axial_size) if val]

        if not self.active_axes:
            raise ValueError("At least one axis must be active in axial_size.")

        def get_kernel_and_padding(kernel_size_base, active_axes):
            kernel = [1, 1, 1]
            padding = [0, 0, 0]
            for axis in active_axes:
                kernel[axis] = kernel_size_base
                padding[axis] = (kernel_size_base - 1) // 2
            return tuple(kernel), tuple(padding)

        kernel_conv1, padding_conv1 = get_kernel_and_padding(kernel_size_base, self.active_axes)
        if kernel_size_base == 1:
            kernel_conv2, padding_conv2 = get_kernel_and_padding(kernel_size_base, self.active_axes)
        else:
            kernel_conv2, padding_conv2 = get_kernel_and_padding(kernel_size_base - 2, self.active_axes)

        self.conv_rest = nn.Conv3d(in_channels, init_channels, kernel_size=(1, 1, 1))
        self.conv1 = nn.Conv3d(multi_channels_1, multi_channels_1, kernel_size=kernel_conv1, padding=padding_conv1,
                               groups=multi_channels_1)
        self.conv2 = nn.Conv3d(multi_channels_2, multi_channels_2, kernel_size=kernel_conv2, padding=padding_conv2,
                               groups=multi_channels_2)

    def forward(self, x):
        x_rest = self.conv_rest(x)

        x1, x2 = torch.split(x_rest, self.split_indexes, dim=1)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        out = torch.cat([x_rest, x1, x2], dim=1)
        return out


class ECA(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)  # BS,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # BS,1,c
        y = self.conv(y)  # BS,1,c
        y = self.sigmoid(y)  # BS,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # BS,c,1,1
        return y


class ECA3D(nn.Module):
    def __init__(self, kernel_size=3):
        super(ECA3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size,
                                padding=(kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [B, channel_num, C_per_group, H, W].
        """
        B, C, D, H, W = x.size()

        y = self.avg_pool(x)  # [B, C, 1, 1, 1]
        y = y.view(B, C, -1)  # [B, C, 1]

        y = y.permute(0, 2, 1)  # [B, 1, C]
        y = self.conv1d(y)  # [B, 1, C]
        y = y.permute(0, 2, 1)  # [B, C, 1]

        y = self.sigmoid(y)  # [B, C, 1]

        y = y.view(B, C, 1, 1, 1)  # [B, C, 1, 1, 1]

        return y


class TriSFDR(nn.Module):
    def __init__(self, dim, out_channels_3d, num_heads=8, qk_scale=None,
                 act_layer=nn.GELU):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5

        self.out = FDRConv3dBN(num_heads, out_channels_3d, act_layer=act_layer, Norm=nn.BatchNorm3d)

        self.qkvc = FDRConv3dBN(out_channels_3d, num_heads * 4)
        self.q_norm = nn.BatchNorm3d(num_heads)
        self.k_norm = nn.BatchNorm3d(num_heads)

        self.proj = FDRConv3dBN(num_heads, num_heads, kernel2=(3, 1, 1), padding2=(1, 0, 0))

        self.proj_encode_row = FDRConv3dBN(num_heads, num_heads, kernel2=(1, 3, 1), padding2=(0, 1, 0))
        self.proj_encode_column = FDRConv3dBN(num_heads, num_heads, kernel2=(1, 1, 3), padding2=(0, 0, 1))

        self.sigmoid = nn.Sigmoid()
        self.eca3d = ECA3D(kernel_size=3)
        self.eca = ECA(kernel_size=3)

    def forward(self, x):
        # x: -> [bs, channels_3d, C, H, W]
        B, G, C, H, W = x.shape

        qkvc = self.qkvc(x)  # [B, num_heads*4, C, H, W]
        qkvc = qkvc.reshape(B, self.num_heads, 4, C, H, W)  # [B, num_heads, 4, C, H, W]
        qkvc = qkvc.permute(2, 0, 1, 3, 4, 5)  # [4, B, num_heads, C, H, W]

        q, k, v, x_cin = qkvc.unbind(0)  # [B, num_heads, C, H, W]
        q, k = self.q_norm(q), self.k_norm(k)

        # Vertical axial attention
        # # squeeze row
        qrow = q.mean(-1).permute(0, 1, 3, 2)  # B, G, H, C
        krow = k.mean(-1)  # B, G, C, H
        vrow = v.mean(-1).permute(0, 1, 3, 2)  # B, G, H, C
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)  # B, G, H, H
        xx_row = torch.matmul(attn_row, vrow)  # B, G, H, C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).unsqueeze(-1))  # B, G, C, H, 1

        # Horizontal axial attention
        # # squeeze column
        qcolumn = q.mean(-2).permute(0, 1, 3, 2)  # B, G, W, C
        kcolumn = k.mean(-2)  # B, G, C, W
        vcolumn = v.mean(-2).permute(0, 1, 3, 2)  # B, G, W, C
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)  # B, G, W, W
        xx_column = torch.matmul(attn_column, vcolumn)  # B, G, W, C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).unsqueeze(-2))  # B, G, C, 1, W

        # channel axial attention
        xc = self.eca3d(x_cin)
        out_c = xc * x_cin  # B, G, C, H, W

        out_c = out_c.view(B, -1, H, W)  # B, G*C, H, W
        xc = self.eca(out_c)
        out_c = xc * out_c  # B, G*C, H, W
        out_c = out_c.view(B, -1, C, H, W)

        xx = xx_row.add(xx_column)  # B, G, C, H, W
        out_wh = self.proj(xx)  # B, G, C, H, W
        out = self.sigmoid(out_wh) * out_c
        out = self.out(out)  # [bs, channels_3d, C, H, W]
        return out, None


class Block(nn.Module):
    def __init__(self, dim, out_channels_3d, num_heads, mlp_ratio=1., act_layer=nn.GELU,
                 qk_scale=None, drop=0., drop_path=0., init_values=0, dataset="sa"):
        super().__init__()
        self.attn = TriSFDR(
            dim=dim, out_channels_3d=out_channels_3d, num_heads=num_heads, qk_scale=qk_scale,
            act_layer=act_layer)
        mlp_hidden_dim = int(out_channels_3d * mlp_ratio)
        if dataset == "pu":
            self.mlp = FFN3D(in_features=out_channels_3d, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp = None

    def forward(self, x, return_attention=False):
        y, attn = self.attn(x)
        if return_attention:
            return attn
        x = x + y
        if self.mlp is not None:
            x = x + self.mlp(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, patch_size=7, embed_dim=64, input_channels=13, out_channels_3d=3,
                 depth=1, num_heads=4, mlp_ratio=2.,
                 qk_scale=None, drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-5),
                 init_values=0, return_all_tokens=True, use_mean_pooling=False,
                 act_layer=nn.GELU, dataset="sa"):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens
        self.num_patches = patch_size * patch_size

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, out_channels_3d=out_channels_3d, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qk_scale=qk_scale, drop=drop_rate, drop_path=0,
                init_values=init_values, act_layer=act_layer, dataset=dataset)
            for _ in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

    def forward(self, x, return_all_tokens=None):
        for blk in self.blocks:
            x = blk(x)

        # x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return x  # [BS, n+1, d]
        return x[:, 0]  # [BS, 1, d]


class overall(nn.Module):
    def __init__(self, patch_size,  # the size of input img
                 in_chans,  # the bands number of input img
                 num_classes=16,
                 out_channels_3d=4,  # the out_channels of 3Dconv in ScaleFDR3D
                 num_heads=4,
                 num_extras=1,  # the num of MultiScale3D in ScaleFDR3D
                 num_stages=1,  # the num_stages of Vit
                 dim=[40, 40, 40],  # the dim in each stage of Vit
                 depths_te=[1, 1, 1],  # the block depth in each stage of Vit
                 mlp_ratios=1,  # mlp_ratios at transformer mlp
                 ratio_2d=2,  # the ratio in FDRConv2dBN
                 ratio_3d=2,  # the ratio in MultiScale3D
                 act_layer=nn.GELU,
                 Scale=5,  # the scale in MultiScale3D
                 dataset="sa"
                 ):
        super(overall, self).__init__()

        self.num_stages = num_stages

        for i in range(num_stages):
            if i == 0:
                input_channels = in_chans
            else:
                input_channels = dim[i - 1]
            embed_dim = dim[i]
            conv2d = FDRConv2dBN(input_channels, embed_dim, ratio=ratio_2d, kernel_size=1, stride=1,
                                   act_layer=act_layer)
            setattr(self, f"conv2d{i + 1}", conv2d)

            extractor = ScaleFDR3D(out_channels_3d=out_channels_3d, num_layer=num_extras,
                                           act_layer=act_layer, ratio_3d=ratio_3d, Scale=Scale)
            setattr(self, f"extractor{i + 1}", extractor)

            te = VisionTransformer(patch_size=patch_size, embed_dim=embed_dim, input_channels=input_channels,
                                   out_channels_3d=out_channels_3d, depth=depths_te[i],
                                   num_heads=num_heads, mlp_ratio=mlp_ratios, return_all_tokens=True,
                                   use_mean_pooling=False, act_layer=act_layer, dataset=dataset)

            setattr(self, f"te{i + 1}", te)

            if self.num_stages > 1 and i >= 0 and i + 1 < self.num_stages:
                conv = nn.Conv3d(out_channels_3d, 1, 1)
                setattr(self, f"conv{i + 1}", conv)

        # mlp_hidden_dim = int(embed_dim[self.num_stages - 1] * mlp_ratios)
        if dataset == "pu":
            self.ffn = FFN3D(in_features=out_channels_3d, hidden_features=out_channels_3d * mlp_ratios, act_layer=act_layer)
        else:
            self.ffn = None

        self.ffn2 = nn.Conv3d(out_channels_3d, 1, kernel_size=(1, 1, 1))
        self.head = nn.Linear(embed_dim, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        for i in range(self.num_stages):
            conv2d = getattr(self, f"conv2d{i + 1}")
            x = conv2d(x.squeeze(1)).unsqueeze(1)

            extractor = getattr(self, f"extractor{i + 1}")
            x = extractor(x)

            te = getattr(self, f"te{i + 1}")
            x = te(x)

            if self.num_stages > 1 and i >= 0 and i + 1 < self.num_stages:
                conv = getattr(self, f"conv{i + 1}")
                x = conv(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.ffn is not None:
            x = self.ffn(x)  # [bs, channels, C, H, W]
        # x = self.ffn2(x).squeeze(2)
        x = self.ffn2(x).squeeze(1)
        x = self.avgpool(x).view(-1, x.shape[1])

        x = self.head(x)

        return x


def ULite_FDRNet(dataset,
              in_chans=103,
              patch_size=9,
              out_channels_3d=4,
              num_heads=2,
              embed_dim=40,
              mlp_ratios=3,
              act_layer=nn.GELU,
              num_extras=1,
              depths_te=1,
              ratio_2d=2,
              ratio_3d=2,
              Scale=5,
              ):
    dim = [embed_dim, embed_dim // 2, embed_dim // 4]
    depths_te = [depths_te, depths_te // 2, depths_te // 4]

    num_classes = 0

    if dataset == 'sa':
        num_classes = 16
    elif dataset == 'pu':
        num_classes = 9
    elif dataset == 'whulk':
        num_classes = 9
    elif dataset == 'whuhh':
        num_classes = 22
    elif dataset == 'whuhc':
        num_classes = 16
    elif dataset == 'hrl':
        num_classes = 14
    elif dataset == 'IP':
        num_classes = 16
    elif dataset == 'BS':
        num_classes = 14
    elif dataset == 'HsU':
        num_classes = 15
    elif dataset == 'KSC':
        num_classes = 13
    elif dataset == 'pc':
        num_classes = 9
    else:
        warnings.warn(f"Unsupported dataset: {dataset}. Returning None model.")
        return None

    assert num_classes != 0, "NUM CLASSES ERROR: num_classes != 0"

    model = overall(patch_size=patch_size,
                    # in_chans=204,
                    in_chans=in_chans,
                    num_classes=num_classes,
                    out_channels_3d=out_channels_3d,
                    num_heads=num_heads,
                    dim=dim,
                    mlp_ratios=mlp_ratios,
                    act_layer=act_layer,
                    num_extras=num_extras,
                    depths_te=depths_te,
                    ratio_2d=ratio_2d,
                    ratio_3d=ratio_3d,
                    Scale=Scale,
                    dataset=dataset
                    )
    return model


if __name__ == "__main__":
    device = torch.device("cuda:{}".format(0))
    dataset_name = "whulk"
    # pca=40
    if dataset_name == "pu":
        embed_dim = 40
        out_channels_3d = 4
        num_heads = 4
    elif dataset_name == "sa":
        embed_dim = 40
        out_channels_3d = 4
        num_heads = 2
    elif dataset_name == "whuhh":
        embed_dim = 64
        out_channels_3d = 4
        num_heads = 2
    else:
        embed_dim = 16
        out_channels_3d = 4
        num_heads = 2

    t = torch.randn(size=(1, 1, 40, 9, 9)).to(device)
    net = ULite_FDRNet(dataset=dataset_name, in_chans=t.shape[2],
                    patch_size=t.shape[-1],
                    out_channels_3d=out_channels_3d,
                    num_heads=num_heads,
                    num_extras=1,
                    depths_te=1,
                    embed_dim=embed_dim,
                    mlp_ratios=3,
                    act_layer=nn.GELU,
                    Scale=5
                    )
    net.to(device)
    print("output shape:", net(t).shape)
    print(net)

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    net.eval()
    flops = FlopCountAnalysis(net, t)
    print(flop_count_table(flops))

    with torch.no_grad():
        sum = summary(net, input_size=(1, 1, t.shape[2], t.shape[-2], t.shape[-1]), verbose=0)
        print(sum)
        print(sum.trainable_params)