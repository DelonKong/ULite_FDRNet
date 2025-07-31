import math

import einops
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.attention import TripletAttention, SAA_Attention


class MulitiScaleAttentionExtractor(nn.Module):
    def __init__(self, input_channels,  # the bands number of img
                 out_channels_3d=3,
                 out_channels_2d=256,
                 stride=[3, 1, 1]
                 ):
        super(MulitiScaleAttentionExtractor, self).__init__()

        self.conv3d_1 = nn.Conv3d(1, out_channels=out_channels_3d, kernel_size=3, stride=stride, padding=1)
        bands = (input_channels - 1) // stride[0] + 1
        self.conv3d_2 = nn.Conv3d(out_channels_3d, out_channels=out_channels_3d, kernel_size=3, stride=stride,
                                  padding=1)
        bands = (bands - 1) // stride[0] + 1
        self.conv2d = nn.Conv2d(out_channels_3d * bands, out_channels=out_channels_2d, kernel_size=1)
        # self.conv2d = nn.Conv2d(out_channels_3d * bands, out_channels=out_channels_2d, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels_2d)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x ->: [batchsize, channel, bands, p, p]
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        # reshape
        x = rearrange(x, 'BS c b w h -> BS (c b) w h')
        # 2Dconv -> BN -> ReLU
        x = self.relu(self.bn(self.conv2d(x)))
        return x


class SpatialExtractor(nn.Module):
    def __init__(self, input_channels=105):
        super(SpatialExtractor, self).__init__()

        c = input_channels

        self.conv1x1_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.conv7x7 = nn.Conv2d(c, out_channels=c, kernel_size=7, padding=3, groups=input_channels)
        self.conv5x5 = nn.Conv2d(c, out_channels=c, kernel_size=5, padding=2, groups=input_channels)
        self.conv3x3 = nn.Conv2d(c, out_channels=c, kernel_size=3, padding=1, groups=input_channels)
        self.bn1 = nn.BatchNorm2d(c * 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv2d(c * 3, out_channels=c, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # 分支1：AvgPool -> Conv1x1 -> Sigmoid
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x1 = torch.cat([max_out, avg_out], dim=1)
        x1 = self.conv1x1_1(x1)
        x1 = self.sigmoid(x1)

        # 分支2：空间维度多尺度卷积模块
        x2_1 = self.conv7x7(x)
        x2_2 = self.conv5x5(x)
        x2_3 = self.conv3x3(x)
        x2 = torch.cat((x2_1, x2_2, x2_3), dim=1)
        x2 = self.bn1(x2)
        x2 = self.relu1(x2)
        x2 = self.conv1x1_2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        # 两个分支的结果进行逐元素相乘运算
        x = (x1 * x2) + x

        return x


class SpectralExtractor(nn.Module):
    def __init__(self, input_channels=105, ratio=16):
        super(SpectralExtractor, self).__init__()

        c = input_channels

        self.spatial_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1x1_1 = nn.Conv2d(c, out_channels=c // ratio, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv2d(c // ratio, out_channels=c, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.conv7group = nn.Conv2d(c, out_channels=c, kernel_size=1, groups=8)
        self.conv5group = nn.Conv2d(c, out_channels=c, kernel_size=1, groups=4)
        self.conv3group = nn.Conv2d(c, out_channels=c, kernel_size=1, groups=2)
        self.bn1 = nn.BatchNorm2d(c * 3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1x1_3 = nn.Conv2d(c * 3, out_channels=c, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # 分支1：SpatialAvgPool -> Conv1x1 -> ReLU -> Conv1x1 -> Sigmoid
        avg_out = self.spatial_avg_pool(x)
        max_out = self.spatial_max_pool(x)
        x1 = avg_out + max_out
        x1 = self.conv1x1_1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv1x1_2(x1)
        x1 = self.sigmoid(x1)

        # 分支2：光谱维度多尺度的分组卷积模块
        x2_1 = self.conv7group(x)
        x2_2 = self.conv5group(x)
        x2_3 = self.conv3group(x)
        x2 = torch.cat((x2_1, x2_2, x2_3), dim=1)
        x2 = self.bn1(x2)
        x2 = self.relu2(x2)
        x2 = self.conv1x1_3(x2)
        x2 = self.bn2(x2)
        x2 = self.relu3(x2)

        x = (x1 * x2) + x

        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, in_channels=210):
        super(AdaptiveFusion, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels=in_channels // 2, kernel_size=1, groups=2)

        # ===========================修改Fusion
        out_channels = in_channels // 2
        self.norm = nn.BatchNorm2d(in_channels)
        # 左分支
        self.dw = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, groups=2)
        self.relu1 = nn.ReLU(inplace=True)
        # 右分支
        self.spatial_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        # 合并
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels=out_channels, kernel_size=1)
        # ===========================
        # 添加ReMix
        self.ln1 = nn.BatchNorm2d(out_channels)
        self.ln2 = nn.BatchNorm2d(out_channels)
        self.ln3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x = self.conv(x)
        # ===========================修改Fusion
        x1 = self.norm(x)
        # 下分支
        x1 = self.dw(x1)
        x1 = self.relu1(x1)
        # 上分支
        avg_out = self.spatial_avg_pool(x)
        max_out = self.spatial_max_pool(x)
        x2 = avg_out + max_out
        x2 = self.conv1x1_1(x2)
        x2 = self.relu2(x2)
        x2 = self.sigmoid(x2)
        # 相乘
        x = x1 * x2
        x = self.conv1x1_2(x)
        # ===========================
        # # 添加ReMix
        # x = x + self.ln1(x)
        # x = x + self.ln2(x)
        # x = x + self.ln3(x)
        return x


# Mulitscale triplet attention Module:
class MTAModule(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=7, branch_ratio=0.125):
        super(MTAModule, self).__init__()

        gc = int(in_channels * branch_ratio)  # 每个DWConv分支的通道数
        if gc < 1:
            gc = 1
        # 最左侧分支默认 3 x 3
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        # 第二个DWConv分支的卷积核默认为 1 x 11
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        # 第三个DWConv分支的卷积核默认为 11 x 1
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        # 切分tensor的通道维度，比如[96 - 3 * 12, 12, 12, 12]
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

        # 条形池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.norm = nn.BatchNorm2d(in_channels)
        self.tripletAttention = TripletAttention(in_channels)

    def forward(self, x):
        bs, c, h, w = x.shape
        # 划分通道块
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)

        # 进入四个分支后拼接
        x_cat = torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

        avg_w = self.pool_w(x)  # [BS, c, 1, p]
        avg_h = self.pool_h(x)  # [BS, c, p, 1]

        avg_w = rearrange(avg_w, 'BS c h w -> BS c w h')

        # cat x_h and x_w in dim = 2，W+H
        # Concat + Conv2d + BatchNorm + Non-linear
        xw = torch.cat((avg_h, avg_w), dim=2)  # [batch_size, c, h+w, 1]
        xw = self.relu1(self.norm1(self.conv1(xw)))  # [batch_size, c, h+w, 1]

        avg_h, avg_w = torch.split(xw, [h, w], dim=2)  # [batch_size, c, h, 1]  and [batch_size, c, w, 1]
        avg_w = rearrange(avg_w, 'BS c w h -> BS c h w')  # [batch_size,c,w,1] -> [batch_size, c, 1, w]
        # Conv2d + Sigmoid
        attention_h = self.sigmoid(self.conv_h(avg_h))
        attention_w = self.sigmoid(self.conv_w(avg_w))

        # x_avg = self.sigmoid(self.fc(self.relu(xw)))

        x_cat = x_cat * attention_w
        x_cat = x_cat * attention_h
        x_cat = self.norm(x_cat)
        x_cat = self.tripletAttention(x_cat)
        x = x + x_cat

        return x


# Mulitscale triplet attention Eembedding
class MTAEembedding(nn.Module):
    def __init__(self, in_chans,
                 out_channels,
                 n_groups,  # =num_heads
                 patch_size,
                 adaptive_patch_size=5
                 ):
        super(MTAEembedding, self).__init__()
        self.n = n_groups
        self.c = in_chans
        self.adaptive_patch_size = adaptive_patch_size

        self.mtam = nn.ModuleList([MTAModule(in_chans // n_groups)
                                   for j in range(n_groups)])
        # self.mtam = MTAModule(in_chans // n_groups, band_kernel_size=patch_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(adaptive_patch_size, adaptive_patch_size))

        self.Proj = nn.Linear(in_chans, out_channels)

    def forward(self, x):
        # x :-> [B, C, s, s]
        x_split = torch.split(x, self.c // self.n, dim=1)
        output_list = []
        i = 0
        for mtam in self.mtam:
            x_i = x_split[i]
            x_i = mtam(x_i)
            x_i = self.avg_pool(x_i)
            x_i = x_i.flatten(2).transpose(1, 2)
            output_list.append(x_i)
            i = i + 1
        output = torch.stack(output_list, dim=1)
        output = rearrange(output, 'BS h n d -> BS n (h d)')
        output = self.Proj(output)
        return output, self.adaptive_patch_size  # [BS, num_patches, num_heads*dim of each head]


# Efficient Head-Interacted Additive Attention:
class EHIAAttention(nn.Module):
    def __init__(self, num_heads, num_patches, dim, groups_m):
        super(EHIAAttention, self).__init__()
        self.num_heads = num_heads
        self.in_dims = dim // num_heads

        # ==================添加两个linear
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)

        # w_g ->: [N, 1]
        self.w_g = nn.Parameter(torch.randn(num_patches, 1))
        self.scale_factor = num_patches ** -0.5
        self.Proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        # ===================== 添加Avg分支
        self.d_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Linear(self.in_dims, dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(num_heads, num_heads)
        self.d_avg2 = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()
        self.saa = SAA_Attention(dim, num_heads_h=num_heads, groups_m=groups_m)

    def forward(self, x):
        # x ->: [BS, num_patches, num_heads*in_dims]
        bs = x.shape[0]

        # ==================添加两个linear
        q = self.fc_q(x)
        x = self.fc_k(x)
        x_t = q.transpose(1, 2)

        # x_T ->: [BS, D, N]
        # x_t = x.transpose(1, 2)

        # query_weight ->: [BS, D, 1] ->: [BS, 1, D]
        query_weight = (x_t @ self.w_g).transpose(1, 2)

        A = query_weight * self.scale_factor
        A = A.softmax(dim=-1)

        # A * x ->: [BS, N, D]
        # G ->: [BS, D]
        G = torch.sum(A * x, dim=1)

        # ===================== 添加Avg分支
        d_avg = self.d_avg(x_t)  # [BS, D, 1]
        d_avg = torch.squeeze(d_avg, 2)  # [BS, D]
        d_avg = d_avg.reshape(bs, self.num_heads, self.in_dims)  # [BS, h, d]
        d_avg = self.gelu(self.fc(d_avg))  # [BS, h, D]
        # d_avg = d_avg.reshape(BS, -1, self.num_heads)  # [BS, D, h]
        # d_avg = self.fc2(d_avg)  # [BS, D, h]
        # d_avg = self.sigmoid(self.d_avg2(d_avg))  # [BS, D, 1]
        # d_avg = torch.squeeze(d_avg, 2)  # [BS, D]
        # ================SAA
        d_avg = self.saa(d_avg)

        G = G * d_avg
        # =====================

        # G ->: [BS, N, D]
        # key.shape[1] = N
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=x.shape[1]
        )

        # out :-> [BS, N, D]
        out = self.Proj(G * x) + self.norm(x)
        # out = self.Proj(out)

        return out


# Efficient Head-Interacted Additive Transformer Encoder
class EHIATransformerEncoder(nn.Module):
    def __init__(self, num_heads, patch_size, in_dims, groups_m, mlp_ratio=1, drop=0.):
        super().__init__()
        dim = in_dims * num_heads
        num_patches = (patch_size * patch_size)

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = EHIAAttention(num_heads, num_patches, dim, groups_m)
        # self.attn2 = EfficientAdditiveAttention(dim, in_dims, num_heads)
        self.attn2 = Attention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # x :-> [BS, num_patches+1, num_heads*dim of each head]
        x = x + self.attn2(self.attn1(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """
    Vit中的Attention
    """

    def __init__(
            self,
            dim,  # 输入token的dim
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个head的dim
        self.scale = self.head_dim ** -0.5  # 缩放点积注意力中的缩放操作，即开跟操作

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 通过全连接层得到q,k,v
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 多个head的Contact操作
        self.proj_drop = nn.Dropout(proj_drop)

        # =============================添加MHSA cat权重
        self.cbam = nn.ModuleList([CBAMWeight(self.head_dim) for j in range(num_heads)])

    def forward(self, x):
        B, N, C = x.shape  # [batch_size, num_patches + 1, total_embed_dim]，论文中为[batchsize, 196+1, 768]

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3（代表qkv）, num_heads（代表head数）, embed_dim_per_head（每个head的qkv维度）]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        # q, k, v: -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale  # 缩放

        # q: -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # k.transpose(-2, -1): -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # attn: -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = q @ k.transpose(-2, -1)  # q,k 相乘，transpose为矩阵转置

        attn = attn.softmax(dim=-1)  # softmax操作
        attn = self.attn_drop(attn)  # dropout层

        x = attn @ v  # 矩阵乘法，*是元素乘法

        # =============================添加MHSA cat权重
        x_split = torch.split(x, 1, dim=1)
        output_list = []
        i = 0
        for cbam in self.cbam:
            x_i = x_split[i].squeeze(1)  # [BS, 1, N, d] ->: [BS, N, d]
            x_i = cbam(x_i)  # [BS, N, d] ->: [BS, 1, d]
            output_list.append(x_i)
            i = i + 1
        att = torch.cat(output_list, dim=2)  # [BS, 1, D]
        att = F.softmax(att, dim=2)
        # =============================
        x = x.transpose(1, 2).reshape(B, N, C)  # reshape相当于把head拼接
        x = att * x
        x = self.proj(x)  # 通过全连接进行拼接射（相当于乘论文中的Wo）
        x = self.proj_drop(x)
        return x


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


class Multi_extractor(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratios=16
                 ):
        super(Multi_extractor, self).__init__()

        self.spa = SpatialExtractor(input_channels=dim)
        self.spe = SpectralExtractor(input_channels=dim, ratio=mlp_ratios)
        self.fus = AdaptiveFusion(in_channels=dim * 2)

    def forward(self, x):
        """
        input: (BS, b, p, p)
        output: (BS, num_patches, dim)
        """
        x_spa = self.spa(x)
        x_spe = self.spe(x)
        x = torch.cat([x_spa, x_spe], dim=1)
        x = self.fus(x)
        return x


class InteractTE(nn.Module):
    def __init__(self,
                 dim,
                 emb_dim,
                 patch_size,
                 num_heads,
                 adaptive_patch_ratio=2,
                 mlp_ratios=1,
                 depths_te=1,
                 groups_m=1
                 ):
        super(InteractTE, self).__init__()

        adaptive_patch_size = patch_size - adaptive_patch_ratio

        self.patch_embed = MTAEembedding(
            in_chans=dim,
            out_channels=emb_dim,
            n_groups=num_heads,
            patch_size=patch_size,
            adaptive_patch_size=adaptive_patch_size
        )

        self.te = nn.ModuleList([EHIATransformerEncoder(
            num_heads=num_heads,
            patch_size=adaptive_patch_size,
            in_dims=emb_dim // num_heads,
            mlp_ratio=mlp_ratios,
            groups_m=groups_m) for j in range(depths_te)])

        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        """
        input: (BS, b, p, p)
        patch_embed:-> (BS, num_patches, dim)
        te:-> (BS, num_patches, dim)
        output: (BS, num_patches, dim)
        """
        x, p = self.patch_embed(x)
        for te in self.te:
            x = te(x)
        x = self.norm(x)
        return x, p


class CBAMWeight(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super().__init__()

        self.fc1 = nn.Linear(in_planes, in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_planes, in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [BS, N, d]

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        avg_out = self.fc2(self.relu1(self.fc1(avg_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_out)))
        out = self.sigmoid(avg_out + max_out)  # [BS, 1, d]

        return out


class overall(nn.Module):
    def __init__(self, patch_size,  # the size of input img
                 in_chans,  # the bands number of input img
                 num_classes=16,
                 out_channels_3d=12,  # the out_channels of 3Dconv in shallow extractor
                 num_heads=16,
                 groups_m=16,
                 num_extras=1,
                 num_stages=1,
                 adaptive_patch_ratio=0,
                 extra_dim=[64, ],
                 dim=[64, 64, 64],
                 depths_te=[1, 1, 1],
                 ratio_SpectralExtractor=4,  # ratio at SpectralExtractor
                 mlp_ratios=1,  # mlp_ratios at transformer mlp

                 ):
        super(overall, self).__init__()

        self.num_stages = num_stages
        self.num_extras = num_extras

        for i in range(num_extras):
            if i == 0:
                input_channels = in_chans
            else:
                input_channels = extra_dim[i - 1]
            extractor = MulitiScaleAttentionExtractor(input_channels=input_channels,
                                                      out_channels_3d=out_channels_3d,
                                                      out_channels_2d=extra_dim[i])

            multi_extractor = Multi_extractor(dim=extra_dim[i],
                                              mlp_ratios=ratio_SpectralExtractor)
            setattr(self, f"extractor{i + 1}", extractor)
            setattr(self, f"multi_extractor{i + 1}", multi_extractor)

        for i in range(num_stages):
            te = InteractTE(dim=dim[0] if i == 0 else dim[i - 1],
                            emb_dim=dim[i],
                            patch_size=patch_size,
                            num_heads=num_heads,
                            adaptive_patch_ratio=adaptive_patch_ratio,
                            mlp_ratios=mlp_ratios,
                            depths_te=depths_te[i],
                            groups_m=groups_m
                            )
            setattr(self, f"te{i + 1}", te)

        # Classifier head
        self.head = nn.Linear(dim[-1], num_classes)

    def forward_features(self, x):
        # B = BS
        B = x.shape[0]
        for i in range(self.num_extras):
            extractor = getattr(self, f"extractor{i + 1}")
            multi_extractor = getattr(self, f"multi_extractor{i + 1}")
            if i == 0:
                pass
            else:
                x = x.unsqueeze(1)
            x = extractor(x)
            x = multi_extractor(x)

        for i in range(self.num_stages):
            te = getattr(self, f"te{i + 1}")
            x, s = te(x)

            # 不是最后一个stage的话
            if i != self.num_stages - 1:
                # [B, N, C] ->: [B, s, s, C] ->: [B, C, s, s]
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


def MHIAIFormer(dataset,
             patch_size=11,
             num_heads=2,
             groups_m=2,
             dim=64,
             ratio_SpectralExtractor=1,
             mlp_ratios=1,
             out_channels_3d=9,
             adaptive_patch_ratio=0,
             depths_te=1
             ):
    model = None
    if dataset == 'sa':
        model = overall(patch_size=patch_size,
                        in_chans=204,
                        num_classes=16,
                        num_heads=num_heads,
                        # dim=dim,
                        ratio_SpectralExtractor=ratio_SpectralExtractor,
                        mlp_ratios=mlp_ratios,
                        out_channels_3d=out_channels_3d,
                        adaptive_patch_ratio=adaptive_patch_ratio,
                        # depths_te=depths_te
                        groups_m=groups_m
                        )
    elif dataset == 'pu':
        model = overall(patch_size=patch_size,
                        in_chans=103,
                        num_classes=9,
                        num_heads=num_heads,
                        # dim=dim,
                        ratio_SpectralExtractor=ratio_SpectralExtractor,
                        mlp_ratios=mlp_ratios,
                        out_channels_3d=out_channels_3d,
                        adaptive_patch_ratio=adaptive_patch_ratio,
                        # depths_te=depths_te
                        groups_m=groups_m
                        )
    elif dataset == 'whulk':
        model = overall(patch_size=patch_size,
                        in_chans=270,
                        num_classes=9,
                        num_heads=num_heads,
                        # dim=dim,
                        ratio_SpectralExtractor=ratio_SpectralExtractor,
                        mlp_ratios=mlp_ratios,
                        out_channels_3d=out_channels_3d,
                        adaptive_patch_ratio=adaptive_patch_ratio,
                        # depths_te=depths_te
                        groups_m=groups_m
                        )
    elif dataset == 'hrl':
        model = overall(patch_size=patch_size,
                        in_chans=176,
                        num_classes=14,
                        num_heads=num_heads,
                        # dim=dim,
                        ratio_SpectralExtractor=ratio_SpectralExtractor,
                        mlp_ratios=mlp_ratios,
                        out_channels_3d=out_channels_3d,
                        adaptive_patch_ratio=adaptive_patch_ratio,
                        # depths_te=depths_te
                        groups_m=groups_m
                        )
    elif dataset == 'whuhh':
        model = overall(patch_size=patch_size,
                        in_chans=270,
                        num_classes=22,
                        # num_heads=num_heads,
                        # dim=dim,
                        ratio_SpectralExtractor=ratio_SpectralExtractor,
                        mlp_ratios=mlp_ratios,
                        out_channels_3d=out_channels_3d,
                        adaptive_patch_ratio=adaptive_patch_ratio,
                        # depths_te=depths_te
                        )
    elif dataset == 'whuhc':
        model = overall(patch_size=patch_size,
                        in_chans=274,
                        num_classes=16,
                        # num_heads=num_heads,
                        # dim=dim,
                        ratio_SpectralExtractor=ratio_SpectralExtractor,
                        mlp_ratios=mlp_ratios,
                        out_channels_3d=out_channels_3d,
                        adaptive_patch_ratio=adaptive_patch_ratio,
                        # depths_te=depths_te
                        )
    elif dataset == 'IP':
        model = overall(patch_size=patch_size,
                        in_chans=200,
                        num_classes=16,
                        # num_heads=num_heads,
                        # dim=dim,
                        ratio_SpectralExtractor=ratio_SpectralExtractor,
                        mlp_ratios=mlp_ratios,
                        out_channels_3d=out_channels_3d,
                        adaptive_patch_ratio=adaptive_patch_ratio,
                        # depths_te=depths_te
                        )
    return model


if __name__ == "__main__":
    # t = torch.randn(size=(1, 1, 100, 7, 7))
    # out = overall(patch_size=7, in_chans=100).forward(t)
    # print("10 shape:", out.shape)

    t = torch.randn(size=(64, 1, 103, 7, 7))
    net = MHIAIFormer(dataset='pu', patch_size=7)
    print("output shape:", net(t).shape)

    t = torch.randn(size=(64, 1, 20, 7, 7))
    net = MulitiScaleAttentionExtractor(input_channels=t.shape[2],  # the bands number of img
                     out_channels_3d=3,
                     out_channels_2d=20,
                     stride=[1, 1, 1]
                     )
    print("output shape:", net(t).shape)

    from fvcore.nn import FlopCountAnalysis, flop_count_table

    net.eval()
    flops = FlopCountAnalysis(net, t)
    print(flop_count_table(flops))

    from thop import profile, clever_format

    flops, params = profile(net, inputs=(t,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
