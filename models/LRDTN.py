import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import einops
import torch.fft
from einops.layers.torch import Rearrange
from torchinfo import summary


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

class SRA(nn.Module):
    def __init__(self, dim: int, sr_ratio=2):
        super().__init__()
        self.dim = dim
        self.q = nn.Conv2d(256, dim, kernel_size=1, groups=dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(256, dim, kernel_size=3, padding=1, groups=dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x1 = x.reshape(B, N, C)
        if self.sr_ratio > 1:
            x_ = self.sr(x)
            x_ = x_.reshape(B, -1, 128)
            x2 = self.norm(x_)
        else:
            x2 = self.q(x).reshape(B, -1, 128)
        return x2


class MH2SD(nn.Module):

    def __init__(self, dim: int, head_dim: int, num_heads: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim

        self.head_dim = head_dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)
        self.osr = SRA(dim, num_heads)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(self.dim)
        self.qkc = nn.Conv2d(self.dim, self.inner_dim * 3, kernel_size=1, padding=0, groups=head_dim, bias=False)
        self.spe = nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=head_dim, bias=False)
        self.bnc = nn.BatchNorm2d(self.inner_dim)
        self.bnc1 = nn.BatchNorm2d(dim)
        self.local = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=self.head_dim,
                               bias=False)
        self.avgpool = nn.AdaptiveAvgPool1d(dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        qkv = self.qkc(self.act(self.bn(x)))
        qkv = qkv.contiguous().view(b, self.num_patch * self.num_patch, self.inner_dim * 3)

        qkv = qkv.chunk(3, dim=-1)
        spe = self.spe(self.act(self.bn(x)))
        spe = self.avg_pool(spe)
        c = x
        q, k, v = map(lambda t: einops.rearrange(t, "b (h d) n -> b n h d", h=self.num_patch), qkv)
        qqkkvv = q = k = v
        qy = self.osr(qqkkvv)
        q = einops.rearrange(qy, "b n (h d) -> b h n d", h=self.num_heads)
        k = v = q
        spe = einops.rearrange(spe, "b (h d) n w -> b h (n w) d", h=self.num_heads)
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        scores = scores * self.scale
        attn = self.attn(scores)
        v_spe = torch.einsum("b h i j, b h j d -> b h i d", v, spe)
        v_spe = v_spe * self.scale
        v_spe1 = self.attn(v_spe)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v_spe1)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        c = self.act(self.bnc1(self.local(c)))
        c = c.reshape(b, self.dim, self.num_patch, self.num_patch).reshape(b, n, -1)
        out = self.avgpool(out + c)

        return out


class FCE(nn.Module):

    def __init__(self, dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.dim = dim
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.depthwise_conv = nn.Sequential(

            nn.Conv2d(dim, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.GELU())

        self.squeeze_conv = nn.Sequential(

            nn.Conv2d(64, 16, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(16), nn.GELU())

        self.expand_conv = nn.Sequential(

            nn.Conv2d(16, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, hw, dim = x.shape
        x_reshape = x.contiguous().view(b, self.dim, self.num_patch, self.num_patch)
        out1 = self.depthwise_conv(x_reshape)
        out2 = self.squeeze_conv(out1)
        out3 = self.expand_conv(out2) + x_reshape
        result = out3.contiguous().view(b, self.num_patch * self.num_patch, self.dim)
        result = result + x
        return result


class LGPT_module(nn.Module):

    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int, num_patch: int, patch_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                nn.Sequential(nn.LayerNorm(dim), MH2SD(dim, head_dim, num_heads, num_patch, patch_size)),
                nn.Sequential(nn.LayerNorm(dim), FCE(dim, num_patch, patch_size))
            ]
            self.layers.append(nn.ModuleList(layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mh, fce in self.layers:
            x = mh(x) + x
            x = fce(x) + x
        return x


class PatchEmbeddings(nn.Module):

    def __init__(self, patch_size: int, patch_dim: int, emb_dim: int):
        super().__init__()
        self.patchify = Rearrange(
            "b c h w-> b (h w) c")
        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(in_features=patch_dim, out_features=emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x

class PositionalEmbeddings(nn.Module):

    def __init__(self, num_pos: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_pos, dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos


class DDC_module(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=2,
                 num_groups=2,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            Conv2dBN(dim,
                     dim // reduction_ratio,
                     kernel_size=1,
                     act_layer=nn.GELU),
            nn.Conv2d(dim // reduction_ratio, dim, kernel_size=1), )
        self.proj1 = nn.Sequential(
            Conv2dBN(dim,
                       dim // reduction_ratio,
                       kernel_size=1,
                       act_layer=nn.GELU),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1), )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        x.cuda()
        B, C, H, W = x.shape
        x1 = self.pool(x)
        scale1 = self.proj(x1)
        scale3 = torch.cat((scale1,x1),dim=1)
        scale4 = scale3.reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale4, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj1(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(B, C, H, W)



class MultiScaleDWConv(nn.Module):
    # def __init__(self, dim, scale=(1, 3, 5, 7)):#原始
    def __init__(self, dim, scale=(1, 3,5,7)):#消融实验
        super().__init__()

        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x






class MsEF_module(nn.Module):
    """
    Mlp implemented by with 1x1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 act_cfg=dict(type='GELU'),
                 drop=0, ):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, in_features//6, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(in_features//6),
        )
        self.dwconv = MultiScaleDWConv(in_features//6)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(in_features//6)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_features//6, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x



class overall(nn.Module):
    def __init__(self, classes, HSI_Data_Shape_C,
                 patch_size, image_size: int =121, emb_dim: int = 128, num_layers: int = 1,
                 num_heads: int =4, head_dim = 64, hidden_dim: int = 128, attn_drop : int = 0, sr_ration:int = 1,dim1=128,
                 act_cfg=dict(type='GELU')):
        super(overall, self).__init__()
        HSI_Data_Shape_H = patch_size-1
        HSI_Data_Shape_W = patch_size-1
        self.classes = classes
        self.HSI_Data_Shape_H = HSI_Data_Shape_H
        self.HSI_Data_Shape_W = HSI_Data_Shape_W
        self.band = HSI_Data_Shape_C
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.channels = HSI_Data_Shape_C
        self.image_size = image_size
        self.num_patches = HSI_Data_Shape_H * HSI_Data_Shape_W
        self.attn_drop = attn_drop
        self.sr_ration = sr_ration
        self.num_patch = int(math.sqrt(self.num_patches))
        patch_dim = HSI_Data_Shape_C
        kernel_size=3
        num_groups=2
        drop=0
        self.dim1=dim1
        self.relu = nn.ReLU()

        """branch 1"""
        self.DDC = DDC_module(dim=self.band, kernel_size=kernel_size, num_groups=num_groups)
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=self.band, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )


        """branch 2"""

        self.patch_embeddings = PatchEmbeddings(patch_size=patch_size, patch_dim=patch_dim, emb_dim=emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_pos=self.num_patches, dim=emb_dim)
        self.LGPT = LGPT_module(dim=emb_dim, num_layers=num_layers, num_heads=num_heads,
                                        head_dim=head_dim, num_patch=self.num_patch, patch_size=patch_size)

        self.MsEF = MsEF_module(in_features=self.band,
                       act_cfg = act_cfg,
                       drop = attn_drop, )

        self.Dconv21 = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(emb_dim),
        )
        self.drop = nn.Dropout(drop)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.finally_fc_classification = nn.Linear(128, self.classes)

    # def forward(self, pixelX , patchX):
    def forward(self, patchX):
        patchX = patchX.squeeze(1)
        pixelX = patchX

        """------------------------branch 1------------------------"""
        pixelX = pixelX.cuda()
        x11 = self.DDC(pixelX)
        x12 = x11 + pixelX
        x13 = self.MsEF(x12)
        x14 = self.conv11(x13+pixelX)
        output_1 = self.global_pooling(x14)

        """------------------------branch 2------------------------"""

        patchX=patchX.cuda()
        x21 = self.patch_embeddings(patchX)
        x22 = self.pos_embeddings(x21)
        x23 = self.LGPT(x22)
        x24 = torch.mean(x23, dim=1)
        output_2 = x24.unsqueeze(-1).unsqueeze(-1)

        """------------------------fusion------------------------"""

        output3 = output_1 + output_2
        output4 = self.Dconv21(output3)
        output5 = self.drop(output4)
        output6 = torch.squeeze(output5,dim=(2,3))
        output7 = self.finally_fc_classification(output6)
        output = F.softmax(output7, dim=1)

        return output

def LRDTN(dataset, patch_size=7):

    model = None
    patch_size = patch_size + 1
    if dataset == 'sa':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=204,
                        classes=16,
                        )
    elif dataset == 'pu':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=103,
                        classes=9,
                        )
    elif dataset == 'whulk':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=270,
                        classes=9,
                        )
    elif dataset == 'hrl':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=176,
                        classes=14,
                        )
    elif dataset == 'whuhh':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=270,
                        classes=22,
                        )
    elif dataset == 'whuhc':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=274,
                        classes=16,
                        )
    elif dataset == 'IP':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=200,
                        classes=16,
                        )
    elif dataset == 'BS':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=145,
                        classes=14,
                        )
    elif dataset == 'HsU':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=144,
                        classes=15,
                        )
    elif dataset == 'KSC':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=176,
                        classes=13,
                        )
    elif dataset == 'pc':
        model = overall(patch_size=patch_size,
                        HSI_Data_Shape_C=102,
                        classes=9,
                        )
    return model


if __name__ == "__main__":
    device = torch.device("cuda:{}".format(0))

    t = torch.randn(size=(64, 1, 103, 9, 9)).to(device)

    net = LRDTN(dataset="pu", patch_size=9)

    net.to(device)
    print("output shape:", net(t).shape)

    with torch.no_grad():
        sum = summary(net, input_size=(1, 1, t.shape[2], t.shape[-2], t.shape[-1]), verbose=0)
        print(sum)