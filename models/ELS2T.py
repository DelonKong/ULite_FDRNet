from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import collections.abc as container_abcs

from torchinfo import summary


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, h, w):
        return self.fn(x, h, w) + x


# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, h, w):
        return self.fn(self.norm(x), h, w)


# 等于 FeedForward
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
        self.gelu = nn.GELU()

    def forward(self, x, h, w):
        branch1 = self.fc1(x)
        branch2 = self.gelu(self.fc2(x))
        out = branch1 * branch2
        out = self.fc3(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_t = nn.Linear(dim, 1)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        t, k, v = self.proj_t(x), self.proj_k(x), self.proj_v(x)
        t = F.softmax(t, dim=1)
        attn = t * k
        attn = torch.sum(attn, dim=1, keepdim=True)
        attn = attn * F.relu(v) * torch.mean(x, dim=1, keepdim=True)
        out = self.proj(attn)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 input_resolution,
                 dim,
                 depth,
                 mlp_dim,
                 dropout):
        super().__init__()
        self.input_resolution = input_resolution
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim))),
                Residual(LayerNormalize(dim, MLP(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        b, n, d = x.size()
        h, w = to_2tuple(self.input_resolution)
        assert n == h * w + 1, "input feature has wrong size"
        for attention, mlp in self.layers:
            x = attention(x, h, w)  # go to attention
            x = mlp(x, h, w)  # go to MLP_Block

        return x


class AFF(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                             bias=False)
        self.conv21 = nn.Conv2d(in_planes, in_planes, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0),
                                bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_max = torch.cat((avg_out, max_out), dim=2)
        avg_max = self.conv21(avg_max)
        avg_max = self.fc1(avg_max)
        out = self.sigmoid(avg_max)
        out = x * out
        out = x + out

        return out


class MCAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.max_pool_h = nn.AdaptiveMaxPool2d((1, None))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((None, 1))
        self.max_pool_w = nn.AdaptiveMaxPool2d((None, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), dilation=3)
        self.conv3 = nn.Conv2d(in_planes, in_planes, kernel_size=(3, 3), stride=(1, 1), padding=(5, 5), dilation=5)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.bn3 = nn.BatchNorm2d(in_planes)
        self.fc1 = nn.Conv2d(in_planes * 3, in_planes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                             bias=True)
        self.bn4 = nn.BatchNorm2d(in_planes)
        self.gelu = nn.GELU()
        self.bn5 = nn.BatchNorm2d(in_planes // 4)
        self.conv4 = nn.Conv2d(in_planes, in_planes // 4, kernel_size=(1, 1), stride=(1, 1))
        self.conv5 = nn.Conv2d(in_planes // 4, in_planes, kernel_size=(1, 1), stride=(1, 1))
        self.conv6 = nn.Conv2d(in_planes // 4, in_planes, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x, h, w):
        h_out = self.max_pool_h(x) + self.avg_pool_h(x)
        w_out = self.max_pool_w(x) + self.avg_pool_w(x)
        x_cat_conv_gelu = self.gelu(self.bn5(self.conv4(torch.cat((h_out, w_out.permute(0, 1, 3, 2)), dim=3))))
        x_h_split, x_w_split = torch.split(x_cat_conv_gelu, [h, w], dim=3)
        x_w_split = x_w_split.permute(0, 1, 3, 2)
        x_h_split_conv = self.conv5(x_h_split)
        x_w_split_conv = self.conv6(x_w_split)
        s1 = self.sigmoid1(x_h_split_conv)
        s2 = self.sigmoid2(x_w_split_conv)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x_merge = torch.cat((x1, x2, x3), dim=1)
        x_merge = F.relu(self.bn4(self.fc1(x_merge)))
        out = x_merge * s1 * s2

        return out


class CNN_Block(nn.Module):
    def __init__(self, in_planes, planes, categroy):
        super().__init__()
        if categroy == "spa" or categroy == "spatial":
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1, 3, 3),
                                   stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
            self.bn1 = nn.BatchNorm3d(planes)
            self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3),
                                   stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
            self.bn2 = nn.BatchNorm3d(planes)
            self.shortcut = nn.Sequential()
        if categroy == "spe" or categroy == "spectral":
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(7, 1, 1),
                                   stride=(1, 1, 1), padding=(3, 0, 0), bias=True)
            self.bn1 = nn.BatchNorm3d(planes)
            self.conv2 = nn.Conv3d(planes, planes, kernel_size=(7, 1, 1),
                                   stride=(1, 1, 1), padding=(3, 0, 0), bias=True)
            self.bn2 = nn.BatchNorm3d(planes)
            self.shortcut = nn.Sequential()
        if in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes,
                          kernel_size=1, bias=True),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class spectral_CNN(nn.Module):
    def __init__(self, in_planes, planes, in_chans):
        super().__init__()
        self.conv1 = nn.Conv3d(1, in_planes, kernel_size=(7, 1, 1),
                               stride=(2, 1, 1), padding=(0, 0, 0), bias=True)
        self.block1 = CNN_Block(in_planes, planes, categroy="spe")
        self.block2 = CNN_Block(planes, planes, categroy="spe")
        self.block3 = CNN_Block(planes, planes, categroy="spe")
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=((in_chans-7)//2+1, 1, 1),
                               stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

    def forward(self, x):
        x = self.conv1(x)
        feature1 = self.block1(x)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)
        out = self.conv2(feature3)
        out = out.squeeze(dim=2)

        return out


class spatial_CNN(nn.Module):
    def __init__(self, in_planes, planes, in_chans):
        super().__init__()
        self.conv1 = nn.Conv3d(1, in_planes, kernel_size=(in_chans, 1, 1),
                               stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.block1 = CNN_Block(in_planes, planes, categroy="spa")
        self.block2 = CNN_Block(planes, planes, categroy="spa")
        self.block3 = CNN_Block(planes, planes, categroy="spa")

    def forward(self, x):
        x = self.conv1(x)
        feature1 = self.block1(x)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)
        out = feature3.squeeze(dim=2)

        return out


class overall(nn.Module):
    def __init__(self,
                 patch_size,
                 in_planes=6,
                 planes=6,
                 in_chans=20,
                 num_classes=16,
                 depth=2,
                 mlp_dim=1,
                 dropout=0.1,
                 emb_dropout=0.1):
        super(overall, self).__init__()

        self.spatial_feature = spatial_CNN(in_planes, planes, in_chans)
        self.spectral_feature = spectral_CNN(in_planes, planes, in_chans)
        self.mcam1 = MCAM(planes)
        self.mcam2 = MCAM(planes)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.aff = AFF(planes * 2)
        self.pos_embedding = nn.Parameter(torch.empty(1, (patch_size ** 2 + 1), planes * 2))
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, planes * 2))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(patch_size, planes * 2, depth, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.nn1 = nn.Linear(planes * 2, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x):
        bs, _, c, h, w = x.shape
        spa = self.spatial_feature(x)
        spe = self.spectral_feature(x)
        spa_spe = spa + spe
        spa_conv = self.conv1(spa_spe)
        spe_conv = self.conv2(spa_spe)
        spa_weight = self.mcam1(spa_conv, h, w)
        spa_w = spa_conv + spa_weight
        spe_weight = self.mcam2(spe_conv, h, w)
        spe_w = spe_conv + spe_weight
        spa = spa + spa_w
        spe = spe + spe_w
        x = torch.cat((spa, spe), dim=1)
        x = self.aff(x)

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)  # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x


def ELS2T(dataset,
          patch_size=7,
          mlp_ratios=1,
          depths_te=2
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
    device = torch.device("cuda:{}".format(0))

    t = torch.randn(size=(64, 1, 103, 9, 9)).to(device)
    net = ELS2T(dataset='pu', patch_size=9)
    net.to(device)
    print("output shape:", net(t).shape)
    with torch.no_grad():
        sum = summary(net, input_size=(1, 1, t.shape[2], t.shape[-2], t.shape[-1]), verbose=0)
        print(sum)
        print(sum.trainable_params)

    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    # net.eval()
    # flops = FlopCountAnalysis(net, t)
    # print(flop_count_table(flops))
    #
    # from thop import profile, clever_format
    # flops, params = profile(net, inputs=(t,))
    # macs, params = clever_format([flops, params], "%.3f")
    # print(macs, params)
    #
    # print("==========================================")
    # t = torch.randn(size=(1, 50, 12)).to(device)
    # taa = Attention(12).to(device)
    # device = torch.device("cuda:{}".format(0))
    # taa.to(device)
    # print(taa(t, 9, 9)[0].shape)
    # taa.eval()
    # flops = FlopCountAnalysis(taa, (t, 9, 9))
    # print("==========================================")
    # print(flop_count_table(flops))