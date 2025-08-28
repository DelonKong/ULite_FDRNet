import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_


'''
@ARTICLE{11142884,
  author={Shu, Zhenqiu and Zeng, Kexin and Zhou, Jun and Wang, Yuyang and Tai, Mingzhu and Yu, Zhengtao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Spectral-Spatial Synergy Guided Attention Network for Hyperspectral Image Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Attention mechanisms;Computer architecture;Computational modeling;Convolution;Computational efficiency;Transformers;Training;Hyperspectral imaging;Data mining;HSIC;CNNs;attention mechanism;dual shift network;guiding tokens;local and global features},
  doi={10.1109/TGRS.2025.3603397}}

'''


class HDFE(nn.Module):
    def __init__(self, channels,):
        super(HDFE, self).__init__()
        self.channels = channels
        self.conv = nn.Sequential(
                                  nn.ReLU6(),
                                  nn.Conv3d(4 * channels, channels, kernel_size=(1, 1, 1), padding=0, groups=1),
                                 )

        self.dwc1 = nn.Sequential(nn.Conv3d(channels, 4 * channels, kernel_size=(1, 1, 1), padding=0, groups=1),
                                  nn.BatchNorm3d(4 * channels),
                                  nn.ReLU6(),
                                 )
        self.multi_spatial_branches_conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), bias=False,
                      groups=channels),
            nn.Sigmoid(),
        )
        self.multi_spatial_branches_conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(5, 5, 1), stride=(1, 1, 1), padding=(2, 2, 0), bias=False,
                      groups=channels),
            nn.Sigmoid(),
        )
        self.multi_spatial_branches_conv3 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(7, 7, 1), stride=(1, 1, 1), padding=(3, 3, 0), bias=False,
                      groups=channels),
            nn.Sigmoid(),
        )
        self.multi_spatial_branches_conv4 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), bias=False,
                      groups=channels),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        for scale_idx in range(3):
            self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(3)])
            self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(3)])

    def forward(self, x):
        x1 = self.dwc1(x)
        y1 = x1[:, self.channels:2 * self.channels, :]
        spatial_att_map1 = self.multi_spatial_branches_conv1(y1)
        feature1 = self.multi_spatial_branches_conv4(
            y1 * (1 - spatial_att_map1) * self.alpha_list[0] + y1 * spatial_att_map1 * self.beta_list[0])
        y2 = x1[:, 2 * self.channels:3 * self.channels, :]
        spatial_att_map2 = self.multi_spatial_branches_conv2(y2)
        feature2 = self.multi_spatial_branches_conv4(
            y2 * (1 - spatial_att_map2) * self.alpha_list[1] + y2 * spatial_att_map2 * self.beta_list[1])
        y3 = x1[:, 3 * self.channels:4 * self.channels, :]
        spatial_att_map3 = self.multi_spatial_branches_conv3(y3)
        feature3 = self.multi_spatial_branches_conv4(
            y3 * (1 - spatial_att_map3) * self.alpha_list[2] + y3 * spatial_att_map3 * self.beta_list[2])

        feature_fusion = self.conv(torch.cat((x1[:, 0:self.channels, :], feature1, feature2, feature3), dim=1))

        return feature_fusion

class SRSFE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.spectral1 = nn.Sequential(nn.Conv3d(self.channels, self.channels, kernel_size=(1, 1, 3), padding=(0, 0, 1),
                                                 ),
                                       nn.BatchNorm3d(self.channels),
                                       nn.Sigmoid()
                                       )
        self.spectral2 = nn.Sequential(nn.Conv3d(2 * self.channels, self.channels, kernel_size=(1, 1, 7),
                                                 padding=(0, 0, 3), ),
                                       nn.BatchNorm3d(self.channels),
                                       nn.Sigmoid()
                                       )
        self.spectral3 = nn.Sequential(nn.Conv3d(2 * self.channels, self.channels, kernel_size=(1, 1, 11),
                                                 padding=(0, 0, 5), ),
                                       nn.BatchNorm3d(self.channels),
                                       nn.Sigmoid()
                                       )
        self.spectral4 = nn.Sequential(nn.Conv3d(2 * self.channels, self.channels, kernel_size=(1, 1, 15),
                                                 padding=(0, 0, 7), ),
                                       nn.BatchNorm3d(self.channels),
                                       nn.Sigmoid()
                                       )
        self.out = nn.Sequential(nn.Conv3d(4 * self.channels, self.channels, kernel_size=(1, 1, 1),
                                           padding=(0, 0, 0), groups=1),
                                 nn.BatchNorm3d(self.channels),
                                 nn.ReLU6(),
                                 )

    def forward(self, x):
        x1 = self.spectral1(x)
        x2 = self.spectral2(torch.cat((x, x1), dim=1))
        x3 = self.spectral3(torch.cat((x, x2), dim=1))
        x4 = self.spectral4(torch.cat((x, x3), dim=1))
        srsfe = self.out(torch.cat((x1, x2, x3, x4), dim=1))
        return srsfe

class GuidedAttention(nn.Module):  # guided attention
    def __init__(self, channels, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, window=14, fc_dim=16,  **kwargs):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window  # 14

        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))

        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))
        # 权重初始化
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.ac_bias, std=.02)
        trunc_normal_(self.ca_bias, std=.02)
        pool_size = int(agent_num ** 0.5)

        self.dwc = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=1, groups=channels),
                                 )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        self.conv = nn.Sequential(nn.Conv2d(num_heads, num_heads, kernel_size=(3, 3), padding=(1, 1)),
                                  nn.BatchNorm2d(num_heads), nn.Sigmoid())
        self.dsc = nn.Sequential(nn.Conv2d(num_heads, num_heads, kernel_size=(3, 3), padding=1, groups=num_heads),
                                 nn.Conv2d(num_heads, num_heads, kernel_size=(1, 1), padding=0, groups=1),
                                 nn.BatchNorm2d(num_heads), nn.Sigmoid()
                                 )
        self.dwc2 = nn.Sequential(nn.Conv2d(num_heads, num_heads, kernel_size=(3, 3), padding=(1, 1), groups=1),
                                  nn.BatchNorm2d(num_heads), nn.Sigmoid()
                                  )
        self.ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1, groups=dim),
                                 nn.BatchNorm2d(dim),
                                 nn.Conv2d(dim, dim, kernel_size=(1, 1), padding=0, groups=1),
                                 nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1, groups=dim),
                                 nn.BatchNorm2d(dim),
                                 )

    def forward(self, x):
        x2 = rearrange(x, 'b n h w c -> b (n h w) c')
        b, n, c = x2.shape

        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qkv1 = self.qkv(x2).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        q1 = q1.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k1 = k1.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v1 = v1.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        agent_attn1 = (q1 * self.scale) @ k1.transpose(-2, -1)
        agent_attn1 = self.softmax(self.dwc2(agent_attn1))
        agent_attn1 = self.attn_drop(agent_attn1)
        z1 = agent_attn1 @ v1

        z1 = z1.transpose(1, 2).reshape(b, n, c)
        z1 = self.proj_drop(self.proj(z1))  # [64, 1296, 49]
        out = z1.reshape(b, self.channels, int((n/self.channels) ** 0.5), int((n/self.channels) ** 0.5), c)
        out += x
        return out

class DSGAN(nn.Module):
    def __init__(self, channels=16, patch=9, bands=200, num_class=16, fc_dim=16, heads=8, drop=0.1):
        super().__init__()
        self.channels = channels
        self.band_reduce = (bands - 9) // 2 + 1
        self.stem = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(1, 1, 9), padding=0, stride=(1, 1, 2)),
                                  )
        self.spectral = SRSFE(channels)
        self.spatial = HDFE(channels)
        self.block2 = nn.Sequential(
            nn.Conv3d(2 * channels, channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, groups=1),
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, groups=channels),
            nn.BatchNorm3d(channels),
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, groups=1),
            nn.ReLU6(),
        )
        self.atten = GuidedAttention(channels, dim=self.band_reduce, num_heads=8)
        self.out = nn.Sequential(nn.Conv3d(channels, fc_dim, kernel_size=(1, 1, self.band_reduce), padding=0),
                                 nn.BatchNorm3d(fc_dim),
                                 nn.ReLU6(),
                                 )
        self.fc = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(), nn.Linear(fc_dim, num_class))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 4, 2)
        b, _, _, _, _ = x.shape
        x = self.stem(x)
        x1 = self.spectral(x)
        x2 = self.spatial(x)
        out1 = self.block2(torch.cat((x1, x2), dim=1))
        feature1 = self.atten(out1)
        feature = self.out(feature1)
        return self.fc(feature)


if __name__ == '__main__':
    t = torch.randn(size=(64, 103, 9, 9))
    print("input shape:", t.shape)
    model = DSGAN(bands=103, num_class=9)
    model.eval()
    print(model)
    print("output shape:", model(t).shape)