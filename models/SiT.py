'''
@ARTICLE{11124215,
author={Chen, Zhonghao and Roy, Swalpa Kumar and Gao, Hongmin and Ding, Yao and Zhang, Bing},
journal={IEEE Transactions on Geoscience and Remote Sensing},
title={SiT: Scale-interaction Transformer for Hyperspectral Image Classification},
year={2025},
volume={},
number={},
pages={1-1},
doi={10.1109/TGRS.2025.3598290}}
'''

import torch

from Base import *
device = torch.device("cuda:"+str(GPUiD) if torch.cuda.is_available() else "cpu")


class Residual_dual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x1, x2, **kwargs):
        # 仅调用一次 self.fn
        output1, output2 = self.fn(x1, x2, **kwargs)
        return output1 + x1, output2 + x2
class LayerNormalize_dual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm(x1), self.norm(x2), **kwargs)

class MLP_Block_Dual(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x1, x2):
        return self.net1(x1), self.net2(x2)

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, padding, kernel_size, stride, G):
        super().__init__()
        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=padding,
                                kernel_size=kernel_size, stride=stride, groups=G)
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )
    def forward(self, X):
        X = self.conv11(X)
        X = self.batch_norm11(X)

        return X


class Hybrid_Attention_Delta(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        '''
        定义多头注意力
        :param dim: 输入维度
        :param heads: 头的数量
        :param head_dim: 每个头的维度
        :param dropout:
        '''
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.to_qkvs1 = nn.Linear(dim, inner_dim * 4, bias=False)
        self.to_qkvs2 = nn.Linear(dim, inner_dim * 4, bias=False)
        self.to_out1 = nn.Sequential(
            nn.Linear(inner_dim, dim)
        )
        self.to_out2 = nn.Sequential(
            nn.Linear(inner_dim, dim)
        )
        self.SG = nn.Sigmoid()

    def forward(self, x_1, x_2):
        b, n, _ = x_1.shape
        h = self.heads
        conv1d1 = torch.nn.Conv1d(in_channels=n, out_channels=n, kernel_size=(5),
                                  padding=int(5 / 2), groups=n, bias=True).to(device)
        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkvs_1 = self.to_qkvs1(x_1).chunk(4, dim=-1)
        qkvs_2 = self.to_qkvs2(x_2).chunk(4, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q_1, k_1, v_1, s_1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkvs_1)
        q_2, k_2, v_2, s_2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkvs_2)

        dots_1 = torch.einsum('bhid,bhjd->bhij', q_1, k_1) * self.scale
        dots_2 = torch.einsum('bhid,bhjd->bhij', q_2, k_2) * self.scale

        MS = s_1 + s_2
        MS = rearrange(MS, 'b h n d -> b n (h d)')
        A = conv1d1(MS)
        A = self.SG(A) * MS
        MS = rearrange(A, 'b n (h d) -> b h n d', h=h)

        dots_3 = torch.einsum('bhid,bhjd->bhij', MS, MS) * self.scale

        attn_1 = dots_1.softmax(dim=-1)

        attn_2 = dots_2.softmax(dim=-1)

        attn_3 = dots_3.softmax(dim=-1)

        attn11 = torch.einsum('bhij,bhjd->bhid', attn_3, v_1) + v_1
        attn11 = torch.einsum('bhij,bhjd->bhid', attn_1, attn11)
        attn21 = torch.einsum('bhij,bhjd->bhid', attn_3, v_2) + v_2
        attn21 = torch.einsum('bhij,bhjd->bhid', attn_2, attn21)


        out_1 = rearrange(attn11, 'b h n d -> b n (h d)')
        out_2 = rearrange(attn21, 'b h n d -> b n (h d)')



        out1 = self.to_out1(out_1)
        out2 = self.to_out2(out_2)

        return out1, out2


class Transformer_inter(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_dual(LayerNormalize_dual(dim, Hybrid_Attention_Delta(dim=dim, heads=heads,
                                                                              head_dim=head_dim,
                                                                          dropout=dropout))),
                Residual_dual(LayerNormalize_dual(dim, MLP_Block_Dual(dim, mlp_dim)))
            ]))
    def forward(self, x1, x2):
        for dual_attention, dual_mlp in self.layers:
            x1, x2 = dual_attention(x1, x2)  # go to attention
            x1, x2 = dual_mlp(x1, x2)
        return x1, x2


class SiT_Model(nn.Module):
    def __init__(self, *, image_size, patch_size, WindowSize2, dim, depth, heads, head_dim, mlp_dim,
                 channels1, channels2, dropout, emb_dropout, num_classes):
        '''

        :param image_size: 输入图像空间尺寸
        :param patch_size: VIT融合尺寸
        :param dim: token的维度, 这里两个输入不同
        :param depth: 几个transformer模块
        :param heads: 头数量
        :param head_dim: 头的维度
        :param mlp_dim: 全连接层输出维度
        :param channels: 输入图像原始维度, 这里两个输入不同
        :param dropout:
        :param emb_dropout: 嵌入丢弃
        :param num_classes: 类别数
        '''
        super().__init__()

        assert image_size % patch_size == 0, '报错：图像没有被patch_size完美分割'
        num_patches_spe = (image_size // patch_size) ** 2
        num_patches_spa = (WindowSize2 // patch_size) ** 2
        patch_dim1 = channels1 * patch_size ** 2  # (P**2 C)：一个patch展平为向量后每个Token的维度
        patch_dim2 = channels2 * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embeddingSpe1 = nn.Parameter(torch.randn(1, num_patches_spe + 1, dim))  # 位置编码, +1是为了适应cls_token, Spectral
        # self.pos_embeddingSpe2 = nn.Parameter(torch.randn(1, num_patches_spe + 1, dim))  # 位置编码, +1是为了适应cls_token, Spectral
        self.pos_embeddingSpa1 = nn.Parameter(torch.randn(1, num_patches_spa + 1, dim))  # 位置编码, Spatial
        # self.pos_embeddingSpa2 = nn.Parameter(torch.randn(1, num_patches_spa + 1, dim))  # 位置编码, Spatial

        self.patch_to_embeddingSpe1 = nn.Linear(patch_dim1, dim)  # 将patch_dim（原图）经过embedding后得到dim维的嵌入向量, Spectral
        self.patch_to_embeddingSpe2 = nn.Linear(patch_dim1, dim)  # 将patch_dim（原图）经过embedding后得到dim维的嵌入向量, Spectral
        self.patch_to_embeddingSpa1 = nn.Linear(patch_dim2, dim)  # 将patch_dim（原图）经过embedding后得到dim维的嵌入向量, Spatial
        self.patch_to_embeddingSpa2 = nn.Linear(patch_dim2, dim)  # 将patch_dim（原图）经过embedding后得到dim维的嵌入向量, Spatial

        self.cls_token_1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_2 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_3 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_4 = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer1 = Transformer_inter(dim, depth, heads, head_dim, mlp_dim, dropout)
        self.transformer2 = Transformer_inter(dim, depth, heads, head_dim, mlp_dim, dropout)

        self.mlp_head_Spe = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes, bias=True),
        )
        self.mlp_head_Spa = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes, bias=True),
        )
        self.mlp_head_T = nn.Sequential(
            nn.Linear(2 * dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes, bias=True),
        )

        self.Conv11 = nn.Sequential(Conv2D(channels1, channels1, (1, 0), (3, 1), (1, 1), G=channels1),
                                    Conv2D(channels1, channels1, (0, 0), (1, 1), (1, 1), G=1))

        self.Conv21 = nn.Sequential(Conv2D(channels1, channels1, (0, 1), (1, 3), (1, 1), G=channels1),
                                    Conv2D(channels1, channels1, (0, 0), (1, 1), (1, 1), G=1))

        self.Conv31 = nn.Sequential(Conv2D(channels2, channels2, (0, 0), (1, 1), (1, 1), G=channels2),
                                    Conv2D(channels2, channels2, (0, 0), (1, 1), (1, 1), G=1))

        self.Conv41 = nn.Sequential(Conv2D(channels2, channels2, (1, 1), (3, 3), (1, 1), G=channels2),
                                    Conv2D(channels2, channels2, (0, 0), (1, 1), (1, 1), G=1))


    def forward(self, img1, img2):
        p = self.patch_size

        x1_1 = self.Conv11(img1)
        x1_2 = self.Conv21(img1)

        x2_1 = self.Conv31(img2)
        x2_2 = self.Conv41(img2)

        X1_1 = rearrange(x1_1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)  # 将H W C 转化成 N (P P C)
        X1_2 = rearrange(x1_2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)  # 将H W C 转化成 N (P P C)

        X1_1 = self.patch_to_embeddingSpe1(X1_1)  # 将(PPC)通过Embedding转化成一维embedding，这里的patch_to_embedding
        X1_2 = self.patch_to_embeddingSpe2(X1_2)  # 将(PPC)通过Embedding转化成一维embedding，这里的patch_to_embedding

        X2_1 = rearrange(x2_1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)  # 将H W C 转化成 N (P P C)
        X2_2 = rearrange(x2_2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)  # 将H W C 转化成 N (P P C)
        X2_1 = self.patch_to_embeddingSpa1(X2_1)  # 将(PPC)通过Embedding转化成一维embedding，这里的patch_to_embedding
        X2_2 = self.patch_to_embeddingSpa2(X2_2)  # 将(PPC)通过Embedding转化成一维embedding，这里的patch_to_embedding


        cls_tokens1 = self.cls_token_1.expand(img1.shape[0], -1, -1)
        cls_tokens2 = self.cls_token_2.expand(img1.shape[0], -1, -1)
        cls_tokens3 = self.cls_token_3.expand(img2.shape[0], -1, -1)
        cls_tokens4 = self.cls_token_4.expand(img2.shape[0], -1, -1)

        X1_1 = torch.cat((cls_tokens1, X1_1), dim=1)  # 将类别信息接入embedding
        X1_2 = torch.cat((cls_tokens2, X1_2), dim=1)  # 将类别信息接入embedding

        X2_1 = torch.cat((cls_tokens3, X2_1), dim=1)  # 将类别信息接入embedding
        X2_2 = torch.cat((cls_tokens4, X2_2), dim=1)  # 将类别信息接入embedding

        X1_1 = X1_1 + self.pos_embeddingSpe1
        X1_2 = X1_2 + self.pos_embeddingSpe1

        X2_1 = X2_1 + self.pos_embeddingSpa1
        X2_2 = X2_2 + self.pos_embeddingSpa1

        X1_1, X1_2 = self.transformer1(X1_1, X1_2)
        X2_1, X2_2 = self.transformer2(X2_1, X2_2)



        X1_1 = X1_1[:, 0]  # 取出class对应的token，用Identity占位
        X1_2 = X1_2[:, 0]  # 取出class对应的token，用Identity占位

        X2_1 = X2_1[:, 0]  # 取出class对应的token，用Identity占位
        X2_2 = X2_2[:, 0]  # 取出class对应的token，用Identity占位

        x_spe = X1_1 + X1_2
        x_spa = X2_1 + X2_2

        x = torch.cat((x_spe, x_spa), dim=1)

        classification_r1 = self.mlp_head_Spe(x_spe)
        classification_r2 = self.mlp_head_Spa(x_spa)
        classification_result = self.mlp_head_T(x)


        return classification_r1, classification_r2, classification_result
